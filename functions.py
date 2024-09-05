import os
import re
import json

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from retry import retry
from langchain_community.document_loaders import UnstructuredWordDocumentLoader	

import google.generativeai as genai
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.sql import select
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from docx import Document as DocxDocument
from langchain_community.document_loaders import UnstructuredWordDocumentLoader	
import pandas as pd
from collections import defaultdict
import re

# insert Gemini API key
# store as private environment variable if hosting publically
os.environ['GOOGLE_API_KEY'] = 'insert API-Key'

@retry((Exception), tries=3, delay=2, backoff=0)
def text_to_sql(query, history):
    '''
    Based on natural language query and chat history, generates an SQL query
    '''
    prompt_template = PromptTemplate(
        template = 
        '''
        **Database Schema:**

        Tables:
        * sample_3 (id: int, incident_type: text, severity: text, incident_title: text, agency_reported: text, system_affected: text, incident_datetime: timestamp, detection_datetime: timestamp, resolution_datetime: timestamp, incident_details: text, post_incident_inquiry: text)

        **Example Questions:**
        1. What are the common methods of resolution?
        2. Which incidents were resolved in less than 3 months?
        3. How many incidents took place in 2024?
        4. Which incidents took place in 2024?
        5. What are some incidents reported by Agency 1?
        6. What are some incidents of a database breach?
        7. Which agencies were the most commonly affected?
        8. What are the incidents involving the most commonly affected system?
        9. Which agencies changed their data management procedure after the incident?
        10. What are some common methods of resolution by Agency 7?
        11. What are some common solutions for Cybersecurity cases in the last year?
            11a. Can you show me the incidents that used these solutions
        12. Can you show the incidents that used review of security protocols and staff training?
        13. What are the root causes for all the incidents?
        14. What measures have been most effective in reducing incident occurrences for data security?
        15. How do response times (in days) for incidents compare across different agencies? 

        **Answer Formatting:**

        * For question 1: List all incidents with similar words in 'incident_details'.
        * For question 2: Table with all incidents where difference in 'detection_datetime' and 'resolution_datetime' is less than 3 hours.
        * For question 3: Single integer with a count of all incidents with 'detection_datetime' year 2024
        * For question 4: List all incidents with 'datetime' year 2024.
        * For question 5: Table with all incidents where 'agency_reported' is Agency 1
        * For question 6: Table with all incidents wnere 'incident_title', 'incident_details' or 'incident_type' contain similar words to 'database breach' such as 'hacking' or 'data leak'
        * For question 7: Table with all incidents from the agency with the highest count in 'agency_reported'
        * For question 8: Table with all incidents from the agency with the highest count in 'agency_reported'
        * For question 9: Table with all incidents where 'post_incident_inquiry' contain similar words to 'data management procedure' such as 'data handling' or 'process' or other similar words
        * For question 10: Table with all incidents where 'agency_reported' is Agency 7.
        * For question 11: Table with all incidents where 'incident_type' is Cybersecurity and interval between now and 'incident_datetime' is 1 year
            * For question 11a: Table with all incidents from question 11
        * For question 12: Table with all incidents where 'post_incident_inquiry' contain similar words to 'review of security protocols' and 'staff training' such as 'safety regulations', 're-evaluation', 'training', and more/
        * For question 13: Table with the 'incident_details' of all incidents
        * For question 14: Table with all incidents where 'incident_type' is Data Breach or CyberSecurity
        * For question 15: Table with all incidents, sorted by different agencies.
        
        **Question:** {query}

        **SQL Query:**

        Write a postgreSQL query to answer the given question.

        **Chat History:** {chat_history}

        **Answer:**

        Present the answer in the desired format based on the example and ALWAYS consider the chat history for context.
        When unsure of what data or query to answer with, simply provide a query to obtain the entire SQL table.
        Make sure you follow the schema and types given, and provide the query for postgreSQL.
        If answer includes any agency, always list each agency separately "Agency 1, Agency 2 and Agency 3" do not put it as "Agencies 1, 2, and 3"
        ''',
        input_variables = ['query', 'chat_history']
    )

    llm = GoogleGenerativeAI(temperature=0, model="gemini-1.5-pro-latest", api_key=os.getenv('GOOGLE_API_KEY'))

    chain = prompt_template | llm

    response = chain.invoke({"query": query, "chat_history": history})
    sql_query = response.strip()

    match = re.search(r'```sql\n(.*?)\n```', sql_query, re.DOTALL)
    if match:
        sql_query = match.group(1).strip()
    else:
        raise ValueError("Failed to extract SQL query from the response")
    
    return sql_query

def ans_to_qns(vector_context, context, query, history):
    '''
    Based on retrieved context from SQL and Vector database, as well as the original query and chat history, generates an answer to the query.
    '''
    prompt_template = PromptTemplate(
        
        template = 
        '''
        **Database Schema:**

        Tables:
        * sample_3 (id: int, incident_type: text, severity: text, incident_title: text, agency_reported: text, system_affected: text, incident_datetime: timestamp, detection_datetime: timestamp, resolution_datetime: timestamp, incident_details: text, post_incident_inquiry: text)

        **Example Questions:**
        1. What are the common methods of resolution?
        2. Which incidents were resolved in less than 3 months?
        3. How many incidents took place in 2024?
        4. Which incidents took place in 2024?
        5. What are some incidents reported by Agency 1?
        6. What are some incidents of a database breach?
        7. Which agencies were the most commonly affected?
        8. What are the incidents involving the most commonly affected system?
        9. Which agencies changed their data management procedure after the incident?
        10. What are some common methods of resolution by Agency 7?
        11. What are some common solutions for Cybersecurity cases in the last year?
            11a. Can you show me the incidents that used these solutions
        12. Can you show the incidents that used review of security protocols and staff training?
        13. What are the root causes for all the incidents?
        14. What measures have been most effective in reducing incident occurrences for data security?
        15. How do response times (in days) for incidents compare across different agencies? 

        **Answer Formatting:**

        * For question 1: Within the context, look for recurring methods of resolution in the 'incident_details', then summarise and shorten.
        * For question 2: Summarise and shorten the 'incident_details' and 'incident_title' of all incidents in the context.       
        * For question 3: Return the number given in context.
        * For question 4: Summarise and shorten the 'incident_details' and 'incident_title' of all incidents in the context.       
        * For question 5: Summarise and shorten the 'incident_details' and 'incident_title' of all incidents in the context.       
        * For question 6: Summarise and shorten the 'incident_details' and 'incident_title' of all incidents in the context.      
        * For question 7: Return the 'agency_reported' of the agencies in context.
        * For question 8: Summarise and shorten the 'incident_details' and 'incident_title' of all incidents in the context.   
        * For question 9: Summarise and shorten the 'incident_details' and 'incident_title' of all incidents in the context, where the 'post_incident_inquiry' does indeed mention a change in some form of data management.        
        * For question 10: Within the context, look for recurring methods of resolution in the 'incident_details', then summarise and shorten.
        * For question 11: Within the context, look for recurring methods of resolution within the 'incident_details', summarise and shorten them.
            * For question 11a: Within the context, based on the recurring methods in question 11, summarise the 'incident_details'
        * For question 12: Within the context, look for 'post_incident_inquiry' which methods reviewing security protocals or staff training, summarise and shorten the 'incident_details' of those incidents
        * For question 13: Within the context, look for root causes of incidents in the 'incident_details', summarise and shorten it.  
        * For question 14: Within the context, look at the solutions and measures in 'incident_types' that have led to the least future cases from the same agency or system. Summarise and shorten it.  
        * For question 15: Within the context, look at 'detection_datetime' and 'resolution_datetime' to detect response times based on different agencies.
        
        **Question:** {query}

        **Context:** {context}

        **Vector Context:** {vector_context}

        **Chat History:** {chat_history}

        **Answer:**

        Present the answer in the desired format based on the example and ALWAYS consider the chat history for context. If an Agency is mentioned, ALWAYS search in the incident title to ensure that the information is relevant.  
        Note that the context and vector context are separate and should bothe be considered to answer the question. 
        Make sure you consider ALL context AND vector context provided. Select only the information that is relevant to the query. 
        If answer includes any agency, always list each agency separately "Agency 1, Agency 2 and Agency 3" do not put it as "Agencies 1, 2, and 3"

        ''',
        input_variables = ['vector_context', 'context', 'query', 'chat_history']
    )

    llm = GoogleGenerativeAI(temperature=0, model="gemini-1.5-pro-latest", api_key=os.getenv('GOOGLE_API_KEY'))
    chain = prompt_template | llm

    response = chain.invoke({"vector_context": vector_context, "context": context, "query": query, "chat_history": history})

    return response

def incident_search(query):
    '''
    Given an incident title, writes an SQL query to retrieve all incident details related to specific incident
    '''
    prompt_template = PromptTemplate(
        template = 
        '''
        **Database Schema:**

        Tables:
        * sample_3 (id: int, incident_type: text, severity: text, incident_title: text, agency_reported: text, system_affected: text, incident_datetime: timestamp, detection_datetime: timestamp, resolution_datetime: timestamp, incident_details: text, post_incident_inquiry: text)

        **Question:** {query}

        **SQL Query:**

        Write a postgreSQL query to answer the given question.

        **Answer:**

        Based on a query, return all the details of the incident with the corresponding title. Do not truncate or shorten any details and provide it all.
        If answer includes any agency, always list each agency separately "Agency 1, Agency 2 and Agency 3" do not put it as "Agencies 1, 2, and 3"
        
        ''',
        input_variables = ['query']
    )

    llm = GoogleGenerativeAI(temperature=0, model="gemini-1.5-pro-latest", api_key=os.getenv('GOOGLE_API_KEY'))

    chain = prompt_template | llm

    response = chain.invoke({"query": query})
    sql_query = response.strip()

    match = re.search(r'```sql\n(.*?)\n```', sql_query, re.DOTALL)
    if match:
        sql_query = match.group(1).strip()
    else:
        raise ValueError("Failed to extract SQL query from the response")
    
    return sql_query


def get_embeddings(text):
   '''
   Obtain embeddings of given text using Google Gemini embedding model
   '''
   
    # insert Gemini API Key
    genai.configure(api_key='insert api-key')

    response = genai.embed_content(model = "models/embedding-001", content = text)

    return response
   

def sql_embedding_search(query_text):
    # insert username and password
    db_url = 'postgresql://username:password@localhost:5432/toy_incidents'
    engine = create_engine(db_url)
    metadata = MetaData()
    # modify with incident_table name
    incidents_table = Table('sample_4', metadata, autoload_with=engine)

    with engine.connect() as connection:

        query_embedding = [x for x in get_embeddings(query_text).values()][0]
        # modify based on different column titles of incident table
        query = select(incidents_table.c.id, incidents_table.c.incident_type, incidents_table.c.severity, incidents_table.c.incident_title, incidents_table.c.agency_reported, incidents_table.c.system_affected,incidents_table.c.incident_datetime, incidents_table.c.detection_datetime, incidents_table.c.resolution_datetime, incidents_table.c.incident_details, incidents_table.c.post_incident_inquiry, incidents_table.c.details_emb, incidents_table.c.post_emb)
        result = connection.execute(query)
        incidents = result.fetchall()

    details_emb = []
    post_emb = []
    # Calculate cosine similarities
    for incident in incidents:
        details = [json.loads(incident.details_emb)]
        details_emb = details_emb + details
        post = [json.loads(incident.post_emb)]
        post_emb = post_emb + post
    
    similarities_details = [(incident, cosine_similarity([query_embedding], details_emb)[0][0]) for incident in incidents]
    similarities_post = [(incident, cosine_similarity([query_embedding], post_emb)[0][0]) for incident in incidents]
    similarities_details.sort(key=lambda x: x[1], reverse=True)
    similarities_post.sort(key=lambda x: x[1], reverse=True)

    # Get the top 10 most similar incidents
    top_incidents_details = similarities_details[:10]
    top_incidents_post = similarities_post[:10]
    return top_incidents_details, top_incidents_post


def report_writing(context, type, closed, prompt):
    '''
    Based on incident type and status, retrieve relevant sample report to use as reference
    
    Modify path to report, or use folder if multiple reports required
    '''

    if type == "data_sec" and closed == True:
        samples = UnstructuredWordDocumentLoader('/data/documents/report.docx').load()[0]
        sample = samples.page_content

    if type == "cybersec" and closed == True:
        samples = UnstructuredWordDocumentLoader('/data/documents/report.docx').load()[0]
        sample = samples.page_content

    if type == "loss_of_equip" and closed == True:
        samples = UnstructuredWordDocumentLoader('/data/documents/report.docx').load()[0]
        sample = samples.page_content

    if type == "serv_avail" and closed == True:
        samples = UnstructuredWordDocumentLoader('/data/documents/report.docx').load()[0]
        sample = samples.page_content

    if type == "data_sec" and closed == False:
        samples = UnstructuredWordDocumentLoader('/data/documents/report.docx').load()[0]
        sample = samples.page_content

    if type == "cybersec" and closed == False:
        samples = UnstructuredWordDocumentLoader('/data/documents/report.docx').load()[0]
        sample = samples.page_content

    if type == "loss_of_equip" and closed == False:
        samples = UnstructuredWordDocumentLoader('/data/documents/report.docx').load()[0]
        sample = samples.page_content

    if type == "serv_avail" and closed == False:
        samples = UnstructuredWordDocumentLoader('/data/documents/report.docx').load()[0]
        sample = samples.page_content


    prompt_template = PromptTemplate( 
        template = 
        '''
        **Database Schema:**

        Tables:
        * sample_3 (id: int, incident_type: text, severity: text, incident_title: text, agency_reported: text, system_affected: text, incident_datetime: timestamp, detection_datetime: timestamp, resolution_datetime: timestamp, incident_details: text, post_incident_inquiry: text)
        
        **Context** {context}

        **Template** {sample}

        **Prompt** {prompt}
            
        **Answer**
        Given the incident in the context, write a situational report based on the given sample and the prompt given. Follow all the headings and include all information in the context. Note that you should only use information if the 'Incident Title' is the same as specified by the user.
        Do not truncate any details and ensure the information is provided in the same writing style as the sample given with proper grammar.
        If answer includes any agency, always list each agency separately "Agency 1, Agency 2 and Agency 3" do not put it as "Agencies 1, 2, and 3"

        ''',
        input_variables = ['context', 'sample', 'prompt']
    )
    
    llm = GoogleGenerativeAI(temperature=0, model="gemini-1.5-pro-latest", api_key=os.getenv('GOOGLE_API_KEY'))
    chain = prompt_template | llm

    response = chain.invoke({"context": context, "sample": sample, "prompt": prompt})

    return response


def organizer (docx_path):
    '''
    Extracts incident title and organizes incident into list based on agency names in incident title of document
    '''
    doc = DocxDocument(docx_path)
    current_incident_title = None
    agencies = defaultdict(list)

    for table in doc.tables:
        # Extract content
        content = "\n".join(["\t".join(cell.text.strip() for cell in row.cells) for row in table.rows])

        # Detect incident title
        if "INCIDENT TITLE" in content:
            title_match = r"INCIDENT TITLE:\s*(.*?)\s*INCIDENT TITLE:"
            match = re.search(title_match, content, re.DOTALL)
            current_incident_title = match.group(1).strip()
            incident_title = current_incident_title
            
            agencies = defaultdict(list)

            while "Agency " in current_incident_title:  # Assuming agency name is in the incident title
                agency = re.search(r'Agency\s+(\w+)', current_incident_title)
                agency_name = 'Agency ' + agency.group(1)  # Extract agency name
                agencies[agency_name].append(incident_title)

                search = agency_name
                current_incident_title = current_incident_title.replace(search, '')

    return agencies
    
        
