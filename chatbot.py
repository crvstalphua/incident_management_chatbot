import os
import streamlit as st

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from sqlalchemy import create_engine, text
from langchain_community.vectorstores import Chroma
from pathlib import Path


from collections import defaultdict

from typing import Optional
from typing import List

from functions import text_to_sql, ans_to_qns, report_writing, incident_search, organizer, sql_embedding_search

from data_cleaner import encrypt, decrypt

# Insert Gemini API Key
# If hosting publically, store as private environment variable
os.environ['GOOGLE_API_KEY'] = 'insert-api-key'

class CustomGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts: List[str], task_type: Optional[str] = None, titles: Optional[List[str]] = None, output_dimensionality: Optional[int] = None) -> List[List[float]]:
        embeddings_repeated = super().embed_documents(texts, task_type, titles, output_dimensionality)
        # convert proto.marshal.collections.repeated.Repeated to list
        embeddings = [list(emb) for emb in embeddings_repeated]
        return embeddings
gemini_embeddings_wrapper = CustomGoogleGenerativeAIEmbeddings(model="models/embedding-001")


vectorstore_disk = Chroma(
                        persist_directory="./database",       # Directory of db
                        embedding_function=gemini_embeddings_wrapper   # Embedding model
                   )

retriever = vectorstore_disk.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5})

abs_path = Path().absolute()
directory = f'{abs_path}/data/documents/'
document_paths = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.docx')]

# creates dictionary and stores incidents based on Agency found in title
agencies = defaultdict(list)
for filename in document_paths: 
    agency_list = organizer(filename)
    for agency, titles in agency_list.items():
        agencies[agency].extend(titles)

def handle_user_input(prompt):
    '''
    Generates an answer to the user query using data retrieved from the SQL database and Chroma database
    '''
    if st.session_state.conversation:
        st.session_state.chatbot_new = False
        st.session_state.conversation.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # encryption of prompt
        prompt = encrypt(prompt)

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            
            # retrieves from SQL database through text-to-sql
            sql_query = text_to_sql(prompt, st.session_state.chat_history)
            print(f'Generated SQL Query:  {sql_query}' + '\n' + '---------------------------------------------------------------------------------------' + '\n')

            # insert username and password of SQL database
            db_url = 'postgresql://username:password@localhost:5432/toy_incidents'
            engine = create_engine(db_url)

            with engine.connect() as connection:

                result = connection.execute(text(sql_query))
                context = result.fetchall()
                for row in context:
                    print(row)

                # retrieves from SQL database through vector embedding similarity search
                sql_vector_details, sql_vector_past = sql_embedding_search(prompt)
                
                print('\n' + '-----------------------------------------------------------------------------')
                print(sql_vector_details)
                print('\n' + '-----------------------------------------------------------------------------')
                print(sql_vector_past)

                context = context + sql_vector_details + sql_vector_past

                print('\n' + '-----------------------------------------------------------------------------')

                # if Agency name is in title, retrieves relevant incident titles from dictionary to be used for vector similarity search
                agency_search = ''
                if "Agency " in prompt:  # Assuming agency name is in the incident title
                    agency_name = 'Agency ' + prompt.split("Agency ")[1].split()[0]
                    agency_search = agency_list[agency_name]

                # retrieves from Chroma database through vector embedding similarity search
                vector_context = retriever.invoke(prompt + str(agency_search))
                print(vector_context)

                # generates answer based on input prompt, retrieved context and chat history
                response = ans_to_qns(vector_context, context, prompt, st.session_state.chat_history)
    
                # decryption of response
                st.session_state.conversation.append({"role": "assistant", "content": decrypt(response)})
                st.session_state.chat_history.append((prompt, response))
                st.write(decrypt(response))
                print('\n' + '-----------------------------------------------------------------------------' + '\n' + response)


def handle_report_request(prompt, type, closed):
    '''
    Generates a report given an incident title, type and status, after retrieving relevant incident information from the databases
    
    Utilizes past reports of various types and statuses as reference
    '''
    if st.session_state.conversation:
        st.session_state.conversation.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # encryption of prompt
        prompt = encrypt(prompt)
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            
            # retrieves incident details from SQL database, using the incident title
            sql_query = incident_search(prompt)
            print(f'Generated SQL Query:  {sql_query}' + '\n' + '---------------------------------------------------------------------------------------' + '\n')

            # insert username and password of SQL database
            db_url = 'postgresql://username:password@localhost:5432/toy_incidents'
            engine = create_engine(db_url)

            with engine.connect() as connection:
        
                result = connection.execute(text(sql_query))
                context = result.fetchall()
                for row in context:
                    print(row)

                print('\n' + '-----------------------------------------------------------------------------')

                # retrieves incident details from vector database, using the incident title
                vector_context = retriever.invoke(prompt)
                print(vector_context)

                context = vector_context + context
                
                # generates a report based on the incident details, type, status and title
                response = report_writing(context, type, closed, prompt)
                
                # decryption of prompt
                st.session_state.conversation.append({"role": "assistant", "content": decrypt(response)})
                st.session_state.chat_history.append((prompt, response))
                st.write(decrypt(response))
                print('\n' + '-----------------------------------------------------------------------------' + '\n' + response)


'''
Chatbot and Report Writer States

To be used with streamlit button clicks for switching between various interfaces
'''

def chatbot():
    st.session_state.chatbot = True
    st.session_state.chatbot_new = True

def writer():
    st.session_state.writer = True
    st.session_state.cybersec = False
    st.session_state.data_sec = False
    st.session_state.serv_avail = False
    st.session_state.loss_of_equip = False
    st.session_state.closed = False
    st.session_state.open = False

def cybersec():
    st.session_state.cybersec = True
    st.session_state.data_sec = False
    st.session_state.serv_avail = False
    st.session_state.loss_of_equip = False

def data_sec():
    st.session_state.cybersec = False
    st.session_state.data_sec = True
    st.session_state.serv_avail = False
    st.session_state.loss_of_equip = False

def serv_avail():
    st.session_state.cybersec = False
    st.session_state.data_sec = False
    st.session_state.serv_avail = True
    st.session_state.loss_of_equip = False

def loss_of_equip():
    st.session_state.cybersec = False
    st.session_state.data_sec = False
    st.session_state.serv_avail = False
    st.session_state.loss_of_equip = True

def closed():
    st.session_state.closed = True
    st.session_state.open = False

def open():
    st.session_state.closed = False
    st.session_state.open = True

'''
Chatbot and Report Writer Main Function

Utilizes various if states to transition between different streamlit interfaces based on user selection
'''


def main():
    st.set_page_config(page_title="Incident Query Chatbot", page_icon=":chatbot:")
    if "conversation" not in st.session_state or st.sidebar.button("Return to homescreen"):
        st.session_state.conversation = [{"role": "assistant", "content": "Please select your mode."}]
        st.session_state.chat_history= []
        st.session_state.chatbot = False
        st.session_state.writer = False
        for msg in st.session_state.conversation:
            st.chat_message(msg["role"]).write(msg["content"])
    
    if st.session_state.chatbot == False and st.session_state.writer == False:
        st.sidebar.title("Incident Query Helper")
        st.sidebar.markdown('''
            This app is an LLM powered helper that answers questions based on an incident database. 
            \n
            Please select a function before you can proceed.
            ''')
        
        st.sidebar.button('Chatbot', on_click = chatbot)
            
        st.sidebar.button('Report Writer', on_click = writer)
    
    if st.session_state.chatbot:
        st.sidebar.title("Incident Query Chatbot 💬")
        st.sidebar.markdown('''
            This app is an LLM powered Chatbot that answers questions based on an incident database. 
            \n
            Here are some commonly asked questions.
            ''')
        if st.session_state.chatbot_new == True or st.sidebar.button("Clear conversation history"):
            st.session_state.conversation = [{"role": "assistant", "content": "What is your query?"}]
            st.session_state.chat_history= []
        
        for msg in st.session_state.conversation:
            st.chat_message(msg["role"]).write(msg["content"])
            
        if st.sidebar.button('Can you identify any recurring vulnerabilities that have been exploited in past incidents?'):
            handle_user_input('Can you identify any recurring vulnerabilities that have been exploited in past incidents?')
        
        if user_question:= st.chat_input('Ask your question'):
            handle_user_input(user_question)

    if st.session_state.writer:
        st.sidebar.title("Incident Report Writer 💬")
        st.sidebar.markdown('''
            This app is an LLM powered Report Writer that can help draft reports based on incident details. 
            \n
            Before proceeding, please select which the case type, followed by report type.
            ''')
        
        if (st.session_state.cybersec == False and st.session_state.data_sec == False and st.session_state.serv_avail == False and st.session_state.loss_of_equip == False) or st.sidebar.button("Clear conversation history", on_click = writer):
            st.session_state.conversation = [{"role": "assistant", "content": "What report would you like me to write?"}]
            st.session_state.chat_history= []
            for msg in st.session_state.conversation:
                st.chat_message(msg["role"]).write(msg["content"])

        st.sidebar.subheader('Incident Type')

        st.sidebar.button('Data Security', on_click = data_sec)
        st.sidebar.button('Cybersecurity', on_click = cybersec)
        st.sidebar.button('Loss of Equipment', on_click = loss_of_equip)
        st.sidebar.button('Service Availability', on_click = serv_avail)

        st.sidebar.subheader('Incident Status')

        st.sidebar.button('Closed Incident', on_click = closed)
        st.sidebar.button('New incident', on_click = open)
    

        if st.session_state.data_sec == True:
            st.header('Date Security SITREP')
            if st.session_state.closed == True:
                st.subheader('Closed Incident')

            if st.session_state.open == True:
                st.subheader('Open Incident')

            for msg in st.session_state.conversation:
                st.chat_message(msg["role"]).write(msg["content"])

            st.session_state.conversation = [{"role": "assistant", "content": "What report would you like me to write?"}]
            st.session_state.chat_history= []
            if report_details := st.chat_input('Provide your report details:'):
                if st.session_state.closed == False:
                    handle_report_request(report_details, 'data_sec', False)
                else:
                    handle_report_request(report_details, 'data_sec', True)

        if st.session_state.cybersec == True:
            st.header('CyberSecurity SITREP')
            if st.session_state.closed == True:
                st.subheader('Closed Incident')

            if st.session_state.open == True:
                st.subheader('Open Incident')

            for msg in st.session_state.conversation:
                st.chat_message(msg["role"]).write(msg["content"])

            st.session_state.conversation = [{"role": "assistant", "content": "What report would you like me to write?"}]
            st.session_state.chat_history= []
            if report_details := st.chat_input('Provide your report details:'):
                if st.session_state.closed == False:
                    handle_report_request(report_details, 'cybersec', False)

                else:
                    handle_report_request(report_details, 'cybersec', True)

        if st.session_state.serv_avail == True:
            st.header('Service Availability SITREP')
            if st.session_state.closed == True:
                st.subheader('Closed Incident')

            if st.session_state.open == True:
                st.subheader('Open Incident')

            for msg in st.session_state.conversation:
                st.chat_message(msg["role"]).write(msg["content"])

            st.session_state.conversation = [{"role": "assistant", "content": "What report would you like me to write?"}]
            st.session_state.chat_history= []
            if report_details := st.chat_input('Provide your report details:'):
                if st.session_state.closed == False:
                    handle_report_request(report_details, 'serv_avail', False)

                else:
                    handle_report_request(report_details, 'serv_avail', True)

        if st.session_state.loss_of_equip == True:
            st.header('Loss of Equipment SITREP')
            if st.session_state.closed == True:
                st.subheader('Closed Incident')

            if st.session_state.open == True:
                st.subheader('Open Incident')

            for msg in st.session_state.conversation:
                st.chat_message(msg["role"]).write(msg["content"])

            st.session_state.conversation = [{"role": "assistant", "content": "What report would you like me to write?"}]
            st.session_state.chat_history= []
            if report_details := st.chat_input('Provide your report details:'):
                if st.session_state.closed == False:
                    handle_report_request(report_details, 'loss_of_equip', False)

                else:
                    handle_report_request(report_details, 'loss_of_equip', True)

if __name__ == '__main__':
    main()
