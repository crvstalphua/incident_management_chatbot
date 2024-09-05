import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import pandas as pd
from sqlalchemy import create_engine

'''
This file is only used to perform vector embeddings for certain columns in the SQL database, in order to perform vector similary search for columns with unstructured data
'''


# insert api key
os.environ['GOOGLE_API_KEY'] = 'insert api key'

# excel file with columns for embedding
df = pd.read_excel('data/incident_samples_2.xlsx')
new_df = []

def get_embeddings(text):
    '''
    Get embeddings for text using Google Gemini embedding model
    '''
    genai.configure(api_key='insert api key')

    response = genai.embed_content(model = "models/embedding-001", content = text)

    return response

# modify code based on number of columns to be embedded as well as names of columns
new_list = []
other_list = []
for i in range(len(df)):
    details = df['incident_details'][i]
    post = df['post_incident_inquiry'][i]
    details_embedding = get_embeddings(details)
    post_embedding = get_embeddings(post)
    new_list += details_embedding.items()[0]
    other_list += post_embedding.items()[0]
    
df['details_emb'] = new_list
df['post_emb'] = other_list

df.to_csv('data/incident_samples_w_embeddings.csv', index=False)

