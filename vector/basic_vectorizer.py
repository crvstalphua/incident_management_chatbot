import os
import shutil

from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import UnstructuredWordDocumentLoader	
from langchain_community.vectorstores import Chroma

from typing import Optional
from typing import List

from docx import Document as DocxDocument
from langchain_community.document_loaders import UnstructuredWordDocumentLoader	
import pandas as pd
from collections import defaultdict
import re

from langchain_text_splitters import CharacterTextSplitter

# Insert API Key
os.environ['GOOGLE_API_KEY'] = 'insert-api-key'


class CustomGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts: List[str], task_type: Optional[str] = None, titles: Optional[List[str]] = None, output_dimensionality: Optional[int] = None) -> List[List[float]]:
        embeddings_repeated = super().embed_documents(texts, task_type, titles, output_dimensionality)
        # convert proto.marshal.collections.repeated.Repeated to list
        embeddings = [list(emb) for emb in embeddings_repeated]
        return embeddings

gemini_embeddings_wrapper = CustomGoogleGenerativeAIEmbeddings(model="models/embedding-001")

#delete and create new db when another file is added (temporarily)

abs_path = Path().absolute()
directory = f'{abs_path}/data/documents/'
document_paths = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.docx')]
folder_path = f'{abs_path}/database'

def check_and_delete_folder(folder_path):
    if os.path.exists(folder_path):
        print(f"Folder '{folder_path}' exists. Deleting it.")
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' has been deleted.")
    else:
        return

def vectorize(document_paths):
    '''
    Perform chunking and splitting for documents given in document path and vectorizes them to be stored in database
    '''
    documents = []
    for filename in document_paths: 
        document_loader = UnstructuredWordDocumentLoader(filename)
        docs = document_loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(document)
    
        documents += (docs)
    print(f"Documents from '{document_paths}' loaded.")

    vectorstore = Chroma.from_documents(
                     documents=documents,  # Data
                     embedding=gemini_embeddings_wrapper,    # Embedding model
                     persist_directory="./database" # Directory to save data
                     )
    print(f"Vector database {abs_path}/database created.")
       
check_and_delete_folder(folder_path)
vectorize(document_paths)
