from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
client = OpenAI()
import json
import numpy as np

PATH_CHUNKS = 'C:/Users/diazr/VSCODE/RAG/project_4/chunks.json'
with open(PATH_CHUNKS, 'r') as f:
    chunks_metadata = json.load(f)

PATH_QUERY = 'C:/Users/diazr/VSCODE/RAG/project_4/query.json'
with open(PATH_QUERY, 'r') as f:
    queries_metadata = json.load(f)


def embedding(text:str) :
    response = client.embeddings.create(
        model = 'text-embedding-3-small', 
        input = text
    )
    return response.data[0].embedding

# VECTOR DATABASE
embeddings_chunk = []
for chunk in chunks_metadata:
    embedding_chunk = embedding(chunk.get('text'))
    embeddings_chunk.append(embedding_chunk)

embeddings_chunk = np.array(embeddings_chunk)
np.save('C:/Users/diazr/VSCODE/RAG/project_4/chunks.npy', embeddings_chunk)

# VECTOR QUERY
embeddings_query = []
for query in queries_metadata:
    embedding_query = embedding(query.get('query'))
    embeddings_query.append(embedding_query)

embeddings_query = np.array(embeddings_query)
np.save('C:/Users/diazr/VSCODE/RAG/project_4/queries.npy', embeddings_query)

