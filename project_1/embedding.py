import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import json
import numpy as np
client = OpenAI()

CHUNK_OVERLAP = "C:/Users/diazr/VSCODE/RAG/project_1/chunk_overlap.json"
CHUNK_RECURSIVE = "C:/Users/diazr/VSCODE/RAG/project_1/chunk_recursive.json"
CHUNK_SENTENCE = "C:/Users/diazr/VSCODE/RAG/project_1/chunk_sentencebased.json"

DATASET_CHUNK = [CHUNK_OVERLAP, CHUNK_RECURSIVE, CHUNK_SENTENCE]

def get_embedding(text:str) -> list:
    response = client.embeddings.create(
        model= "text-embedding-3-small",
        input=text
    )
    result = response.data[0].embedding
    return result

for path in DATASET_CHUNK:
    with open(path, 'r') as f:
        chunks = json.load(f)

    embeddings = []
    for chunk in chunks:
        text = chunk.get('text')
        text_embedding = get_embedding(text)
        embeddings.append(text_embedding)
    embeddings_npy = np.array(embeddings)
    filename = os.path.splitext(path)[0]
    np.save(filename, embeddings_npy)