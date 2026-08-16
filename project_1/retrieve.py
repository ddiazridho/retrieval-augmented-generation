from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import json
load_dotenv()
client = OpenAI()
import os

GROUND_TRUTHS ='C:/Users/diazr/VSCODE/RAG/ground_truth.json'

# (304, 1536)
EMB_CHUNK_OVERLAP = 'C:/Users/diazr/VSCODE/RAG/project_1/chunk_overlap.npy'
# (600, 1536)
EMB_CHUNK_SENTENCE = 'C:/Users/diazr/VSCODE/RAG/project_1/chunk_sentencebased.npy'
# (686, 1536)
EMB_CHUNK_RECURSIVE = 'C:/Users/diazr/VSCODE/RAG/project_1/chunk_recursive.npy'

DATASET_EMB = [EMB_CHUNK_OVERLAP, EMB_CHUNK_RECURSIVE, EMB_CHUNK_SENTENCE]

def get_embedding(text:str) -> list:
    response = client.embeddings.create(
        model='text-embedding-3-small',
        input=text
    )
    result = response.data[0].embedding
    return result

with open(GROUND_TRUTHS, 'r') as f:
    ground_truths = json.load(f)

query_embeddings = []
for truth in ground_truths:
    query = truth.get('query')
    query_embedding = get_embedding(query)
    query_embeddings.append(query_embedding)

for dataset in DATASET_EMB:
    chunk_embeddings = np.load(dataset)
    similarity = cosine_similarity(query_embeddings, chunk_embeddings)

    # top k similarity
    k = 3
    top_k_indices_sklearn = np.argsort(similarity, axis=1)[:, -k:]

    # save
    filename = os.path.splitext(dataset)[0]
    filename = filename.replace("chunk_", "top_k_indices_chunk_") + ".npy"
    filename = filename
    np.save(filename, top_k_indices_sklearn)
