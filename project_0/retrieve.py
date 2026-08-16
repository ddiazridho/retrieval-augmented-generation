from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()
import json
PATH_EMBEDDING = 'C:/Users/diazr/VSCODE/RAG/project_0/embedding_chunk.npy'
PATH_TRUTHS = 'C:/Users/diazr/VSCODE/RAG/project_0/ground_truth.json'

# NOTE load
embedding_npy = np.load(PATH_EMBEDDING)
with open(PATH_TRUTHS, 'r') as f:
    truths = json.load(f)

# NOTE Fungsi embedding
def get_embedding(text:str) -> list:
    response = client.embeddings.create(
        model='text-embedding-3-small',
        input=text
    )
    result = response.data[0].embedding
    return result

embeddings_query = []
# NOTE Retrieve
for truth in truths:
    query = truth.get("query")
    embedding_query = get_embedding(query)
    embeddings_query.append(embedding_query)

# NOTE cosine bikin sendiri
def cosine_manual(v1, v2):
    cos_sim_manual = np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )
    return cos_sim_manual

scores = [] # cosine
for query_vector in embeddings_query:
    query_scores = []

    for vector in embedding_npy:
        score = cosine_manual(query_vector, vector)
        query_scores.append(score)

    scores.append(query_scores)

# NOTE cosine sklearn
similarity = cosine_similarity(embeddings_query, embedding_npy)

# NOTE - TOP K, ambil indeks bukan nilai
k = 3

top_k_indices_sklearn = np.argsort(similarity, axis=1)[:, -k:]
top_k_indices_manual = np.argsort(scores, axis=1)[:, -k:]

np.save('top_indices_sklearn.npy', top_k_indices_sklearn)
np.save('top_indices_manual.npy', top_k_indices_manual)

# ANCHOR - Result dimensi (10,3)