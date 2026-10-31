from httpx2 import query
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
import json

from torch import rand_like


# NOTE Model rerangking/ cross encoder
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')

# NOTE - DATABASE VECTOR
PATH_EMBEDDING = 'C:/Users/diazr/VSCODE/RAG/project_2/chunk_overlap.npy'
DATABASE_VECTOR = np.load(PATH_EMBEDDING)
# NOTE - DATABASE CHUNK
with open('C:/Users/diazr/VSCODE/RAG/project_2/chunk_overlap.json', 'r') as f:
    DATABASE_CHUNK = json.load(f)

# NOTE - QUERY VECTOR
PATH_QUERY = 'C:/Users/diazr/VSCODE/RAG/query_embedding.npy'
QUERY_VECTOR = np.load(PATH_QUERY)
# NOTE - QUERY TEXT
with open('ground_truth.json', 'r') as f:
    QUERY_TEXT = json.load(f)

# AMBIL QUERY PERTAMA SAJA
query_vector_1 = QUERY_VECTOR[0]
query_vector_1 = query_vector_1.reshape(1, -1)
query_text_1 = QUERY_TEXT[0]
query_text_1 = query_text_1.get('query')

# FUNGSI RETRIEVAL NOTE
def top_similarity(query, vector_database, top_k):
    """ This function use for top k similarity of a query"""
    similarity = cosine_similarity(query, vector_database)

    top_indices = np.argsort(similarity[0])[-top_k:][::-1]
    top_scores = similarity[0][top_indices]

    # (array([190, 187, 195]), array([0.55558981, 0.54651483, 0.52206762]))
    return top_indices, top_scores


def cross_encoder(query, passages):
    ranks = model.rank(query=query, documents=passages)
    return ranks

def pipeline_rag_reranking(query_text, query_vector, vector_database, top_k=3):
    # NOTE - Stage 1 retrieval
    top_indices, top_scores = top_similarity(query_vector, vector_database, top_k)
    top_indices = list(top_indices)

    chunks = []
    for i in top_indices:
        top_chunk_metadata = DATABASE_CHUNK[i]
        chunk = top_chunk_metadata.get('text')
        chunks.append(chunk)
    
    # NOTE - Stage 2 reranking
    ranks = cross_encoder(query_text, chunks)
    return ranks

print(pipeline_rag_reranking(query_text_1, query_vector_1, DATABASE_VECTOR))

