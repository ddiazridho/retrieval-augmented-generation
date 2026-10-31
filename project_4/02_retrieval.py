from httpx import get
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# NOTE - Chunks
PATH_CHUNKS = 'C:/Users/diazr/VSCODE/RAG/project_4/chunks.json'
with open(PATH_CHUNKS, 'r') as f:
    chunks_metadata = json.load(f)
PATH_CHUNKS_EMB = 'C:/Users/diazr/VSCODE/RAG/project_4/chunks.npy'
embeddings_chunks = np.load(PATH_CHUNKS_EMB)

# NOTE - Queries
PATH_QUERIES = 'C:/Users/diazr/VSCODE/RAG/project_4/query.json'
with open(PATH_QUERIES, 'r') as f:
    queries_metadata = json.load(f)
PATH_QUERIES_EMB = 'C:/Users/diazr/VSCODE/RAG/project_4/queries.npy'
embeddings_queries = np.load(PATH_QUERIES_EMB)

# NOTE - Retrieval result untuk menyimpan hasil top_chunks
PATH_RETRIEVAL_RESULTS = 'C:/Users/diazr/VSCODE/RAG/project_4/retrieval_results.json'
with open(PATH_RETRIEVAL_RESULTS, 'r') as f:
    results_metadata = json.load(f)


def similarity_search(query, vector_database, top_k=2):
    similarity = cosine_similarity(query, vector_database)

    top_indices = np.argsort(similarity[0])[-top_k:][::-1]
    top_scores = similarity[0][top_indices]

    return top_indices, top_scores

# untuk key results di retrieval results.json
results = []
# looping query 
for embedding_query in embeddings_queries:
    embedding_query = embedding_query.reshape(1,-1)

    top_indices, top_scores = similarity_search(embedding_query, embeddings_chunks)

    result_per_query = []
    # top index similarity
    for index, score in zip(top_indices, top_scores):
            # looping query metadata untuk save hasil retrieval
            result_per_query.append(
                 {  
                      'doc_id' : chunks_metadata[index].get('doc_id'),
                      'chunk_id' : chunks_metadata[index].get('chunk_id'),
                      'text' : chunks_metadata[index].get('text'),
                      'retrieval_score' : float(score)
                 }
            )
    results.append(result_per_query)

# looping key result dan query di metadata
for result, result_metadata in zip(results, results_metadata):
    result_metadata['results'] = result
                
with open(PATH_RETRIEVAL_RESULTS, 'w') as f:
    json.dump(results_metadata, f, indent=4)