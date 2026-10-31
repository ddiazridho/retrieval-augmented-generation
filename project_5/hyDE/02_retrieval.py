from math import hypot
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# hypothesis answer hasil query 14 bariss
HYPOTHESIS_EMB_PATH = 'C:/Users/diazr/VSCODE/RAG/project_5/hypothesis_emb.npy'
hypothesis_emb = np.load(HYPOTHESIS_EMB_PATH)
HYPOTHESIS_ANSWER_JSON = 'C:/Users/diazr/VSCODE/RAG/project_5/hypothesis_answer.json'
with open(HYPOTHESIS_ANSWER_JSON, 'r') as f:
    hypothesis_answer_json = json.load(f)

# vector db 18 baris
VECTOR_DB_PATH = 'C:/Users/diazr/VSCODE/RAG/project_5/chunks.npy'
vector_db = np.load(VECTOR_DB_PATH)
CHUNKS_METADATA = 'C:/Users/diazr/VSCODE/RAG/project_5/chunks.json'
with open(CHUNKS_METADATA, 'r') as f:
    chunks_metadata = json.load(f)

# fungsi bekerja untuk 1 query
def similarity_search(query, vector_db, top_k = 2):
    similarity = cosine_similarity(query, vector_db)

    top_indices = np.argsort(similarity[0])[-top_k:][::-1]
    top_scores = similarity[0][top_indices]

    return top_indices, top_scores

# untuk field results di retrieval_results
results = []
# looping query embdding
for emb_query in hypothesis_emb:
    emb_query = emb_query.reshape(1, -1)

    top_indices, top_scores = similarity_search(emb_query, vector_db)

    # isi dari field results
    result_per_query = []
    for index, score in zip(top_indices, top_scores):
        result_per_query.append(
            {
                "doc_id" : chunks_metadata[index].get('doc_id'),
                "chunk_id" : chunks_metadata[index].get('chunk_id'),
                "text" : chunks_metadata[index].get('text'),
                "score" : float(score)
            }
        )
    results.append(result_per_query)

for result, result_metadata in zip(results, hypothesis_answer_json):
    result_metadata["results"] = result

with open('C:/Users/diazr/VSCODE/RAG/project_5/retrieval_results.json', 'w') as f:
    json.dump(hypothesis_answer_json, f, indent=4)


