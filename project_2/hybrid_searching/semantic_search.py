from math import cos

from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np

query = np.load('C:/Users/diazr/VSCODE/RAG/query_embedding.npy')
query_pertama = query[0].reshape(1,-1)
document_emb = np.load('C:/Users/diazr/VSCODE/RAG/project_0/embedding_chunk.npy')

class SemanticSearch:
    def __init__ (self, document_embeddings: np.ndarray):
        self.document_embeddings = document_embeddings

    def search(self, query_embedding, top_n=3):

        # similarity query dengan document
        similarities = cosine_similarity(self.document_embeddings, query_embedding)

        # ranking
        ranked_indices = np.argsort(similarities, axis=0)[::-1].flatten()[:top_n]
        ranked_results = [(similarities[idx, 0], idx) for idx in ranked_indices]

        # (similarity, index docs tertinggi)
        # [(np.float64(0.5555898119749583), np.int64(153)), (np.float64(0.5220676195399596), np.int64(157))]
        return ranked_results

test = SemanticSearch(document_embeddings=document_emb)
hasil = test.search(query_pertama, top_n=2)

print(hasil)