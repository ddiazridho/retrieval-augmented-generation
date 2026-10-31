import numpy as np
import json
import faiss

PATH_QUERY = 'C:/Users/diazr/VSCODE/RAG/query_embedding.npy'
PATH = 'project_2/chunk_overlap.npy'
embeddings = np.load(PATH)
query_vector = np.load(PATH_QUERY)

# INDEX FLAT IP (BRUTE FORCE), DOT PRODUCT
dim = embeddings.shape[1] #1536
index_flat = faiss.IndexFlatIP(dim) #object faiss
index_flat.add(embeddings) # mapping kamus embedding 

score, indices = index_flat.search(query_vector, k=3) # similarity search

print('Flat Index Results:')
print(indices)

# HNSW
dim = embeddings.shape[1]
index_hnsw = faiss.IndexHNSWFlat(dim, 32) # 32 koneksi tiap node
index_hnsw.add(embeddings)

index_hnsw.hnsw.efSearch = 16
score, indices = index_hnsw.search(query_vector, k=3)

print('HNSW Index Results:')
print(indices)

