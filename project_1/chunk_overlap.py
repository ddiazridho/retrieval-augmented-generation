from pathlib import Path
import json

PATH_1 = "C:/Users/diazr/VSCODE/RAG/data/hermes.md"
PATH_2 = "C:/Users/diazr/VSCODE/RAG/data/keras.md"
PATH_3 = "C:/Users/diazr/VSCODE/RAG/data/kubernetes.md"
PATH_4 = "C:/Users/diazr/VSCODE/RAG/data/langchain.md"
PATH_5 = "C:/Users/diazr/VSCODE/RAG/data/langgraph.md"
PATH_6 = "C:/Users/diazr/VSCODE/RAG/data/laravel.md"
PATH_7 = "C:/Users/diazr/VSCODE/RAG/data/pytorch.md"
PATH_8 = "C:/Users/diazr/VSCODE/RAG/data/redis.md"
PATH_9 = "C:/Users/diazr/VSCODE/RAG/data/tf.md"
PATH_10 = "C:/Users/diazr/VSCODE/RAG/data/vercel.md"

DATASET_PATH = [PATH_1, PATH_2, PATH_3, PATH_4, PATH_5, PATH_6, PATH_7, PATH_8, PATH_9, PATH_10]

chunk_metadata = []
chunk_size = 500
overlap = 100

for path in DATASET_PATH:
    doc_id = Path(path).stem
    with open(path, mode='r', encoding='utf-8') as readme:
        text = readme.read()

    i = 0
    for start in range(0, len(text), chunk_size - overlap):
        chunk_text = text[start:start+chunk_size]

        chunk_metadata.append({
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_chunk_{i}",
            "text": chunk_text
        })
        i += 1

with open("project_1/chunk_overlap.json", 'w', encoding='utf-8') as f:
    json.dump(chunk_metadata, f)