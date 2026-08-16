from pathlib import Path
import json

PATH_1 = "C:/Users/diazr/VSCODE/RAG/project_0/data/hermes.md"
PATH_2 = "C:/Users/diazr/VSCODE/RAG/project_0/data/keras.md"
PATH_3 = "C:/Users/diazr/VSCODE/RAG/project_0/data/kubernetes.md"
PATH_4 = "C:/Users/diazr/VSCODE/RAG/project_0/data/langchain.md"
PATH_5 = "C:/Users/diazr/VSCODE/RAG/project_0/data/langgraph.md"
PATH_6 = "C:/Users/diazr/VSCODE/RAG/project_0/data/laravel.md"
PATH_7 = "C:/Users/diazr/VSCODE/RAG/project_0/data/pytorch.md"
PATH_8 = "C:/Users/diazr/VSCODE/RAG/project_0/data/redis.md"
PATH_9 = "C:/Users/diazr/VSCODE/RAG/project_0/data/tf.md"
PATH_10 = "C:/Users/diazr/VSCODE/RAG/project_0/data/vercel.md"

DATASET_PATH = [PATH_1, PATH_2, PATH_3, PATH_4, PATH_5, PATH_6, PATH_7, PATH_8, PATH_9, PATH_10]

chunk_metadata = []
chunk_size = 500
for path in DATASET_PATH:
    docs_id = Path(path).stem
    with open(path, mode='r', encoding='utf-8') as readme:
        text = readme.read()

    for chunk_id, start in enumerate(range(0, len(text), chunk_size)):
        chunk_text = text[start:start+chunk_size]

        chunk_metadata.append({
            "doc_id" : docs_id,
            "chunk_id" : chunk_id,
            "text" : chunk_text
        })

with open("project_0/chunk.json", 'w') as f:
    json.dump(chunk_metadata, f)