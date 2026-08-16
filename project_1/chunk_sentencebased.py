import nltk
nltk.download('punkt_tab')
import json
from pathlib import Path


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

def sentence_chunk(text, sentences_per_chunk=3):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i+sentences_per_chunk])
        chunks.append(chunk)
    return sentences # list of chunk

chunk_metadata = []
for path in DATASET_PATH :
    doc_id = Path(path).stem
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    list_of_chunk = sentence_chunk(text)

    for chunk_id, chunk in enumerate(list_of_chunk):
        chunk_metadata.append({
            "doc_id" : doc_id,
            "chunk_id" : chunk_id,
            "text" : chunk,
        })

with open('project_1/chunk_sentencebased.json', 'w') as f:
    json.dump(chunk_metadata, f)