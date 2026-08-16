from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import json
load_dotenv()

client = OpenAI()
PATH_CHUNK = 'C:/Users/diazr/VSCODE/RAG/project_0/chunk.json'

def get_embedding(text:str) -> list:
    response= client.embeddings.create(
        model='text-embedding-3-small',
        input=text
    )
    result = response.data[0].embedding
    return result

with open(PATH_CHUNK, 'r') as f:
    # NOTE - len 245
    chunks = json.load(f)

embedding_list = []
for chunk in chunks:
    text = chunk.get('text')
    text_embedding = get_embedding(text=text)
    embedding_list.append(text_embedding)

embedding_npy = np.array(embedding_list)
np.save('embedding_chunk.npy', embedding_npy)




