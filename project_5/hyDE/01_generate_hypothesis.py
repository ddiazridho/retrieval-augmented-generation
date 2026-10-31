from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
client = OpenAI()
import numpy as np
import json


def hypotesis_answer(query:str) -> str:
    prompt = f"""Tulis satu paragraf dokumentasi teknis yang menjawab pertanyaan berikut. 
                Tulis dengan gaya seperti dokumentasi resmi (formal, langsung ke fakta), 
                BUKAN seperti jawaban chat atau percakapan. Jangan awali dengan "Jawabannya adalah" atau semacamnya.

                Pertanyaan: {query}

                Dokumentasi:"""
    response = client.responses.create(
        model='gpt-4o-mini',
        input=prompt,
        temperature=0.7,
        max_output_tokens=200
    )
    return response.output_text

def hypotesis_embedding(text:str) -> list[float]:
    response = client.embeddings.create(
        input=text,
        model='text-embedding-3-small'
    )
    # outputnya vector (dim,)
    return response.data[0].embedding

    
with open('C:/Users/diazr/VSCODE/RAG/project_5/query.json', 'r') as f:
    query_metadata = json.load(f)

# loop untuk mendapatkan query, lalu generate jawaban dan melakukan embd
hypothesis_answers = []
for i in query_metadata:
    # hypotesis asnwer
    query = i.get('query')
    hypothesis_answ = hypotesis_answer(query)

    i['hypothesis_answer'] = hypothesis_answ

    # hypotesis embd
    hypothesis_answer = i.get('hypothesis_answer')
    hypothesis_emb = hypotesis_embedding(hypothesis_answer)
    hypothesis_answers.append(hypothesis_emb)

# save hypothesis answer text
with open('C:/Users/diazr/VSCODE/RAG/project_5/hypothesis_answer.json', 'w') as f:
    json.dump(query_metadata, f, indent=4)

# save hypothesis answer embedding
hypothesis_answers = np.array(hypothesis_answers)
np.save('C:/Users/diazr/VSCODE/RAG/project_5/hypothesis_emb.npy', hypothesis_answers)







