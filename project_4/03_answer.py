from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
client = OpenAI()

import json

def generate_answer(context_str, query):
    prompt = f"""Kamu adalah asisten yang menjawab pertanyaan HANYA berdasarkan konteks yang diberikan di bawah ini.

                ATURAN PENTING:
                1. Jawab HANYA menggunakan informasi yang ada di konteks. Jangan gunakan pengetahuan umum di luar konteks ini.
                2. Jika jawaban tidak ditemukan di konteks, katakan dengan jelas: "Saya tidak menemukan informasi ini di dokumen yang tersedia." Jangan mengarang atau menebak.
                3. Setiap klaim dalam jawabanmu harus bisa ditelusuri ke salah satu bagian konteks di bawah. Sebutkan nomor sumber [1], [2], dst yang mendukung tiap klaim.

                KONTEKS:
                {context_str}

                PERTANYAAN:
                {query}

                JAWABAN:"""
    
    response = client.responses.create(
        input= prompt,
        model= 'gpt-4o-mini'
    )

    return response.output_text

with open('C:/Users/diazr/VSCODE/RAG/project_4/retrieval_results.json', 'r') as f:
    retrieval = json.load(f)

# looping untuk mendapatkan query dan konteks, lalu kirim llm
for i in retrieval:
    query = i.get('query')
    context = i.get('results')
    output_text = generate_answer(context, query)

    i['answer_gpt4omini'] = output_text

with open('C:/Users/diazr/VSCODE/RAG/project_4/retrieval_results.json', 'w') as f:
    json.dump(retrieval, f, indent=4 )

     