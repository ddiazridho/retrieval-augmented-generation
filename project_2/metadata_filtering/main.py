from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

client = OpenAI()

intructions = "Tentukan content yang masuk sesuai kategori, terdapat dua kategori ['ai framework', 'web development']. response kategorinya saja"

def query_user(text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": intructions},
            {"role": "user", "content": text}]
    )

    return response.choices[0].message.content

print(query_user("keras adalah framework untuk deep learning"))


# NOTE - NEXT STEP : Buat query untuk search chunk_with_metadata by category. Baru lanjut ke inti RAG