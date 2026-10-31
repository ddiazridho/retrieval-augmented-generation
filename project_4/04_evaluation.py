from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
client = OpenAI()

import json

PATH_RETRIEVAL_RESULTS = 'C:/Users/diazr/VSCODE/RAG/project_4/retrieval_results.json'
with open(PATH_RETRIEVAL_RESULTS, 'r') as f:
    retrieval_metadata = json.load(f)

def llm_judge(prompt):
    response = client.responses.create(
        model='gpt-4o-mini',
        input= prompt,
        temperature=0, 
        text= {
            "format": {
                "type": "json_object"
            }
        }
    )
    return response.output_text

# Context relevance evaluation (query + 1 chunk)
def context_relevance(query, chunk_text):
    prompt = f"""Nilai apakah potongan teks berikut relevan untuk menjawab pertanyaan.

                Pertanyaan: {query}

                Potongan teks: {chunk_text}

                Beri skor di range 0-1 (0 = tidak relevan, 1 = relevan dan langsung menjawab pertanyaan).
                Jawab dalam format JSON: {{"score": <angka>, "reason": "<alasan singkat>"}}"""
    context_relevance_output = json.loads(llm_judge(prompt=prompt))
    return context_relevance_output

def faithfulness(context, generated_answer):
    prompt = f"""Kamu adalah fact-checker. Cek apakah jawaban di bawah ini SEPENUHNYA didukung oleh konteks yang diberikan.

                KONTEKS:
                {context}

                JAWABAN YANG DICEK:
                {generated_answer}

                Instruksi:
                - Identifikasi klaim-klaim dalam jawaban.
                - Untuk tiap klaim, cek apakah didukung konteks (supported), bertentangan (contradicted), atau tidak disebutkan sama sekali di konteks (unsupported/hallucinated).
                - Beri skor di range 0-1 (1 = semua klaim didukung penuh, 0 = jawaban mengarang tanpa dasar konteks sama sekali).

                Jawab dalam format JSON: {{"score": <angka>, "unsupported_claims": ["<klaim yang tidak didukung, jika ada>"], "reason": "<penjelasan singkat>"}}"""
    faithfulness_output = json.loads(llm_judge(prompt))
    return faithfulness_output

def answer_relevance(query, generated_answer):
    prompt = f"""Nilai apakah jawaban berikut benar-benar menjawab pertanyaan yang diajukan (bukan menjawab hal lain yang cuma mirip topiknya).

                Pertanyaan: {query}

                Jawaban: {generated_answer}

                Beri skor di range 0-1 (0 = melenceng total dari pertanyaan, 1 = menjawab tepat sasaran).
                Jawab dalam format JSON: {{"score": <angka>, "reason": "<alasan singkat>"}}"""
    answer_relevance_output =  json.loads(llm_judge(prompt))
    return answer_relevance_output


evaluation_metadata = []
for i in retrieval_metadata:
    query = i.get('query')
    results = i.get('results')
    generated_answer = i.get('answer_gpt4omini')

    scores_context_relv = []
    for result in results:
        context = result.get('text')
        context_relv = context_relevance(query, context)
        score_context_relv = context_relv.get('score')
        scores_context_relv.append(
            {
                "doc_id" : result.get('doc_id'),
                "chunk_id" : result.get('chunk_id'),
                "score" : score_context_relv
            }
        )

    contexts = [result.get('text') for result in results]
    faithfulness_test = faithfulness(context=contexts, generated_answer=generated_answer)
    score_faithfulness = faithfulness_test.get('score')

    answer_relv = answer_relevance(query=query, generated_answer=generated_answer)
    score_answer_relv = answer_relv.get('score')

    metadata = {
        "query_id" : i.get('query_id'),
        "query" : query,
        "type" : i.get('type'),
        "generated_answer" : generated_answer,
        "results" : results,
        "context_relevance_per_chunk" : scores_context_relv,
        "faithfulness" : score_faithfulness,
        "answer_relevance" : score_answer_relv
    }

    evaluation_metadata.append(metadata)

with open('C:/Users/diazr/VSCODE/RAG/project_4/evaluation.json', 'w') as f:
    json.dump(evaluation_metadata, f, indent=4)