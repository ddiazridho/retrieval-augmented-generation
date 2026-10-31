# RAG FUNDAMENTAL

## project_0 FUNDAMENTAL
1. Preparing document
2. Ground truth for evaluating
3. Fixed size chunking document
4. Embedd the chunk
5. Retrieve (similarity between vector query and vector chunk)
6. Top K similarity per query
7. Manual Validate

## project_1 CHUNK STRATEGY
- Variants:
	- Chunk overlap (beberapa kalimat chunk t-1 masuk ke chunk t)
	- Chunk recursive (split per paragraf, jika lebih dari maxsize maka di split per sentence, jika kurang dianggap satu chunk)
	- Chunk sentencebased (split by k-sentence)
- Outputs:
	- eval: `project_1/eval_results_*.json`
	- metrics: `project_1/metrics_summary.json`

## project_2 RETRIEVAL QUALITY (INDEX DAN SEARCHING)
- Brute force vs HNSW (accurate vs time computation)
- Metadata filtering (search category of query user in chunk)
- Hybrid search (menyatukan vector search dan keyword search)

## project_3 RERANGKING
Melakukan cross encoder terhadap vektor terbaik dengan query setelah retrieval. Proses ini melakukan pencarian relevansi menggunakan model.

## project_4 EVALUATION
- context relevance = untuk mengevaluasi hasil chunk rag, apakah relevan dengan querynya (kualitas rag)
- faithfulness = mengevaluasi apakah jawaban yang dibuat llm sesuai dengan context yang diberikan (halusinasi model)
- answer_relevance = mengaluasi apakah jawaban sesuai dengan query dari user (kesesuaian jawaban dengan pertanyaan)

## project_5 ADVANCE RAG
- Query rewrite/expansion = 
menggunakan hyDE. meminta LLM untuk generate jawaban hipotesis sesuai query, lalu melakukan similarity(hypothesis_answer, vector_db). jawaban hipotesis bisa salah secara fakta, namun akan memperkuat similarity searchnya.
menggunakan rewrite querynya