from project_2.hybrid_searching.semantic_search import SemanticSearch
from project_2.hybrid_searching.keyword_search import KeywordSearch
import numpy as np

documents = [
        """Machine learning is a field of artificial intelligence that uses statistical techniques 
        to give computer systems the ability to learn from data, without being explicitly programmed. 
        This includes supervised learning, unsupervised learning, and reinforcement learning.""",
    
        """Natural language processing (NLP) is a branch of artificial intelligence focused on the 
        interaction between computers and humans through natural language. NLP includes tasks like 
        language translation, sentiment analysis, and speech recognition.""",
        
        """BM25 is an algorithm used in information retrieval that ranks documents based on their 
        relevance to a given search query. It is often applied in keyword-based search systems 
        and is particularly effective in ranking large collections of text data.""",
        
        """Hybrid search combines the strengths of keyword-based search with semantic search. 
        Keyword-based search like BM25 focuses on matching exact terms, while semantic search 
        uses dense vector representations to capture the contextual meaning of terms in a query.""",
        
        """Deep learning has revolutionized computer vision, enabling applications like image 
        classification, object detection, and facial recognition. Convolutional neural networks 
        (CNNs) are a common architecture used in these applications, designed specifically to 
        process grid-like data such as images."""
    ]
document_embeddings = np.array([
        [0.634, 0.234, 0.867, 0.042, 0.249],
        [0.123, 0.456, 0.789, 0.321, 0.654],
        [0.987, 0.654, 0.321, 0.123, 0.456],
        [0.456, 0.789, 0.123, 0.456, 0.987],
        [0.321, 0.654, 0.234, 0.123, 0.789]
])
keyword_object = KeywordSearch(documents)
semantic_object = SemanticSearch(document_embeddings)

class HybridSearch:
    def __init__(self, keyword_search, semantic_search, alpha=0.5):
        """
        keyword_search : object
        semantic_search : object
        """
        self.keyword_search = keyword_search
        self.semantic_search = semantic_search
        self.alpha = alpha

    def search(self, query, query_embedding, top_n=3):
        """
        query : teks asli
        query_embdding : teks embdding
        """
        # NOTE - keyword proses
        # menjalankan method search dari object key, [(score, doc)]
        keyword_results = self.keyword_search.search(query, top_n=top_n)
        # score, doc
        sparse_scores, documents = zip(*keyword_results)


        # NOTE - Semantic proses
        query_embedding = query_embedding.reshape(1,-1)
        # top n mengikuti output dari keyword output, bisa aja ga sesuai top n
        dense_scores = [self.semantic_search.search(query_embedding, top_n=len(documents))]
        # mengambil scorenya saja
        dense_scores = [score for score, _ in dense_scores[0]]

        # NOTE - Combined semantic score dan keyword score
        combined_scores = [
            (1 - self.alpha) * sparse_score + self.alpha * dense_score
            for sparse_score, dense_score in zip(sparse_scores, dense_scores)
        ]

        # NOTE - Rerank
        ranked_indices = np.argsort(combined_scores)[::-1]
        ranked_results = [(combined_scores[idx], documents[idx]) for idx in ranked_indices[:top_n]]


        return ranked_results

test = HybridSearch(keyword_object, semantic_object)
print(test.search('machine learning and artificial intelligence', np.array([0.765, 0.234, 0.678, 0.543, 0.123])))


