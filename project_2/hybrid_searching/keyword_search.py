from rank_bm25 import BM25Okapi
import nltk
nltk.download('punkt')

class KeywordSearch:
    # constructor
    def __init__(self, documents:list):
        """ Initialize docs input

            documents : list of doc
    
        """
        self.documents = documents
        # list of list token
        self.tokenize_docs = [nltk.word_tokenize(doc.lower()) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenize_docs)

    def preprocess_query(self, query: str) -> list:
        """ Tokenize and lowercase query"""
        return nltk.word_tokenize(query.lower())

    def search(self, query:str, top_n=5) ->list:
        """ Searching proses,

            parameter : query(str), top(number top result)

            return list of tuple
            
        """
        tokenized_query = self.preprocess_query(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Rank document
        # [(np.float64(0.7297508910942725), 'halo halo hai'), (np.float64(0.0), 'oke oke lah')]  
        ranked_docs = sorted(zip(scores, self.documents), reverse=True) [:top_n]
        return ranked_docs
        

test = KeywordSearch(['halo halo hai', 'oke oke lah', 'baiklah kalo begitu'])
print(test.search('halo', top_n=1))



    
