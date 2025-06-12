from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from doc_parser import CleanTextExtractor
from doc_chunker import TextChunker
import os
class TFIDFEmbedder:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.vectorizer = None
        if model_path and os.path.exists(model_path):
            self.load_vectorizer()

    def fit_vectorizer(self, documents):
        if not self.vectorizer:
            self.vectorizer = TfidfVectorizer()
            self.vectorizer.fit(documents)
            if self.model_path:
                self.save_vectorizer()

    def transform(self, documents):
        if not self.vectorizer:
            raise ValueError("Not fitted yet")
        return self.vectorizer.transform(documents)

    def fit_transform(self, documents):
        self.fit_vectorizer(documents)
        tfidf_matrix = self.transform(documents)
        return tfidf_matrix

    def save_vectorizer(self):
        if self.model_path:
            joblib.dump(self.vectorizer, self.model_path)

    def load_vectorizer(self):
        self.vectorizer = joblib.load(self.model_path)


def print_chunk_embeddings(chunks, tfidf_matrix, vectorizer):
    feature_names = vectorizer.get_feature_names_out()
    for idx, chunk in enumerate(chunks):
        print("\nChunk:", chunk)
        dense_vector = tfidf_matrix[idx].todense().tolist()[0]
        print("Embedding:", dense_vector)


'''
if __name__ == '__main__':
    data_directory = "./data2"
    documents = []
    all_chunks = []
    for filename in os.listdir(data_directory):
        filepath = os.path.join(data_directory, filename)
        extractor = CleanTextExtractor(filepath)
        document_text = extractor.extract_text()
        chunker = TextChunker(document_text)
        chunks = chunker.chunk_by_sentences()
        documents.extend(chunks)
        all_chunks.extend(chunks)

    embedder = TFIDFEmbedder()
    tfidf_matrix = embedder.fit_transform(documents)
    print(tfidf_matrix)
    #print_chunk_embeddings(all_chunks, tfidf_matrix, embedder.vectorizer)
'''
