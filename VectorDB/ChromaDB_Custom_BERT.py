import os
import chromadb
from doc_parser import CleanTextExtractor
from doc_chunker import TextChunker
from chromadb import EmbeddingFunction
from BERT_embedding import BertEmbedder
import time

class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        # self.tfidf_embedder = TFIDFEmbedder()
        self.bert_embedder = BertEmbedder()

    def __call__(self, texts):
        # tfidf_matrix = self.tfidf_embedder.fit_transform(texts)
        embeddings = self.bert_embedder.fit_transform(texts)
        result = []
        '''for idx, text in enumerate(texts):
            dense_vector = embeddings[idx].todense().tolist()[0]
            result.append(dense_vector)'''
        return embeddings


class ChromaDBManagerBert:
    def __init__(self, db_path):
        self.client = chromadb.PersistentClient(path=db_path)

    def create_collection(self, name):
        return self.client.get_or_create_collection(name=name, embedding_function=CustomEmbeddingFunction())

    def add_documents(self, collection, documents, ids):
        collection.add(documents=documents, ids=ids)

    '''def get_collection(self):
        return self.client.get(include=[ "ids" ])'''

    def get_ids(self, collection):
        info = collection.get()
        return info['ids']

    def delete_collection(self, name):
        self.client.delete_collection(name=name)

    def query_collection(self, collection, query, top_k=3):
        return collection.query(query_texts=query, n_results=top_k)



if __name__ == '__main__':
    db_path = "path"
    data_directory = "./wiki2txt"

    db_manager = ChromaDBManagerBert(db_path)
    collection_name = "documents_test_B"
    collection = db_manager.create_collection(collection_name)
    print()
    dbk = collection.get()
    #print(dbk['ids'])
    print(db_manager.get_ids(collection))
    documents = []
    ids = []
    for filename in os.listdir(data_directory):
        filepath = os.path.join(data_directory, filename)
        extractor = CleanTextExtractor(filepath)
        document_text = extractor.extract_text()
        chunker = TextChunker(document_text)
        chunks = chunker.chunk_by_sentences()
        if [f"{filename}_chunk_{i + 1}" for i in range(len(chunks))] in db_manager.get_ids(collection):
            documents.extend(chunks)
            ids.extend([f"{filename}_chunk_{i + 1}" for i in range(len(chunks))])
    start = time.time()
    if documents:
        db_manager.add_documents(collection, documents, ids)
    end = time.time()
    print("---------------------------", end - start)

    while True:
        query = input("Ask (or type 'exit'): ")
        if query.lower() == 'exit':
            break
        start2 = time.time()
        results = collection.query(query_texts=[query], n_results=3)
        end2 = time.time()
        print("---------------------------", end2 - start2)
        print(results)
