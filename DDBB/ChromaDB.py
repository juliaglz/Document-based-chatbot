import os
import chromadb
import time
from doc_parser import CleanTextExtractor
from doc_chunker import TextChunker
class ChromaDBManagerS:
    def __init__(self, db_path):
        self.client = chromadb.PersistentClient(path=db_path)

    def create_collection(self, name):
        return self.client.get_or_create_collection(name=name)

    def add_documents(self, collection, documents, ids):
        collection.add(documents=documents, ids=ids)

    def get_ids(self, collection):
        info = collection.get()
        return info['ids']

    def delete_collection(self, name):
        self.client.delete_collection(name=name)

    def query_collection(self, collection, query, top_k=1):
        return collection.query(query_texts=query, n_results=top_k)


if __name__ == '__main__':
    db_path = "C:\\Users\\julia\\PycharmProjects\\TFG\\TFG_iteration3"
    data_directory = "./wiki2txt"

    db_manager = ChromaDBManagerS(db_path)
    collection_name = "MIGUEL_no"
    collection = db_manager.create_collection(collection_name)

    documents = []
    ids = []
    existing_ids = set(db_manager.get_ids(collection))
    print(existing_ids)
    for filename in os.listdir(data_directory):
        filepath = os.path.join(data_directory, filename)
        extractor = CleanTextExtractor(filepath)
        document_text = extractor.extract_text()
        chunker = TextChunker(document_text)
        chunks = chunker.chunk_by_sentences()

        for i, chunk in enumerate(chunks, start=1):
            chunk_id = f"{filename}_chunk_{i}"
            if chunk_id not in existing_ids:
                documents.append(chunk)
                ids.append(chunk_id)
    start = time.time()
    if documents:
        db_manager.add_documents(collection, documents, ids)
    while True:
        query = input("Ask (or type 'exit'): ")
        if query.lower() == 'exit':
            break
        start2 = time.time()
        results = collection.query(query_texts=[query], n_results=3)
        end2 = time.time()
        print("---------------------------", end2 - start2)
        print(results)
