import logging
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from doc_parser import CleanTextExtractor
from doc_chunker import TextChunker
from ChromaDB_Custom_TFIDF import ChromaDBManager
def generate_response(context, query):
    logging.getLogger('transformers').setLevel(logging.ERROR)

    model_name = "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    input_text = f"context: {context} query: {query}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    print(f"Number of input tokens: {len(input_ids[0])}")
    outputs = model.generate(
        input_ids,
        max_length=1000,
        min_length=40,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        temperature=0.8,
        do_sample=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    db_path = "path"
    db_manager = ChromaDBManager(db_path)
    collection_name = "T5"
    collection = db_manager.create_collection(collection_name)

    data_directory = "./data2"
    documents = []
    ids = []

    for filename in os.listdir(data_directory):
        filepath = os.path.join(data_directory, filename)
        extractor = CleanTextExtractor(filepath)
        document_text = extractor.extract_text()
        chunker = TextChunker(document_text)
        chunks = chunker.chunk_by_sentences()
        documents.extend(chunks)
        ids.extend([f"{filename}_chunk_{i + 1}" for i in range(len(chunks))])

    db_manager.add_documents(collection, documents, ids)

    while True:
        query = input("Ask (or type 'exit'): ")
        if query.lower() == 'exit':
            break
        results = db_manager.query_collection(collection, query, 3)
        print("Reference file:", results['ids'])
        context = results['documents']
        print("Context:", str(context))
        response = generate_response(context, query)
        print("Generated response:",response)

if __name__ == '__main__':
    main()



'''
existing_ids = db_manager.get_ids(collection)

    data_directory = "./data2"
    documents = []
    ids = []

    for filename in os.listdir(data_directory):
        filepath = os.path.join(data_directory, filename)
        extractor = CleanTextExtractor(filepath)
        document_text = extractor.extract_text()
        chunker = TextChunker(document_text)
        chunks = chunker.chunk_by_sentences()
        new_ids = [f"{filename}_chunk_{i + 1}" for i in range(len(chunks))]

        for idx, doc_id in enumerate(new_ids):
            if doc_id not in existing_ids:
                documents.append(chunks[idx])
                ids.append(doc_id)

    if documents:
        db_manager.add_documents(collection, documents, ids)
'''
