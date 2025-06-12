from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os
from doc_parser import CleanTextExtractor
from doc_chunker import TextChunker
from ChromaDB_Custom_TFIDF import ChromaDBManager
def generate_response(context, query):
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    prompt_text = f"Using the details strictly from the context provided below, answer the question.\n\nContext: {context}\nQuestion: {query}\n Answer:"
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")


    outputs = model.generate(
        input_ids,
        max_length=1000,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

def main():
    db_path = "path"
    db_manager = ChromaDBManager(db_path)
    collection_name = "GPT2"
    collection = db_manager.create_collection(collection_name)

    data_directory = "./wiki2txt"
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
