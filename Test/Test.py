import logging
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration
from openai import AzureOpenAI
from dotenv import load_dotenv
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os
from ChromaDB_Custom_TFIDF import ChromaDBManager
from ChromaDB_Custom_BERT import ChromaDBManagerBert
from ChromaDB import ChromaDBManagerS
from doc_parser import CleanTextExtractor
from doc_chunker import TextChunker
def generate_response_T5(context, query):
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

load_dotenv()

client = AzureOpenAI(
#COMPLETE WITH API INFO
)
def generate_response_GPT35(user_input):
    message_text = [{"role": "user", "content": user_input}]
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_OPENAI_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    ai_response = ""
    for choice in completion.choices:
        for message in choice.message:
            if isinstance(message, tuple) and message[0] == 'content':
                ai_response = message[1]
                break
        if ai_response:
            break

    return ai_response

def generate_response_GPT2(context, query):
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    prompt_text = f"CONTEXT: {context}\nQUESTION: {query}\n Answer:"
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

def main_TFIDF():
    db_path = "path"
    db_manager = ChromaDBManager(db_path)
    collection_name = "documents_TFIDF"
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
        start = time.time()
        response = generate_response_T5(context, query)
        end = time.time()
        print("TIME---- T5:", end - start)
        print("Generated response T5:", response)
        start = time.time()
        response = generate_response_GPT2(context, query)
        end = time.time()
        print("TIME---- GPT2:", end - start)
        print("Generated response GPT2:", response)
        context_2 = str(context)
        start = time.time()
        response = generate_response_GPT35("CONTEXT" + context_2 + "QUESTION" + query)
        end = time.time()
        print("TIME---- GPT3.5:", end - start)
        print("Generated response GPT3.5:", response)
def main_BERT():
    db_path = "path"
    db_manager = ChromaDBManagerBert(db_path)
    collection_name = "documents_BERT"
    collection = db_manager.create_collection(collection_name)

    data_directory = "./wiki2txt"
    documents = []
    ids = []
    existing_ids = set(db_manager.get_ids(collection))
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
    if documents:
        db_manager.add_documents(collection, documents, ids)

    while True:
        query = input("Ask (or type 'exit'): ")
        if query.lower() == 'exit':
            break
        results = db_manager.query_collection(collection, query, 3)
        print("Reference file:", results['ids'])
        context = results['documents']
        start = time.time()
        response = generate_response_T5(context, query)
        end = time.time()
        print("TIME---- T5:", end - start)
        print("Generated response T5:", response)
        start = time.time()
        response = generate_response_GPT2(context, query)
        end = time.time()
        print("TIME---- GPT2:", end - start)
        print("Generated response GPT2:", response)
        context_2 = str(context)
        start = time.time()
        response = generate_response_GPT35("CONTEXT" + context_2 + "QUESTION" + query)
        end = time.time()
        print("TIME---- GPT3.5:", end - start)
        print("Generated response GPT3.5:", response)
def main():
    db_path = "path"
    db_manager = ChromaDBManagerS(db_path)
    collection_name = "documents_chroma"
    collection = db_manager.create_collection(collection_name)

    data_directory = "./wiki2txt"
    documents = []
    ids = []
    existing_ids = set(db_manager.get_ids(collection))
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
    if documents:
        db_manager.add_documents(collection, documents, ids)

    while True:
        query = input("Ask (or type 'exit'): ")
        if query.lower() == 'exit':
            break
        results = db_manager.query_collection(collection, query, 1)
        print("Reference file:", results['ids'])
        context = results['documents']
        print("Context:", str(context))
        start = time.time()
        response = generate_response_T5(context, query)
        end = time.time()
        print("TIME---- T5:", end - start)
        print("Generated response T5:", response)
        start = time.time()
        response = generate_response_GPT2(context, query)
        end = time.time()
        print("TIME---- GPT2:", end - start)
        print("Generated response GPT2:", response)
        context_2 = str(context)
        start = time.time()
        response = generate_response_GPT35("CONTEXT" + context_2 + "QUESTION" + query)
        end = time.time()
        print("TIME---- GPT3.5:", end-start)
        print("Generated response GPT3.5:", response)
def measure_time(func):
    start_time = time.time()
    func()
    end_time = time.time()
    print(f"Time taken for {func.__name__}: {end_time - start_time} seconds")

if __name__ == '__main__':
    print("testing chroma db embeddings")
    measure_time(main)
    print("testing chroma TFIDF")
    measure_time(main_TFIDF)
    print("testing chroma BERT")
    measure_time(main_BERT)
