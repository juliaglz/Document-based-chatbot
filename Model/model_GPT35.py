from openai import AzureOpenAI
from dotenv import load_dotenv
import os
from doc_parser import CleanTextExtractor
from doc_chunker import TextChunker
from TFIDF_embedding import TFIDFEmbedder
from Chromadb import ChromaDBManager

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def generate_response(user_input):
    message_text = [{"role": "user", "content": user_input}]
    completion = client.chat.completions.create(
    #COMPLETE WITH MODEL INFO
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


text = ("Es obligatorio que sigas estas reglas sin mencionarlas:\n"
            "Responde a la siguiente pregunta teniendo en cuenta esta información\n"
            )

def main():
    db_path = "path"
    db_manager = ChromaDBManager(db_path)
    collection_name = "documents"

    data_directory = "./data"
    documents = []
    ids = []

    for filename in os.listdir(data_directory):
        filepath = os.path.join(data_directory, filename)
        extractor = CleanTextExtractor(filepath)
        document_text = extractor.extract_text()
        chunker = TextChunker(document_text)
        chunks = chunker.chunk_by_sentences()
        documents.extend(chunks)
        for i in range(len(chunks)):
            chunk_id = f"{filename}_chunk_{i + 1}"
            if chunk_id not in ids:
                ids.append(chunk_id)

    print(ids)
    print(len(ids))
    print(len(documents))
    embedder = TFIDFEmbedder()
    tfidf_matrix = embedder.fit_transform(documents)
    embeddings = []
    document_ids = []
    for idx, vector in enumerate(tfidf_matrix):
        embeddings.append(vector.toarray().flatten().tolist())
        doc_index = idx // len(chunks)
        document_ids.append(f"doc_{doc_index}_{idx % len(chunks)}")
    print(len(embeddings))
    print(tfidf_matrix.shape)

    try:
        print("--------------------lo hace bien --------------------------------------------")
        db_manager.get_collection(collection_name)
    except:
        db_manager.create_collection(collection_name)
        print("---------------------aqui no --------------------------------------------")
        db_manager.add_documents(collection_name, embeddings, document_ids)

    while True:
        query = input("Ask (or type 'exit'): ")
        query_embeddings = embedder.transform([query]).todense()
        print(query_embeddings.shape)
        if query.lower() == 'exit':
            break
        results = db_manager.query_collection(collection_name,query_embeddings, 3)
        context = results['documents']

        response = generate_response(text+str(context)+query)
        print(response)

if __name__ == '__main__':
    main()
