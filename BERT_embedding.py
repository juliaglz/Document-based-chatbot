import torch
from transformers import BertTokenizer, BertModel
import os
from doc_parser import CleanTextExtractor
from doc_chunker import TextChunker
class BertEmbedder:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

    def fit_transform(self, documents):
        return [self.get_embedding(doc) for doc in documents]


def print_chunk_embeddings(chunks, embeddings):
    for idx, chunk in enumerate(chunks):
        print("\nChunk:", chunk)
        print("Embedding:", embeddings[idx])


# Example Usage
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

    embedder = BertEmbedder()
    tfidf_matrix = embedder.fit_transform(documents)
    #print(tfidf_matrix)
    print_chunk_embeddings(all_chunks, tfidf_matrix)
