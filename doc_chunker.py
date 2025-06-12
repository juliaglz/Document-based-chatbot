import re
from doc_parser import CleanTextExtractor
import os
import time
class TextChunker:
    def __init__(self, text, max_length=1000):
        self.text = text
        self.max_length = max_length

    def chunk_by_sentences(self):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', self.text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.max_length:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
'''
if __name__ == '__main__':
    data_directory = "./wiki2txt"
    cummulative=0
    for filename in os.listdir(data_directory):
        filepath = os.path.join(data_directory, filename)
        extractor = CleanTextExtractor(filepath)
        document_text = extractor.extract_text()

        # Measure time for chunking
        start_time = time.time()
        chunker = TextChunker(document_text)
        chunks = chunker.chunk_by_sentences()
        end_time = time.time()
        cummulative += (end_time - start_time)
        print(f"Time taken to chunk the text in {filename}: {end_time - start_time} seconds")
        print(f"Number of chunks: {len(chunks)}")
    print("total time",cummulative)
'''