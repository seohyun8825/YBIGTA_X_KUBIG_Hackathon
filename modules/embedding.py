import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

class EmbeddingHandler:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained("UFNLP/gatortron-medium")
        self.model = AutoModel.from_pretrained("UFNLP/gatortron-medium").to(self.device)
        self.index = faiss.IndexFlatL2(768)  # Assuming 768-dimension embeddings

    def get_embedding(self, text):
        # Tokenize input text, move tensors to GPU
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Move embeddings to CPU and convert to NumPy
        embedding = outputs.pooler_output.cpu().numpy().squeeze()
        return embedding

    def insert_data_into_faiss(self, data):
        # Efficiently gather embeddings for all documents
        vectors = [self.get_embedding(d["documents"]) for d in tqdm(data)]

        # Convert list of vectors to a single NumPy array
        vectors = np.stack(vectors)

        # Normalize vectors for optimal FAISS performance
        faiss.normalize_L2(vectors)

        # Add vectors to the FAISS index
        self.index.add(vectors)
        faiss.write_index(self.index, "faiss_index.bin")
