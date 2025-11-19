import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

class EmbeddingGenerator:
    """Generate semantic embeddings for merchant names and descriptions"""
    
    def __init__(self):
        self.merchant_vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 3),
            analyzer='char_wb'
        )
        self.description_vectorizer = TfidfVectorizer(
            max_features=50,
            ngram_range=(1, 2),
            analyzer='word'
        )
        self.is_fitted = False
        
    def fit(self, merchants, descriptions):
        """Fit vectorizers on training data"""
        self.merchant_vectorizer.fit(merchants)
        self.description_vectorizer.fit(descriptions)
        self.is_fitted = True
        
    def generate_embeddings(self, merchant, description):
        """Generate combined embeddings for a transaction"""
        if not self.is_fitted:
            # Use pre-fitted or return zero embeddings
            merchant_emb = np.zeros(100)
            desc_emb = np.zeros(50)
        else:
            merchant_emb = self.merchant_vectorizer.transform([merchant]).toarray()[0]
            desc_emb = self.description_vectorizer.transform([description]).toarray()[0]
        
        # Combine embeddings
        combined = np.concatenate([merchant_emb, desc_emb])
        return combined
    
    def get_similarity(self, emb1, emb2):
        """Calculate cosine similarity between two embeddings"""
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def find_similar_transactions(self, query_embedding, transaction_embeddings, top_k=5):
        """Find most similar transactions"""
        similarities = []
        
        for i, emb in enumerate(transaction_embeddings):
            sim = self.get_similarity(query_embedding, emb)
            similarities.append((i, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def save(self, path='models/embeddings.pkl'):
        """Save vectorizers"""
        with open(path, 'wb') as f:
            pickle.dump({
                'merchant_vectorizer': self.merchant_vectorizer,
                'description_vectorizer': self.description_vectorizer,
                'is_fitted': self.is_fitted
            }, f)
    
    def load(self, path='models/embeddings.pkl'):
        """Load vectorizers"""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.merchant_vectorizer = data['merchant_vectorizer']
                self.description_vectorizer = data['description_vectorizer']
                self.is_fitted = data['is_fitted']
