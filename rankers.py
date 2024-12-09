import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Tuple
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
 
import networkx as nx
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


class BaseSentenceRanker:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.scaler = MinMaxScaler()
        self.lemmatizer = WordNetLemmatizer()
        
    def preprocess_text(self, text: str) -> str:
        text = re.sub(r'[^\w\s]', '', text.lower())
        text = re.sub(r'\d+', '', text)
        words = word_tokenize(text)
        words = [
            self.lemmatizer.lemmatize(word)
            for word in words 
            if word not in self.stop_words
        ]
        return ' '.join(words)

class TFIDFRanker(BaseSentenceRanker):
    def rank_sentences(self, sentences: List[str]) -> List[Tuple[str, float]]:
        processed_sentences = [self.preprocess_text(sent) for sent in sentences]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(processed_sentences)
        scores = np.array([tfidf_matrix[i].sum() for i in range(len(sentences))])
        normalized_scores = self.scaler.fit_transform(scores.reshape(-1, 1)).flatten()
        ranked = list(zip(sentences, normalized_scores))
        return sorted(ranked, key=lambda x: x[1], reverse=True)
    
    def get_visualization_data(self, sentences: List[str]) -> Dict:
        processed_sentences = [self.preprocess_text(sent) for sent in sentences]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(processed_sentences).toarray()
        
        # Choose dimensionality reduction method based on number of sentences
        if len(sentences) >= 4:
            reducer = TSNE(n_components=2, perplexity=min(30, len(sentences)-1), random_state=42)
        else:
            reducer = PCA(n_components=2)
            
        coords_2d = reducer.fit_transform(tfidf_matrix)
        
        # Calculate scores for coloring
        scores = [tfidf_matrix[i].sum() for i in range(len(sentences))]
        normalized_scores = self.scaler.fit_transform(np.array(scores).reshape(-1, 1)).flatten()
        
        # Generate color gradient
        colors = [f'rgb({int(255*s)}, {int(100 + 155*(1-s))}, {int(255*(1-s))})' 
                 for s in normalized_scores]
        
        return {
            'x': coords_2d[:, 0].tolist(),
            'y': coords_2d[:, 1].tolist(),
            'labels': [s[:30] + '...' if len(s) > 30 else s for s in sentences],
            'colors': colors,
            'scores': normalized_scores.tolist()
        }

class Word2VecRanker(BaseSentenceRanker):
    def __init__(self):
        super().__init__()
        self.vector_size = 100
        self.window_size = 5
        self.min_count = 1
        
    def rank_sentences(self, sentences: List[str]) -> List[Tuple[str, float]]:
        # Preprocess and tokenize sentences
        tokenized_sentences = [word_tokenize(self.preprocess_text(sent)) 
                             for sent in sentences]
        print (tokenized_sentences[0])
        
        # Train Word2Vec model on the current document
        model = Word2Vec(sentences=tokenized_sentences, 
                        vector_size=self.vector_size, 
                        window=self.window_size, 
                        min_count=self.min_count)
        
        # Calculate sentence vectors
        sentence_vectors = []
        for sent in tokenized_sentences:
            word_vectors = [model.wv[word] for word in sent if word in model.wv]
            if word_vectors:
                sent_vector = np.mean(word_vectors, axis=0)
            else:
                sent_vector = np.zeros(self.vector_size)
            sentence_vectors.append(sent_vector)
        
        # Calculate document centroid
        doc_vector = np.mean(sentence_vectors, axis=0)
        
        # Calculate cosine similarities with document centroid
        similarities = []
        for sv in sentence_vectors:
            if np.any(sv):
                sim = np.dot(sv, doc_vector) / (np.linalg.norm(sv) * np.linalg.norm(doc_vector))
            else:
                sim = 0
            similarities.append(sim)
        
        # Normalize scores
        normalized_scores = self.scaler.fit_transform(np.array(similarities).reshape(-1, 1)).flatten()
        
        # Create ranked list
        ranked = list(zip(sentences, normalized_scores))
        return sorted(ranked, key=lambda x: x[1], reverse=True)
    
    def get_visualization_data(self, sentences: List[str]) -> Dict:
        # Preprocess and tokenize
        tokenized_sentences = [word_tokenize(self.preprocess_text(sent)) 
                             for sent in sentences]
        
        # Train Word2Vec model
        model = Word2Vec(sentences=tokenized_sentences, 
                        vector_size=self.vector_size, 
                        window=self.window_size, 
                        min_count=self.min_count)
        
        # Get sentence vectors
        sentence_vectors = []
        for sent in tokenized_sentences:
            word_vectors = [model.wv[word] for word in sent if word in model.wv]
            if word_vectors:
                sent_vector = np.mean(word_vectors, axis=0)
            else:
                sent_vector = np.zeros(self.vector_size)
            sentence_vectors.append(sent_vector)
        
        # Choose dimensionality reduction method
        if len(sentences) >= 4:
            reducer = TSNE(n_components=2, perplexity=min(30, len(sentences)-1), random_state=42)
        else:
            reducer = PCA(n_components=2)
            
        coords_2d = reducer.fit_transform(np.array(sentence_vectors))
        
        # Calculate similarities for coloring
        doc_vector = np.mean(sentence_vectors, axis=0)
        similarities = []
        for sv in sentence_vectors:
            if np.any(sv):
                sim = np.dot(sv, doc_vector) / (np.linalg.norm(sv) * np.linalg.norm(doc_vector))
            else:
                sim = 0
            similarities.append(sim)
        
        normalized_scores = self.scaler.fit_transform(np.array(similarities).reshape(-1, 1)).flatten()
        colors = [f'rgb({int(255*s)}, {int(100 + 155*(1-s))}, {int(255*(1-s))})' 
                 for s in normalized_scores]
        
        return {
            'x': coords_2d[:, 0].tolist(),
            'y': coords_2d[:, 1].tolist(),
            'labels': [s[:30] + '...' if len(s) > 30 else s for s in sentences],
            'colors': colors,
            'scores': normalized_scores.tolist()
        }
   

class TextRankRanker(BaseSentenceRanker):
    """TextRank implementation for sentence ranking"""
    
    def __init__(self, similarity_threshold=0.3, damping_factor=0.85):
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.damping_factor = damping_factor
        self.vectorizer = TfidfVectorizer()
        
    def calculate_similarity_matrix(self, sentences):
        # Create TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform([
            self.preprocess_text(sent) for sent in sentences
        ])
        
        # Calculate similarity matrix using cosine similarity
        similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
        
        # Apply threshold
        similarity_matrix[similarity_matrix < self.similarity_threshold] = 0
        
        return similarity_matrix
    
    def rank_sentences(self, sentences: List[str]) -> List[Tuple[str, float]]:
        if len(sentences) < 2:
            return [(sentences[0], 1.0)] if sentences else []
        
        # Build similarity matrix
        similarity_matrix = self.calculate_similarity_matrix(sentences)
        
        # Create graph
        graph = nx.from_numpy_array(similarity_matrix)
        
        # Apply PageRank
        scores = nx.pagerank(
            graph,
            alpha=self.damping_factor,
            tol=1e-6
        )
        
        # Normalize scores
        score_values = np.array(list(scores.values()))
        normalized_scores = self.scaler.fit_transform(score_values.reshape(-1, 1)).flatten()
        
        # Create ranked list
        ranked = list(zip(sentences, normalized_scores))
        return sorted(ranked, key=lambda x: x[1], reverse=True)
    
    def get_visualization_data(self, sentences: List[str]) -> Dict:
        if len(sentences) < 2:
            return {
                'x': [0],
                'y': [0],
                'labels': sentences,
                'colors': ['rgb(0,0,255)'],
                'scores': [1.0]
            }
            
        similarity_matrix = self.calculate_similarity_matrix(sentences)
        
        # Create graph layout
        G = nx.from_numpy_array(similarity_matrix)
        pos = nx.spring_layout(G, k=1/np.sqrt(len(sentences)))
        
        # Get scores
        scores = nx.pagerank(G, alpha=self.damping_factor)
        normalized_scores = self.scaler.fit_transform(
            np.array(list(scores.values())).reshape(-1, 1)
        ).flatten()
        
        # Generate visualization data
        coords = np.array(list(pos.values()))
        colors = [
            f'rgb({int(255*s)}, {int(100 + 155*(1-s))}, {int(255*(1-s))})' 
            for s in normalized_scores
        ]
        
        return {
            'x': coords[:, 0].tolist(),
            'y': coords[:, 1].tolist(),
            'labels': [s[:30] + '...' if len(s) > 30 else s for s in sentences],
            'colors': colors,
            'scores': normalized_scores.tolist()
        }

class TransformerRanker(BaseSentenceRanker):
    """BERT-based sentence ranking using contextual embeddings"""
    
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def get_embeddings(self, sentences: List[str]) -> np.ndarray:
        # Tokenize sentences
        encoded_input = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Move to GPU if available
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            
        # Mean Pooling
        attention_mask = encoded_input['attention_mask']
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings.cpu().numpy()
    
    def rank_sentences(self, sentences: List[str]) -> List[Tuple[str, float]]:
        if not sentences:
            return []
            
        # Get sentence embeddings
        embeddings = self.get_embeddings(sentences)
        
        # Calculate document embedding (centroid)
        doc_embedding = np.mean(embeddings, axis=0)
        
        # Calculate cosine similarities
        similarities = np.dot(embeddings, doc_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(doc_embedding)
        )
        
        # Normalize scores
        normalized_scores = self.scaler.fit_transform(similarities.reshape(-1, 1)).flatten()
        
        # Create ranked list
        ranked = list(zip(sentences, normalized_scores))
        return sorted(ranked, key=lambda x: x[1], reverse=True)
    
    def get_visualization_data(self, sentences: List[str]) -> Dict:
        if not sentences:
            return {
                'x': [],
                'y': [],
                'labels': [],
                'colors': [],
                'scores': []
            }
            
        # Get embeddings
        embeddings = self.get_embeddings(sentences)
        
        # Reduce dimensionality
        reducer = TSNE(n_components=2, perplexity=min(30, len(sentences)-1)) if len(sentences) >= 4 else PCA(n_components=2)
        coords_2d = reducer.fit_transform(embeddings)
        
        # Calculate importance scores
        doc_embedding = np.mean(embeddings, axis=0)
        similarities = np.dot(embeddings, doc_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(doc_embedding)
        )
        
        # Normalize scores
        normalized_scores = self.scaler.fit_transform(similarities.reshape(-1, 1)).flatten()
        
        # Generate colors
        colors = [
            f'rgb({int(255*s)}, {int(100 + 155*(1-s))}, {int(255*(1-s))})' 
            for s in normalized_scores
        ]
        
        return {
            'x': coords_2d[:, 0].tolist(),
            'y': coords_2d[:, 1].tolist(),
            'labels': [s[:30] + '...' if len(s) > 30 else s for s in sentences],
            'colors': colors,
            'scores': normalized_scores.tolist()
        }