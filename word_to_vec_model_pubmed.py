import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
import pandas as pd
import xml.etree.ElementTree as ET
import os
import re
from tqdm import tqdm
import json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from adjustText import adjust_text
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from umap import UMAP
import networkx as nx
import pandas as pd

def parseXML(xmlfile):
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    docs = []
    for article in root.findall('.//PubmedArticle'):
        pmid = article.find('.//PMID').text
        title = article.find('.//ArticleTitle')
        abstractAll = article.findall('.//Abstract/AbstractText')
        abstract = [ET.tostring(abstract, method='text', encoding='unicode') for abstract in abstractAll]
        abstract = " ".join(abstract)
        
        abstract = re.sub(r'<.*?>', ' ', abstract)

        pmid = pmid if pmid is not None else ""
        title_text = title.text if title is not None else ""
        abstract_text = abstract if abstract is not None else ""
        if title_text and abstract_text:
            docs.append({'pmid': pmid, 'title': title_text, 'abstract': abstract_text})
    return docs


def load_documents(folder):
    documents=[]
    files = os.listdir(folder)
    for file in tqdm(files):
        if file.endswith('.xml'):
            fullpath = os.path.join(folder, file)
            documents.extend(parseXML(fullpath))
    return documents


class PubmedProcessor:
    def __init__(self):
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.model = None
        
    def preprocess_text(self, text):
        """Preprocess text by removing special characters, converting to lowercase,
        removing stopwords, and lemmatizing."""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return tokens
    
    
    def save_documents(self, filename):
        """Save documents to a JSON file."""
        with open(filename, 'w') as f:
            json.dump([{k: v for k, v in doc.items() if k != 'processed_text'} 
                      for doc in self.documents], f)

    def load_documents(self, documents):
        """Load documents from a JSON file."""
        self.documents = documents
        print (f"Loaded {len(self.documents)} documents")
        for doc in tqdm( documents):
            doc['processed_text'] = self.preprocess_text(doc['title'] + " " + doc['abstract'])


    def add_document(self, pmid, title, abstract):
        """Add a document to the collection."""
        doc = {
            'pmid': pmid,
            'title': title,
            'abstract': abstract,
            'processed_text': self.preprocess_text(title + " " + abstract)
        }
        self.documents.append(doc)

    def train_word2vec(self, vector_size=30, window=2, min_count=1, sg=0):
        """Train the Word2Vec model on all documents using CBOW architecture."""
        # Prepare sentences for training
        sentences = [doc['processed_text'] for doc in self.documents]
        
        # Train the model with CBOW (sg=0 specifies CBOW architecture)
        self.model = Word2Vec(sentences=sentences,
                            vector_size=vector_size,
                            window=window,
                            min_count=min_count,
                            workers=4,
                            sg=sg)  # sg=0 for CBOW (default)
        
        # save the model
        # self.model.save('word2vec.model')

    def find_similar_words(self, word, topn=10):
        """Find similar words based on the CBOW model."""
        try:
            # Preprocess the input word
            processed_word = self.lemmatizer.lemmatize(word.lower())
            
            # Check if the word exists in the vocabulary
            if processed_word in self.model.wv:
                # Get similar words
                similar_words = self.model.wv.most_similar(processed_word, topn=topn)
                return similar_words
            else:
                return []
        except KeyError:
            return []

    def suggest_words(self, context_words, topn=5):
        """Suggest words based on context using CBOW model."""
        try:
            # Preprocess context words
            processed_context = [
                self.lemmatizer.lemmatize(word.lower())
                for word in context_words
                if self.lemmatizer.lemmatize(word.lower()) in self.model.wv
            ]
            
            if not processed_context:
                return []
            
            # Get the average context vector
            context_vectors = [self.model.wv[word] for word in processed_context]
            average_vector = np.mean(context_vectors, axis=0)
            
            # Find most similar words to the context
            similar_words = self.model.wv.similar_by_vector(average_vector, topn=topn)
            return similar_words
        except (KeyError, ValueError):
            return []

    def get_word_vector(self, word):
        """Get the vector representation of a word."""
        try:
            processed_word = self.lemmatizer.lemmatize(word.lower())
            if processed_word in self.model.wv:
                return self.model.wv[processed_word]
            return None
        except KeyError:
            return None

    def calculate_word_similarity(self, word1, word2):
        """Calculate cosine similarity between two words."""
        vec1 = self.get_word_vector(word1)
        vec2 = self.get_word_vector(word2)
        
        if vec1 is not None and vec2 is not None:
            similarity = np.dot(vec1, vec2) / (
                np.linalg.norm(vec1) * np.linalg.norm(vec2)
            )
            return similarity
        return None
        
    def get_document_vector(self, processed_text):
        """Calculate the average vector for a document."""
        vectors = []
        for word in processed_text:
            if word in self.model.wv:
                vectors.append(self.model.wv[word])
        
        if vectors:
            return np.mean(vectors, axis=0)
        return None

    def query_similar_documents(self, query_text, top_n=5):
        """Find documents similar to the query text."""
        # Preprocess query
        processed_query = self.preprocess_text(query_text)
        query_vector = self.get_document_vector(processed_query)
        
        if query_vector is None:
            return []
        
        # Calculate similarity with all documents
        similarities = []
        for doc in self.documents:
            doc_vector = self.get_document_vector(doc['processed_text'])
            if doc_vector is not None:
                similarity = np.dot(query_vector, doc_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
                )
                similarities.append((doc, similarity))
        
        # Sort by similarity and return top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]

class VectorVisualizer:
    def __init__(self, processor):
        self.processor = processor
        
    def get_vectors_and_words(self, n_words=100):
        """Get the top N most common words and their vectors"""
        # Get words sorted by frequency
        words = [word for word, vocab in 
                sorted(self.processor.model.wv.key_to_index.items(),
                      key=lambda x: self.processor.model.wv.get_vecattr(x[0], "count"),
                      reverse=True)[:n_words]]
        
        # Get vectors for these words
        vectors = np.array([self.processor.model.wv[word] for word in words])

        # counts of words
        counts = [self.processor.model.wv.get_vecattr(word, "count") for word in words]
        return vectors, words, counts
    
    
    def plot_vector_space(self, n_words=100, method='tsne', figsize=(20, 15),
                         annotate=True, arrows=False):
        """
        Plot word vectors in 2D space using either t-SNE or PCA with improved label placement
        
        Parameters:
        -----------
        n_words : int
            Number of most frequent words to plot
        method : str
            'tsne' or 'pca' for dimensionality reduction
        figsize : tuple
            Figure size
        annotate : bool
            Whether to show word labels
        arrows : bool
            Whether to show arrows from origin to points
        """
        # Get vectors and words
        vectors, words, counts = self.get_vectors_and_words(n_words)
        
        # Normalize counts for better visualization
        counts = np.array(counts)
        min_size = 100  # Minimum dot size
        max_size = 2000  # Maximum dot size
        
        # Log-scale the counts to handle large variations better
        counts_log = np.log1p(counts)  # log1p to handle zeros
        
        # Min-max scaling to normalize sizes between min_size and max_size
        if counts_log.max() - counts_log.min() > 0:  # Avoid division by zero
            sizes = min_size + (max_size - min_size) * (
                (counts_log - counts_log.min()) / (counts_log.max() - counts_log.min())
            )
        else:
            sizes = np.full_like(counts, min_size)
        
        # Reduce dimensionality
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, n_words-1))

        elif method.lower() == 'umap':
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(30, n_words-1))
        else:
            reducer = PCA(n_components=2)
            
        vectors_2d = reducer.fit_transform(vectors)
        
        # Create plot with white background
        plt.figure(figsize=figsize, facecolor='white')
        ax = plt.gca()
        ax.set_facecolor('white')
        
        # Plot points with scaled sizes
        scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], 
                            c='lightblue', alpha=0.5, s=sizes)
        
        if arrows:
            # Plot arrows from origin to points
            plt.quiver(np.zeros(vectors_2d.shape[0]), 
                      np.zeros(vectors_2d.shape[0]),
                      vectors_2d[:, 0], vectors_2d[:, 1],
                      angles='xy', scale_units='xy', scale=1, alpha=0.1)
        
        if annotate:
            # Create text annotations
            texts = []
            for i, (word, count) in enumerate(zip(words, counts)):
                # Add count information to labels
                label = f"{word}"
                
                # highlight the word 'caregiver'
                if word == 'caregiver':
                    plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1], 
                              c='red', s=sizes[i], label='caregiver')
                
                texts.append(plt.text(vectors_2d[i, 0], vectors_2d[i, 1], label,
                                    fontsize=12, alpha=0.7))
            
            # Adjust text positions to minimize overlap
            adjust_text(texts, 
                       arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5),
                       expand_points=(1.5, 1.5),
                       force_points=(0.1, 0.25))
        
        # Add legend for size reference
        legend_elements = [
            plt.scatter([], [], s=min_size, c='lightblue', alpha=0.7, 
                       label=f'Min freq: {int(np.min(counts))}'),
            plt.scatter([], [], s=max_size, c='lightblue', alpha=0.7, 
                       label=f'Max freq: {int(np.max(counts))}')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Add title and labels with improved styling
        plt.title(f'Word Vectors Visualization using {method.upper()}\n'
                 f'(dot size represents word frequency)',
                 pad=20, size=14, weight='bold')
        plt.xlabel('Dimension 1', size=12)
        plt.ylabel('Dimension 2', size=12)
        
        # Add grid with improved styling
        plt.grid(True, alpha=0.2, linestyle='--')
        
        # Tight layout to prevent label cutoff
        plt.tight_layout()
        
        return plt

    
    def plot_word_neighborhood(self, target_word, n_neighbors=10, figsize=(12, 8)):
        """
        Plot a specific word and its nearest neighbors with improved label placement
        
        Parameters:
        -----------
        target_word : str
            Word to analyze
        n_neighbors : int
            Number of nearest neighbors to show
        figsize : tuple
            Figure size
        """
        # Get similar words
        try:
            similar_words = self.processor.find_similar_words(target_word, n_neighbors)
            if not similar_words:
                print(f"Word '{target_word}' not found in vocabulary")
                return None
        except KeyError:
            print(f"Word '{target_word}' not found in vocabulary")
            return None
            
        # Get vectors for target word and neighbors
        words = [target_word] + [word for word, _ in similar_words]
        similarities = [1.0] + [sim for _, sim in similar_words]  # Include similarity scores
        vectors = np.array([self.processor.model.wv[word] for word in words])
        
        # Reduce to 2D using t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_neighbors-1))
        vectors_2d = tsne.fit_transform(vectors)
        
        # Create plot with white background
        plt.figure(figsize=figsize, facecolor='white')
        ax = plt.gca()
        ax.set_facecolor('white')
        
        # Plot target word
        plt.scatter(vectors_2d[0, 0], vectors_2d[0, 1], c='red', s=150, label='Target word')
        
        # Plot neighbors with color gradient based on similarity
        scatter = plt.scatter(vectors_2d[1:, 0], vectors_2d[1:, 1],
                            c=similarities[1:], cmap='Blues',
                            alpha=0.6, s=100, label='Similar words')
        
        # Add colorbar
        plt.colorbar(scatter, label='Similarity score')
        
        # Add labels with adjust_text
        texts = []
        # Add target word label
        texts.append(plt.text(vectors_2d[0, 0], vectors_2d[0, 1], target_word,
                            color='red', fontsize=12, weight='bold'))
        
        # Add neighbor labels with similarity scores
        for i, (word, sim) in enumerate(zip(words[1:], similarities[1:]), 1):
            label = f"{word}\n({sim:.2f})"
            texts.append(plt.text(vectors_2d[i, 0], vectors_2d[i, 1],
                                label, fontsize=10))
        
        # Adjust text positions to minimize overlap
        adjust_text(texts,
                   arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5),
                   expand_points=(1.5, 1.5),
                   force_points=(0.1, 0.25))
            
        # Add title and labels with improved styling
        plt.title(f'Word Neighborhood: {target_word}',
                 pad=20, size=14, weight='bold')
        plt.xlabel('t-SNE Dimension 1', size=12)
        plt.ylabel('t-SNE Dimension 2', size=12)
        
        # Add grid with improved styling
        plt.grid(True, alpha=0.2, linestyle='--')
        
        # Add legend
        plt.legend(loc='upper right')
        
        # Tight layout to prevent label cutoff
        plt.tight_layout()
        
        return plt
    
    # plot suggest words
    def plot_suggest_words(self, context_words, n_suggestions=5, figsize=(12, 8)):
        """
        Plot a specific word and its nearest neighbors with improved label placement
        
        Parameters:
        -----------
        context_words : list
            List of context words
        n_suggestions : int
            Number of suggested words to show
        figsize : tuple
            Figure size
        """
        # Get suggested words
        try:
            suggested_words = self.processor.suggest_words(context_words, n_suggestions)
            if not suggested_words:
                print(f"No suggestions found for context: {context_words}")
                return None
        except KeyError:
            print(f"No suggestions found for context: {context_words}")
            return None
            
        # Get vectors for suggested words
        words = [word for word, _ in suggested_words]
        similarities = [sim for _, sim in suggested_words]
        vectors = np.array([self.processor.model.wv[word] for word in words])

        # Reduce to 2D using PCA
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(vectors)

        # Create plot with white background
        plt.figure(figsize=figsize, facecolor='white')
        ax = plt.gca()
        ax.set_facecolor('white')

        # Plot context words
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c='lightblue', s=100, label='Context words')

        # Plot suggested words with color gradient based on similarity
        scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1],
                            c=similarities, cmap='Blues',
                            alpha=0.6, s=100, label='Suggested words')
        
        # Add colorbar
        plt.colorbar(scatter, label='Similarity score')

        # Add labels with adjust_text
        texts = []
        # Add context word labels
        for i, word in enumerate(context_words):
            texts.append(plt.text(vectors_2d[i, 0], vectors_2d[i, 1], word,
                                color='red', fontsize=12, weight='bold'))
            
        # Add suggested word labels with similarity scores

        for i, (word, sim) in enumerate(zip(words, similarities)):
            label = f"{word}\n({sim:.2f})"
            texts.append(plt.text(vectors_2d[i, 0], vectors_2d[i, 1],
                                label, fontsize=10))
            
        # Adjust text positions to minimize overlap
        adjust_text(texts,
                   arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5),
                   expand_points=(1.5, 1.5),
                   force_points=(0.1, 0.25))
        
        # Add title and labels with improved styling
        plt.title(f'Suggested Words for Context: {context_words}',
                    pad=20, size=14, weight='bold')
        plt.xlabel('PCA Dimension 1', size=12)
        plt.ylabel('PCA Dimension 2', size=12)

        # Add grid with improved styling
        plt.grid(True, alpha=0.2, linestyle='--')

        # Add legend
        plt.legend(loc='upper right')

        # Tight layout to prevent label cutoff
        plt.tight_layout()

        return plt
    


# Example usage
# processor = ICDProcessor()

# # Load documents
# # documents = load_documents('./pubmed4/')

# # Process documents
# for doc in documents:

#     processor.add_document(doc['code'], doc['title'], doc['abstract'])

# Train the model
# processor.train_word2vec()

# Example query
# results = processor.query_similar_documents("thailand taiwan", top_n=5)
# for doc, similarity in results:
#     print(f"code: {doc['code']}")
#     print(f"Title: {doc['title']}")
#     print(f"Abstract: {doc['abstract']}")
#     print(f"Similarity: {similarity:.4f}")
#     print()

# sugest words
# print(processor.suggest_words(['thailand', 'taiwan']))

# similar words
# print(processor.find_similar_words('age'))

# Example usage
# visualizer = VectorVisualizer(processor)

# Plot overall vector space
# plt.figure(1)
# visualizer.plot_vector_space(n_words=100, method='tsne')
# plt.show()

# plt.figure(2)
# visualizer.plot_vector_space(n_words=100, method='pca')
# plt.show()

# # Plot neighborhood of a specific word
# plt.figure(2)
# visualizer.plot_word_neighborhood('patient', n_neighbors=20)
# plt.show()

# # Plot suggested words for a context
# plt.figure(3)
# visualizer.plot_suggest_words(['patient', 'treatment', 'disease', 'risk'], n_suggestions=20)
# plt.show()

import numpy as np
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px
import plotly.graph_objects as go
from umap import UMAP
import networkx as nx
import pandas as pd

class AdvancedVisualizer:
    def __init__(self, processor):
        self.processor = processor
    
    def get_vectors_and_words(self, n_words=100, min_freq=5):
        """
        Get the top N most common words and their vectors
        
        Parameters:
        -----------
        n_words : int
            Number of words to include
        min_freq : int
            Minimum frequency threshold for words
        """
        # Get words sorted by frequency with minimum threshold
        words = [word for word, vocab in 
                sorted(self.processor.model.wv.key_to_index.items(),
                      key=lambda x: self.processor.model.wv.get_vecattr(x[0], "count"),
                      reverse=True)
                if self.processor.model.wv.get_vecattr(word, "count") >= min_freq][:n_words]
        
        # Get vectors for these words
        vectors = np.array([self.processor.model.wv[word] for word in words])
        
        # Normalize vectors for better clustering
        vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
        
        return vectors, words
        
    def plot_word_clusters(self, n_words=100, n_clusters=5, method='umap', 
                      interactive=False, min_freq=5, font_size=10, figsize=(20, 12)):
        """
        Plot word clusters using UMAP or t-SNE with interactive Plotly visualization
        and cluster decision boundaries
        """
        # Get vectors and words
        vectors, words = self.get_vectors_and_words(n_words, min_freq)
        
        # Ensure vectors are float64
        vectors = vectors.astype(np.float64)
        
        # Reduce dimensionality with improved parameters
        if method.lower() == 'umap':
            reducer = UMAP(
                n_components=2,
                n_neighbors=30,
                min_dist=0.3,
                metric='cosine',
                random_state=42
            )
        if method.lower() == 'pca':
            reducer = PCA(n_components=2)
        else:
            reducer = TSNE(
                n_components=2,
                perplexity=min(30, len(words)-1),
                metric='cosine',
                n_iter=1000,
                random_state=42
            )
            
        vectors_2d = reducer.fit_transform(vectors)
        # Ensure 2D vectors are float64
        vectors_2d = vectors_2d.astype(np.float64)
        
        # Perform clustering on the 2D reduced vectors
        kmeans_2d = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        clusters = kmeans_2d.fit_predict(vectors_2d)
        
        # Get cluster centers and find representative words
        centers = kmeans_2d.cluster_centers_
        cluster_labels = {}
        for i in range(n_clusters):
            cluster_words = [word for j, word in enumerate(words) if clusters[j] == i]
            frequencies = [self.processor.model.wv.get_vecattr(word, "count") 
                        for word in cluster_words]
            top_indices = np.argsort(frequencies)[-3:]
            representative_words = [cluster_words[idx] for idx in top_indices]
            cluster_labels[i] = ", ".join(representative_words)
        
        if interactive:
            # Create interactive Plotly visualization with boundaries
            df = pd.DataFrame({
                'x': vectors_2d[:, 0],
                'y': vectors_2d[:, 1],
                'word': words,
                'cluster': [f'Cluster {c}\n({cluster_labels[c]})' for c in clusters],
                'frequency': [self.processor.model.wv.get_vecattr(word, "count") 
                            for word in words]
            })
            
            # Create mesh grid for decision boundaries with explicit float64 dtype
            x_min, x_max = vectors_2d[:, 0].min() - 1, vectors_2d[:, 0].max() + 1
            y_min, y_max = vectors_2d[:, 1].min() - 1, vectors_2d[:, 1].max() + 1
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 100, dtype=np.float64),
                np.linspace(y_min, y_max, 100, dtype=np.float64)
            )
            
            # Create the mesh points array with explicit float64 dtype
            mesh_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float64)
            
            # Use the 2D KMeans model to predict boundaries
            Z = kmeans_2d.predict(mesh_points)
            Z = Z.reshape(xx.shape)
            
            # Create figure with adjusted size for legend
            fig = go.Figure()
            
            # Add contour for decision boundaries
            fig.add_trace(go.Contour(
                z=Z,
                x=np.linspace(x_min, x_max, 100),
                y=np.linspace(y_min, y_max, 100),
                colorscale='Viridis',
                showscale=False,
                opacity=0.3,
                line=dict(width=0),
                contours=dict(showlabels=False)
            ))
            
            # Add scatter points
            for i in range(n_clusters):
                mask = clusters == i
                fig.add_trace(go.Scatter(
                    x=vectors_2d[mask, 0],
                    y=vectors_2d[mask, 1],
                    text=df[mask]['word'],
                    mode='markers+text',
                    name=f'Cluster {i}\n({cluster_labels[i]})',
                    marker=dict(size=df[mask]['frequency'] / df['frequency'].max() * 30 + 5),
                    textposition='top center',
                    textfont=dict(size=10)
                ))
            
            # Adjust layout with better legend positioning
            fig.update_layout(
                height=1200,
                width=1600,
                showlegend=True,
                title=f'Word Clusters using {method.upper()}\n{n_clusters} clusters identified',
                title_x=0.5,
                title_y=0.95,
                template='plotly_white',
                legend=dict(
                    yanchor="top",
                    y=0.98,
                    xanchor="left",
                    x=1.02,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='rgba(0, 0, 0, 0.2)',
                    borderwidth=1
                ),
                margin=dict(r=250)
            )
            
            return fig
        
        else:
            # Create static Matplotlib visualization with boundaries
            plt.figure(figsize=figsize)
            main_ax = plt.axes([0.1, 0.1, 0.7, 0.8])
            
            # Create mesh grid for decision boundaries with explicit float64 dtype
            x_min, x_max = vectors_2d[:, 0].min() - 1, vectors_2d[:, 0].max() + 1
            y_min, y_max = vectors_2d[:, 1].min() - 1, vectors_2d[:, 1].max() + 1
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 100, dtype=np.float64),
                np.linspace(y_min, y_max, 100, dtype=np.float64)
            )
            
            # Create the mesh points array with explicit float64 dtype
            mesh_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float64)
            
            # Use the 2D KMeans model to predict boundaries
            Z = kmeans_2d.predict(mesh_points)
            Z = Z.reshape(xx.shape)
            
            # Plot decision boundaries
            plt.contourf(xx, yy, Z, alpha=0.3, cmap='tab20')
            
            # Create scatter plot with sized points by frequency
            frequencies = [self.processor.model.wv.get_vecattr(word, "count") for word in words]
            sizes = 100 * np.array(frequencies) / max(frequencies)
            scatter = plt.scatter(
                vectors_2d[:, 0],
                vectors_2d[:, 1],
                c=clusters,
                cmap='tab20',
                s=sizes
            )
            
            texts = []
            for i, word in enumerate(words):
                texts.append(plt.text(
                    vectors_2d[i, 0],
                    vectors_2d[i, 1],
                    word,
                    fontsize=font_size,
                    alpha=0.8
                ))
            
            adjust_text(
                texts,
                arrowprops=dict(arrowstyle='-', color='gray', alpha=0.8),
                expand_points=(1.5, 1.5),
                force_points=(0.1, 0.25)
            )
            
            legend_ax = plt.axes([0.85, 0.1, 0.15, 0.8])
            legend_ax.axis('off')
            
            legend_elements = []
            for i in range(n_clusters):
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                            markerfacecolor=f'C{i}', markersize=10,
                                            label=f'Cluster {i}\n{cluster_labels[i]}'))
            
            legend = legend_ax.legend(handles=legend_elements,
                                    title='Clusters',
                                    bbox_to_anchor=(0, 1),
                                    loc='upper left',
                                    frameon=True,
                                    fancybox=True,
                                    shadow=True,
                                    fontsize=10)
            legend.get_title().set_fontsize(12)
            
            main_ax.set_title(f'Word Clusters using {method.upper()}\n{n_clusters} clusters identified')
            main_ax.grid(True, alpha=0.3)
            
            return  plt
    def plot_word_hierarchy(self, n_words=50, figsize=(15, 10)):
        """Create a hierarchical clustering dendrogram of words"""
        # Get vectors and words
        vectors, words = self.get_vectors_and_words(n_words)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(vectors, method='ward')
        
        # Create dendrogram
        plt.figure(figsize=figsize)
        dendrogram(linkage_matrix, labels=words, leaf_rotation=90)
        plt.title('Hierarchical Clustering of Words')
        plt.xlabel('Words')
        plt.ylabel('Distance')
        plt.tight_layout()  # Prevent label cutoff
        return plt
    
    def plot_similarity_heatmap(self, words, figsize=(12, 10)):
        """Create a heatmap of word similarities"""
        # Calculate similarity matrix
        n_words = len(words)
        similarity_matrix = np.zeros((n_words, n_words))
        
        for i in range(n_words):
            for j in range(n_words):
                similarity_matrix[i, j] = self.processor.calculate_word_similarity(
                    words[i], words[j]
                )
        
        # Create heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(similarity_matrix, xticklabels=words, yticklabels=words,
                   cmap='YlOrRd', annot=True, fmt='.2f', square=True)
        plt.title('Word Similarity Heatmap')
        plt.tight_layout()
        return plt
    
    def plot_word_network(self, target_word, n_neighbors=10, threshold=0.5):
        """Create an interactive network visualization of word relationships"""
        # Get similar words
        similar_words = self.processor.find_similar_words(target_word, n_neighbors)
        
        if not similar_words:
            print(f"No similar words found for '{target_word}'")
            return None
            
        # Create network graph
        G = nx.Graph()
        G.add_node(target_word, size=20)
        
        # Add edges and nodes
        for word, similarity in similar_words:
            if similarity >= threshold:
                G.add_node(word, size=10)
                G.add_edge(target_word, word, weight=similarity)
        
        # Get node positions
        pos = nx.spring_layout(G)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_size.append(G.nodes[node]['size'])
            
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition='top center',
            marker=dict(
                showscale=True,
                size=node_size,
                colorscale='YlOrRd',
                reversescale=True,
                color=[20 if node == target_word else 10 for node in G.nodes()],
                line_width=2))
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=f'Word Relationship Network: {target_word}',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           width=800,
                           height=600
                       ))
        
        return fig
    
    def plot_topic_distribution(self, query_text, n_similar=10):
        """Visualize topic distribution for a query across similar documents"""
        # Get similar documents
        results = self.processor.query_similar_documents(query_text, n_similar)
        
        if not results:
            print(f"No similar documents found for query: '{query_text}'")
            return None
            
        # Extract similarities and titles
        titles = [doc['title'][:50] + '...' for doc, _ in results]
        similarities = [sim for _, sim in results]
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        bars = plt.barh(range(len(titles)), similarities)
        plt.yticks(range(len(titles)), titles)
        plt.xlabel('Similarity Score')
        plt.title(f'Document Similarity Distribution for Query: {query_text[:30]}...')
        
        # Add value labels
        for i, v in enumerate(similarities):
            plt.text(v, i, f'{v:.3f}', va='center')
            
        plt.tight_layout()
        return plt

# Example usage
# visualizer = AdvancedVisualizer(processor)

# # 1. Interactive word clusters
# cluster_fig = visualizer.plot_word_clusters(n_words=100, method='umap', interactive=True)
# cluster_fig.show()

# # 2. Hierarchical clustering
# visualizer.plot_word_hierarchy(n_words=30)
# plt.show()

# # 3. Similarity heatmap
# words_of_interest = ['age', 'patient', 'disease', 'treatment', 'study']
# visualizer.plot_similarity_heatmap(words_of_interest)
# plt.show()

# # 4. Word network
# network_fig = visualizer.plot_word_network('age', n_neighbors=8)
# network_fig.show()

# # 5. Topic distribution
# visualizer.plot_topic_distribution("cancer treatment")
# plt.show()