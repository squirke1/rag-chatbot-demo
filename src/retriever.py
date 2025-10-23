"""
Document Retrieval Module

This module handles semantic search over the vector database:
1. Load FAISS index and document chunks
2. Convert queries to embeddings
3. Find similar documents using various strategies
4. Return relevant chunks with scores

Usage:
    python src/retriever.py --query "What is RAG?" --k 5
"""

import os
import yaml
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Sentence transformers for query embeddings
from sentence_transformers import SentenceTransformer

# FAISS for vector similarity search
import faiss

# LangChain document structure
from langchain_core.documents import Document


def load_config(config_path: str = "configs/rag.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing all configuration settings
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_vector_store(index_path: str) -> Tuple[faiss.Index, List[Document], Dict[str, Any]]:
    """
    Load the FAISS index, document chunks, and configuration from disk.
    
    What we're loading:
    - FAISS index: The vector database for fast similarity search
    - Chunks: The actual text content with metadata
    - Config: Settings used to create the index
    
    Why load all three?
    - Index: Provides the vector IDs of similar documents
    - Chunks: Contains the actual text to return to users
    - Config: Ensures we use the same embedding model
    
    Args:
        index_path: Directory containing the vector store files
        
    Returns:
        Tuple of (faiss_index, document_chunks, config)
        
    Raises:
        FileNotFoundError: If index files don't exist
    """
    index_file = os.path.join(index_path, "faiss.index")
    chunks_file = os.path.join(index_path, "chunks.pkl")
    config_file = os.path.join(index_path, "config.pkl")
    
    # Check if files exist
    if not os.path.exists(index_file):
        raise FileNotFoundError(
            f"FAISS index not found at {index_file}. "
            "Please run src/ingest.py first to create the index."
        )
    
    # Load FAISS index
    print(f"Loading FAISS index from {index_file}...")
    index = faiss.read_index(index_file)
    print(f"  Loaded {index.ntotal} vectors")
    
    # Load document chunks
    print(f"Loading chunks from {chunks_file}...")
    with open(chunks_file, 'rb') as f:
        chunks = pickle.load(f)
    print(f"  Loaded {len(chunks)} chunks")
    
    # Load configuration
    print(f"Loading config from {config_file}...")
    with open(config_file, 'rb') as f:
        config = pickle.load(f)
    
    return index, chunks, config


def initialize_embedding_model(config: Dict[str, Any]) -> SentenceTransformer:
    """
    Initialize the embedding model for converting queries to vectors.
    
    Why the same model?
    - Must use the SAME model that created the document embeddings
    - Different models produce incompatible vector spaces
    - Like trying to compare temperatures in Celsius vs Fahrenheit
    
    Args:
        config: Configuration dictionary with model name
        
    Returns:
        Initialized SentenceTransformer model
    """
    model_name = config['embeddings']['model']
    print(f"\nInitializing embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    return model


def similarity_search(
    query: str,
    index: faiss.Index,
    chunks: List[Document],
    model: SentenceTransformer,
    k: int = 5
) -> List[Tuple[Document, float]]:
    """
    Perform similarity search to find most relevant documents.
    
    How it works:
    1. Convert query text to embedding vector
    2. Search FAISS index for k nearest neighbors
    3. Retrieve corresponding document chunks
    4. Return chunks with similarity scores
    
    Distance vs Similarity:
    - FAISS returns L2 distance (lower = more similar)
    - Distance of 0 = identical vectors
    - Distance increases as vectors differ
    
    Args:
        query: User's search query
        index: FAISS index containing document embeddings
        chunks: List of document chunks (same order as index)
        model: Embedding model for query encoding
        k: Number of results to return
        
    Returns:
        List of (Document, distance_score) tuples, sorted by relevance
    """
    print(f"\nQuery: '{query}'")
    print(f"Searching for top {k} similar chunks...")
    
    # Step 1: Convert query to embedding
    # The model returns a 384-dimensional vector for our query
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Step 2: FAISS requires float32 type
    query_embedding = query_embedding.astype('float32')
    
    # Step 3: Search the index
    # Returns: distances (L2 distances), indices (positions in index)
    # Lower distance = more similar
    distances, indices = index.search(query_embedding, k)  # type: ignore
    
    # Step 4: Extract results (FAISS returns 2D arrays)
    distances = distances[0]  # Get first row (we only searched one query)
    indices = indices[0]      # Get first row
    
    # Step 5: Retrieve corresponding chunks
    results = []
    for idx, distance in zip(indices, distances):
        if idx < len(chunks):  # Ensure index is valid
            chunk = chunks[idx]
            results.append((chunk, float(distance)))
    
    print(f"Found {len(results)} results")
    return results


# We'll add mmr_search() in the next step
# We'll add print_results() and main() after that
