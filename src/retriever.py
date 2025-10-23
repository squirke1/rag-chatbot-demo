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


# We'll add similarity_search() in the next step
# We'll add mmr_search() after that
# We'll add print_results() and main() at the end
