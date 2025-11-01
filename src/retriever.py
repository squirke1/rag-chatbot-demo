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
from typing import Dict, Any, List, Tuple, Optional

# Sentence transformers for query embeddings
from sentence_transformers import SentenceTransformer

# FAISS for vector similarity search
import faiss

from src.configuration import resolve_config_path

# LangChain document structure
from langchain_core.documents import Document


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Optional path to the YAML configuration file
        
    Returns:
        Dictionary containing all configuration settings
    """
    resolved_path = resolve_config_path(config_path)
    with open(resolved_path, 'r') as f:
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


def mmr_search(
    query: str,
    index: faiss.Index,
    chunks: List[Document],
    model: SentenceTransformer,
    k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5
) -> List[Tuple[Document, float]]:
    """
    Maximal Marginal Relevance (MMR) search for diverse results.
    
    What is MMR?
    - Balances relevance AND diversity
    - Avoids returning multiple similar chunks
    - Useful when chunks might be redundant
    
    Example scenario:
    - Query: "What is Python?"
    - Without MMR: Might return 5 chunks all from the same section
    - With MMR: Returns chunks from different sections/topics
    
    How it works:
    1. Fetch more candidates than needed (fetch_k = 20)
    2. Select most relevant chunk first
    3. For remaining selections, balance:
       - Similarity to query (relevance)
       - Dissimilarity to already selected chunks (diversity)
    
    Lambda parameter:
    - lambda = 1.0: Pure relevance (same as similarity search)
    - lambda = 0.5: Balanced (recommended)
    - lambda = 0.0: Pure diversity (might return irrelevant docs)
    
    Args:
        query: User's search query
        index: FAISS index
        chunks: Document chunks
        model: Embedding model
        k: Number of final results to return
        fetch_k: Number of candidates to fetch initially
        lambda_mult: Balance between relevance (1.0) and diversity (0.0)
        
    Returns:
        List of (Document, distance) tuples with diverse results
    """
    print(f"\nMMR Query: '{query}'")
    print(f"Fetching {fetch_k} candidates, selecting {k} diverse results...")
    print(f"Lambda: {lambda_mult} (1.0=relevance, 0.0=diversity)")
    
    # Step 1: Get query embedding
    query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
    
    # Step 2: Fetch more candidates than we need
    distances, indices = index.search(query_embedding, fetch_k)  # type: ignore
    distances = distances[0]
    indices = indices[0]
    
    # Step 3: Get embeddings of candidate chunks
    # We need these to measure diversity between candidates
    candidate_embeddings = []
    candidate_chunks = []
    for idx in indices:
        if idx < len(chunks):
            candidate_chunks.append(chunks[idx])
            # Reconstruct the embedding from the index
            candidate_embeddings.append(index.reconstruct(int(idx)))  # type: ignore
    
    candidate_embeddings = np.array(candidate_embeddings)
    
    # Step 4: MMR selection algorithm
    selected_indices = []
    selected_embeddings = []
    
    # Always select the first (most relevant) document
    selected_indices.append(0)
    selected_embeddings.append(candidate_embeddings[0])
    
    # Step 5: Select remaining k-1 documents using MMR formula
    while len(selected_indices) < k and len(selected_indices) < len(candidate_chunks):
        best_score = -float('inf')
        best_idx = None
        
        # Evaluate each non-selected candidate
        for i, candidate_emb in enumerate(candidate_embeddings):
            if i in selected_indices:
                continue
            
            # Relevance: How similar is this to the query?
            # Negative L2 distance (higher = more similar)
            relevance = -np.linalg.norm(query_embedding[0] - candidate_emb)
            
            # Diversity: How different is this from already selected docs?
            # Find maximum similarity to any selected doc
            if len(selected_embeddings) > 0:
                similarities = [
                    -np.linalg.norm(candidate_emb - sel_emb)
                    for sel_emb in selected_embeddings
                ]
                max_similarity = max(similarities)
            else:
                max_similarity = 0
            
            # MMR score: balance relevance and diversity
            # Higher lambda = favor relevance
            # Lower lambda = favor diversity
            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_similarity
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        
        # Add the best candidate to selected set
        if best_idx is not None:
            selected_indices.append(best_idx)
            selected_embeddings.append(candidate_embeddings[best_idx])
    
    # Step 6: Return selected chunks with their distances
    results = []
    for idx in selected_indices:
        chunk = candidate_chunks[idx]
        distance = float(np.linalg.norm(query_embedding[0] - candidate_embeddings[idx]))
        results.append((chunk, distance))
    
    print(f"Selected {len(results)} diverse results")
    return results


def print_results(results: List[Tuple[Document, float]], show_content: bool = True) -> None:
    """
    Display search results in a readable format.
    
    Args:
        results: List of (Document, score) tuples
        show_content: Whether to display full chunk content
    """
    print("\n" + "="*60)
    print("SEARCH RESULTS")
    print("="*60)
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n[{i}] Score: {score:.4f}")
        
        # Show metadata if available
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        
        # Show content
        if show_content:
            content = doc.page_content
            # Truncate if too long
            if len(content) > 300:
                content = content[:300] + "..."
            print(f"Content: {content}")
        
        print("-" * 60)


def main():
    """
    Main function for command-line retrieval testing.
    
    This CLI allows you to test the retriever with different:
    - Queries
    - Search methods (similarity vs MMR)
    - Number of results (k)
    - Lambda values for MMR
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Retrieve relevant documents using semantic search")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["similarity", "mmr"],
        default="similarity",
        help="Search method: similarity or mmr (default: similarity)"
    )
    parser.add_argument(
        "--lambda-mult",
        type=float,
        default=0.5,
        help="MMR lambda multiplier for relevance vs diversity (default: 0.5)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file. Defaults to environment via RAG_ENV or RAG_CONFIG_PATH."
    )
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    index_path = config['vector_store']['index_path']
    
    # Load vector store
    print("\n" + "="*60)
    print("LOADING VECTOR STORE")
    print("="*60)
    index, chunks, stored_config = load_vector_store(index_path)
    
    # Initialize embedding model
    model = initialize_embedding_model(stored_config)
    
    # Perform search
    print("\n" + "="*60)
    print(f"SEARCHING ({args.method.upper()} METHOD)")
    print("="*60)
    
    if args.method == "similarity":
        results = similarity_search(args.query, index, chunks, model, args.k)
    elif args.method == "mmr":
        results = mmr_search(
            args.query, 
            index, 
            chunks, 
            model, 
            k=args.k,
            lambda_mult=args.lambda_mult
        )
    
    # Display results
    print_results(results)
    
    print("\n" + "="*60)
    print("RETRIEVAL COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
