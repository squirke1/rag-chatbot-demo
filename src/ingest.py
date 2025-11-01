"""
Document Ingestion Module

This module handles the first stage of the RAG pipeline:
1. Load documents from a directory
2. Split them into chunks
3. Generate embeddings
4. Create and save a FAISS vector index

Usage:
    python src/ingest.py --data-dir data
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse
import numpy as np
import pickle

# LangChain document loaders for different file types
from langchain_community.document_loaders import (
    TextLoader,           # For .txt files
    PyPDFLoader,          # For .pdf files
    UnstructuredMarkdownLoader,  # For .md files
    UnstructuredHTMLLoader,      # For .html files
    Docx2txtLoader,       # For .docx files
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Sentence transformers for generating embeddings
from sentence_transformers import SentenceTransformer

# FAISS for vector similarity search
import faiss

from src.configuration import resolve_config_path


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing all configuration settings
    """
    resolved_path = resolve_config_path(config_path)
    with open(resolved_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a summary of the configuration settings.
    Useful for debugging and understanding what settings are being used.
    """
    print("\n" + "="*60)
    print("DOCUMENT INGESTION CONFIGURATION")
    print("="*60)
    
    print(f"\nDocument Processing:")
    print(f"   Chunk Size: {config['document']['chunk_size']} characters")
    print(f"   Chunk Overlap: {config['document']['chunk_overlap']} characters")
    print(f"   Supported Formats: {', '.join(config['document']['supported_formats'])}")
    
    print(f"\nEmbeddings:")
    print(f"   Model: {config['embeddings']['model']}")
    print(f"   Dimension: {config['embeddings']['dimension']}")
    print(f"   Batch Size: {config['embeddings']['batch_size']}")
    
    print(f"\nVector Store:")
    print(f"   Type: {config['vector_store']['type'].upper()}")
    print(f"   Index Path: {config['vector_store']['index_path']}")
    print(f"   Similarity Metric: {config['vector_store']['similarity_metric']}")
    
    print("\n" + "="*60 + "\n")


def load_single_document(file_path: str) -> List[Document]:
    """
    Load a single document based on its file extension.
    
    Each file type requires a different loader:
    - .txt: Plain text files (simplest)
    - .pdf: Requires parsing PDF structure
    - .md: Markdown files with formatting
    - .html: Web pages with HTML tags
    - .docx: Microsoft Word documents
    
    Args:
        file_path: Path to the document file
        
    Returns:
        List of Document objects (usually one, but PDFs can have multiple pages)
        
    Raises:
        ValueError: If file extension is not supported
    """
    ext = Path(file_path).suffix.lower()
    
    # Map file extensions to their appropriate loaders
    loaders = {
        '.txt': TextLoader,
        '.pdf': PyPDFLoader,
        '.md': UnstructuredMarkdownLoader,
        '.html': UnstructuredHTMLLoader,
        '.docx': Docx2txtLoader,
    }
    
    if ext not in loaders:
        raise ValueError(f"Unsupported file type: {ext}")
    
    # Instantiate the loader and load the document
    loader = loaders[ext](file_path)
    return loader.load()


def load_documents(data_dir: str, supported_formats: List[str]) -> List[Document]:
    """
    Load all supported documents from a directory.
    
    This function:
    1. Scans the directory for files
    2. Filters by supported extensions
    3. Loads each document
    4. Combines them into a single list
    
    Args:
        data_dir: Directory containing documents
        supported_formats: List of file extensions to process (e.g., ['pdf', 'txt'])
        
    Returns:
        List of all loaded Document objects
    """
    documents = []
    data_path = Path(data_dir)
    
    # Convert format list to extensions (add dots)
    extensions = [f".{fmt}" for fmt in supported_formats]
    
    print(f"\nScanning directory: {data_dir}")
    print(f"Looking for files: {', '.join(extensions)}")
    
    # Walk through directory and subdirectories
    for file_path in data_path.rglob("*"):
        # Skip directories and hidden files
        if file_path.is_dir() or file_path.name.startswith('.'):
            continue
            
        # Check if file extension is supported
        if file_path.suffix.lower() in extensions:
            try:
                print(f"  Loading: {file_path.name}...", end=" ")
                docs = load_single_document(str(file_path))
                documents.extend(docs)
                print(f"OK ({len(docs)} page(s))")
            except Exception as e:
                print(f"FAILED - {str(e)}")
                continue
    
    print(f"\nTotal documents loaded: {len(documents)}")
    return documents


def split_documents(documents: List[Document], config: Dict[str, Any]) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.
    
    Why chunking?
    - LLMs have token limits (can't process entire books)
    - Smaller chunks = more precise retrieval
    - Balance: too small = loss of context, too large = irrelevant info
    
    Why overlap?
    - Maintains context across chunk boundaries
    - Example: "...models learn" | "learn from data..." 
    - Without overlap, "learn" is context-less in chunk 2
    
    Args:
        documents: List of loaded Document objects
        config: Configuration dictionary with chunk_size and chunk_overlap
        
    Returns:
        List of Document chunks with preserved metadata
    """
    # Get chunking parameters from config
    chunk_size = config['document']['chunk_size']
    chunk_overlap = config['document']['chunk_overlap']
    
    # Create text splitter
    # RecursiveCharacterTextSplitter tries to split on:
    # 1. Paragraphs (\n\n) first
    # 2. Then sentences (\n)
    # 3. Then words ( )
    # 4. Finally characters (last resort)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    # Split all documents
    chunks = text_splitter.split_documents(documents)
    
    return chunks


def generate_embeddings(chunks: List[Document], config: Dict[str, Any]) -> np.ndarray:
    """
    Generate embeddings for text chunks using sentence-transformers.
    
    What are embeddings?
    - Numerical representation of text meaning
    - Similar texts have similar vectors
    - Enables semantic search (meaning-based, not keyword-based)
    
    How it works:
    - Neural network converts text → 384-dimensional vector
    - Each dimension captures some aspect of meaning
    - Example: "dog" and "puppy" have similar vectors
    
    Why batch processing?
    - Process multiple chunks simultaneously
    - Much faster than one-by-one (15x speedup)
    - Efficient use of GPU/CPU resources
    
    Args:
        chunks: List of Document chunks to embed
        config: Configuration dictionary with model name and batch size
        
    Returns:
        NumPy array of shape (num_chunks, embedding_dimension)
        Example: (100, 384) = 100 chunks, each with 384-dimensional vector
    """
    model_name = config['embeddings']['model']
    batch_size = config['embeddings']['batch_size']
    
    print(f"\nInitializing embedding model: {model_name}")
    print("(First time will download the model - ~90MB)")
    
    # Load the sentence transformer model
    # This model converts text to 384-dimensional vectors
    model = SentenceTransformer(model_name)
    
    print(f"\nGenerating embeddings for {len(chunks)} chunks...")
    print(f"Processing in batches of {batch_size}")
    
    # Extract text content from Document objects
    texts = [chunk.page_content for chunk in chunks]
    
    # Generate embeddings in batches
    # show_progress_bar=True displays a progress bar
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print(f"\nEmbeddings generated!")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Size: {embeddings.nbytes / (1024*1024):.2f} MB")
    
    return embeddings


def create_vector_index(
    embeddings: np.ndarray, 
    chunks: List[Document], 
    config: Dict[str, Any]
) -> None:
    """
    Create a FAISS vector index and save it to disk along with document chunks.
    
    What is FAISS?
    - Facebook AI Similarity Search
    - Efficient library for searching similar vectors
    - Can search millions of vectors in milliseconds
    
    What is a vector index?
    - A data structure optimized for finding similar vectors
    - Like a database index, but for numerical vectors
    - Enables fast "semantic search" (search by meaning, not keywords)
    
    How it works:
    1. Store all embeddings in the index
    2. When user asks a question:
       - Convert question to embedding
       - Find K nearest neighbors in the index
       - Return the corresponding documents
    
    Why save chunks separately?
    - FAISS only stores vectors (numbers)
    - We need the original text to show users
    - Pickle file stores: text content, metadata, source info
    
    Args:
        embeddings: NumPy array of shape (num_chunks, embedding_dim)
        chunks: List of Document objects with text and metadata
        config: Configuration dictionary with index path
    """
    dimension = config['embeddings']['dimension']
    index_path = config['vector_store']['index_path']
    
    print(f"\nCreating FAISS index...")
    print(f"  Dimension: {dimension}")
    print(f"  Number of vectors: {len(embeddings)}")
    
    # Create FAISS index
    # IndexFlatL2 uses L2 (Euclidean) distance
    # "Flat" means it searches all vectors (exact search, no approximation)
    # This is perfect for smaller datasets (< 1 million vectors)
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to the index
    # FAISS requires float32 format
    embeddings_float32 = embeddings.astype('float32')
    index.add(embeddings_float32)  # type: ignore
    
    print(f"  Vectors added to index: {index.ntotal}")
    
    # Create directory if it doesn't exist
    os.makedirs(index_path, exist_ok=True)
    
    # Save FAISS index
    index_file = os.path.join(index_path, "faiss.index")
    faiss.write_index(index, index_file)
    print(f"\n  Saved FAISS index to: {index_file}")
    
    # Save document chunks (text + metadata)
    # We need this to retrieve the actual text content later
    chunks_file = os.path.join(index_path, "chunks.pkl")
    with open(chunks_file, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"  Saved {len(chunks)} chunks to: {chunks_file}")
    
    # Save configuration for reference
    config_file = os.path.join(index_path, "config.pkl")
    with open(config_file, 'wb') as f:
        pickle.dump(config, f)
    print(f"  Saved config to: {config_file}")
    
    print("\nVector index created successfully!")
    print(f"Index location: {index_path}")
    
    # Display index statistics
    print(f"\nIndex Statistics:")
    print(f"  Total vectors: {index.ntotal}")
    print(f"  Dimension: {dimension}")
    print(f"  Index type: Flat (exact search)")
    print(f"  Index size: {os.path.getsize(index_file) / (1024*1024):.2f} MB")


def main():
    """
    Main function to orchestrate the document ingestion process.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Ingest documents into vector database")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing documents to ingest (default: data)"
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
    
    # Display configuration summary
    print_config_summary(config)
    
    # Load documents from directory
    print("\n" + "="*60)
    print("LOADING DOCUMENTS")
    print("="*60)
    
    documents = load_documents(
        data_dir=args.data_dir,
        supported_formats=config['document']['supported_formats']
    )
    
    if len(documents) == 0:
        print("\nNo documents found!")
        print(f"Please add documents to the '{args.data_dir}' directory.")
        print(f"Supported formats: {', '.join(config['document']['supported_formats'])}")
        return
    
    # Split documents into chunks
    print("\n" + "="*60)
    print("SPLITTING DOCUMENTS INTO CHUNKS")
    print("="*60)
    
    print(f"Splitting {len(documents)} document(s) into chunks...")
    chunks = split_documents(documents, config)
    print(f"Created {len(chunks)} chunks")
    
    # Generate embeddings for chunks
    print("\n" + "="*60)
    print("GENERATING EMBEDDINGS")
    print("="*60)
    
    embeddings = generate_embeddings(chunks, config)
    
    # Create and save FAISS vector index
    print("\n" + "="*60)
    print("CREATING VECTOR INDEX")
    print("="*60)
    
    create_vector_index(embeddings, chunks, config)
    
    print("\n" + "="*60)
    print("INGESTION COMPLETE!")
    print("="*60)
    print(f"Documents processed: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Embeddings generated: {embeddings.shape[0]} vectors of {embeddings.shape[1]} dimensions")
    print(f"Index location: {config['vector_store']['index_path']}")
    print("\nYour RAG system is ready!")
    print("Next steps:")
    print("  - Implement retriever.py for semantic search")
    print("  - Test querying the vector database")


if __name__ == "__main__":
    main()
