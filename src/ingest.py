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
from typing import Dict, Any, List
import argparse
import numpy as np

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
        default="configs/rag.yaml",
        help="Path to configuration file (default: configs/rag.yaml)"
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
    
    # TODO: Next step will be implemented in subsequent iteration
    # - Create FAISS vector index
    # - Save index to disk
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Embeddings generated: {embeddings.shape[0]} vectors of {embeddings.shape[1]} dimensions")
    print("\nNext step (coming soon):")
    print("  - Create and save FAISS vector index")


if __name__ == "__main__":
    main()