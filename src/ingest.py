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
from typing import Dict, Any


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


def main():
    """
    Main function to orchestrate the document ingestion process.
    """
    # Load configuration
    print("Loading configuration...")
    config = load_config()
    
    # Display configuration summary
    print_config_summary(config)
    
    # TODO: Next steps will be implemented in subsequent iterations
    # - Load documents
    # - Split into chunks
    # - Generate embeddings
    # - Create vector index
    
    print("Configuration loaded successfully!")
    print("(Document loading will be implemented in the next step)")


if __name__ == "__main__":
    main()