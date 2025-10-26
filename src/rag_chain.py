"""
RAG Chain - Complete Retrieval-Augmented Generation Pipeline

This module orchestrates the full RAG workflow:
1. Retrieve relevant documents from vector store
2. Compose prompt with context
3. Generate answer using LLM

Usage:
    rag = RAGChain(config_path="configs/rag.yaml")
    answer = rag.query("What is RAG?")
"""

import os
import yaml
from typing import Dict, Any, List, Optional

# Import our custom modules
from retriever import load_vector_store, initialize_embedding_model, similarity_search, mmr_search
from prompt import create_messages_format

# OpenAI for LLM
from openai import OpenAI

# LangChain document structure
from langchain_core.documents import Document


class RAGChain:
    """
    Complete RAG pipeline that integrates retrieval and generation.
    
    The RAG Chain combines:
    - Document retrieval (from FAISS)
    - Prompt engineering (context injection)
    - LLM generation (OpenAI)
    
    This is the "brain" of the RAG system.
    """
    
    def __init__(self, config_path: str = "configs/rag.yaml"):
        """
        Initialize the RAG chain with all components.
        
        What happens during initialization:
        1. Load configuration
        2. Load FAISS vector store
        3. Initialize embedding model
        4. Initialize OpenAI client
        
        Args:
            config_path: Path to configuration file
        """
        print("Initializing RAG Chain...")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Load vector store and embedding model
        print("\nLoading vector store...")
        index_path = self.config['vector_store']['index_path']
        self.index, self.chunks, self.stored_config = load_vector_store(index_path)
        self.embedding_model = initialize_embedding_model(self.stored_config)
        
        # Initialize OpenAI client
        print("\nInitializing LLM...")
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it: export OPENAI_API_KEY='your-key-here'"
            )
        self.client = OpenAI(api_key=api_key)
        self.llm_config = self.config['llm']
        
        # Retrieval settings
        self.retrieval_config = self.config['retrieval']
        
        print("\nRAG Chain initialized successfully!")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)


# We'll add query() method in the next step
