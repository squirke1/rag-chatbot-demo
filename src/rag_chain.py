"""
RAG Chain - Complete Retrieval-Augmented Generation Pipeline

This module orchestrates the full RAG workflow:
1. Retrieve relevant documents from vector store
2. Compose prompt with context
3. Generate answer using LLM

Usage:
    rag = RAGChain()
    answer = rag.query("What is RAG?")
"""

import os
import yaml
from typing import Dict, Any, List, Optional

# Import our custom modules
from src.retriever import load_vector_store, initialize_embedding_model, similarity_search, mmr_search
from src.prompt import create_messages_format
from src.configuration import resolve_config_path

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
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the RAG chain with all components.
        
        What happens during initialization:
        1. Load configuration
        2. Load FAISS vector store
        3. Initialize embedding model
        4. Initialize OpenAI client
        
        Args:
            config_path: Optional explicit path to configuration file
        """
        print("Initializing RAG Chain...")

        resolved_config_path = resolve_config_path(config_path)
        print(f"Loading configuration from: {resolved_config_path}")

        # Load configuration
        self.config = self._load_config(resolved_config_path)
        
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
    
    def retrieve(self, question: str, method: str = "similarity") -> List[Document]:
        """
        Retrieve relevant documents for a question.
        
        This is Step 1 of the RAG pipeline: RETRIEVE
        
        What happens here:
        1. Convert question to embedding vector
        2. Search FAISS index for similar documents
        3. Return top-k most relevant chunks
        
        Why separate retrieve method?
        - Can be used independently for testing
        - Allows inspection of retrieved context
        - Makes debugging easier
        
        Args:
            question: User's question
            method: Search method - "similarity" or "mmr"
            
        Returns:
            List of relevant Document objects
        """
        k = self.retrieval_config.get('top_k', 5)
        
        if method == "similarity":
            results = similarity_search(
                query=question,
                index=self.index,
                chunks=self.chunks,
                model=self.embedding_model,
                k=k
            )
        elif method == "mmr":
            lambda_mult = self.retrieval_config.get('mmr_lambda', 0.5)
            results = mmr_search(
                query=question,
                index=self.index,
                chunks=self.chunks,
                model=self.embedding_model,
                k=k,
                lambda_mult=lambda_mult
            )
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
        
        # Extract just the documents (without scores)
        documents = [doc for doc, score in results]
        return documents
    
    def query(
        self, 
        question: str, 
        method: str = "similarity",
        return_context: bool = False
    ) -> Dict[str, Any]:
        """
        Complete RAG query: Retrieve → Compose → Generate
        
        This is the full RAG pipeline in action:
        
        Step 1 - RETRIEVE: Get relevant documents from vector store
        Step 2 - COMPOSE: Format documents into prompt with question
        Step 3 - GENERATE: Send to LLM and get answer
        
        Why RAG works:
        - LLM gets specific, relevant context
        - Reduces hallucination (LLM can't make stuff up)
        - Answer is grounded in your documents
        - Can cite sources in the response
        
        Args:
            question: User's question
            method: Retrieval method - "similarity" or "mmr"
            return_context: If True, include retrieved documents in response
            
        Returns:
            Dictionary with:
            - answer: LLM's response
            - sources: List of source documents (optional)
            - context: Retrieved documents (if return_context=True)
        """
        print(f"\n{'='*60}")
        print("RAG QUERY PIPELINE")
        print(f"{'='*60}")
        print(f"Question: {question}")
        
        # Step 1: RETRIEVE relevant documents
        print(f"\nStep 1: Retrieving documents (method: {method})...")
        documents = self.retrieve(question, method=method)
        print(f"Retrieved {len(documents)} documents")
        
        # Step 2: COMPOSE prompt with context
        print("\nStep 2: Composing prompt...")
        messages = create_messages_format(question, documents)
        
        # Step 3: GENERATE answer using LLM
        print("\nStep 3: Generating answer...")
        try:
            response = self.client.chat.completions.create(
                model=self.llm_config['model'],
                messages=messages,  # type: ignore
                temperature=self.llm_config.get('temperature', 0.1),
                max_tokens=self.llm_config.get('max_tokens', 1000)
            )
            
            answer = response.choices[0].message.content
            print("Answer generated successfully!")
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            answer = f"Error: Could not generate answer. {str(e)}"
        
        # Prepare response
        result = {
            "answer": answer,
            "sources": [doc.metadata.get('source', 'Unknown') for doc in documents]
        }
        
        if return_context:
            result["context"] = documents
        
        print(f"{'='*60}\n")
        return result


def main():
    """
    CLI interface for testing the RAG chain.
    
    This allows you to:
    - Ask questions interactively
    - Test different retrieval methods
    - See retrieved sources
    - Verify the full pipeline works
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Question Answering System")
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["similarity", "mmr"],
        default="similarity",
        help="Retrieval method (default: similarity)"
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Show retrieved context documents"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file. Defaults to environment via RAG_ENV or RAG_CONFIG_PATH."
    )
    args = parser.parse_args()
    
    # Initialize RAG chain
    print("\n" + "="*60)
    print("RAG QUESTION ANSWERING SYSTEM")
    print("="*60)
    rag = RAGChain(config_path=args.config)
    
    # Query the system
    result = rag.query(
        question=args.question,
        method=args.method,
        return_context=args.show_context
    )
    
    # Display results
    print("\n" + "="*60)
    print("ANSWER")
    print("="*60)
    print(result['answer'])
    
    print("\n" + "="*60)
    print("SOURCES")
    print("="*60)
    for i, source in enumerate(result['sources'], 1):
        print(f"[{i}] {source}")
    
    # Show context if requested
    if args.show_context and 'context' in result:
        print("\n" + "="*60)
        print("RETRIEVED CONTEXT")
        print("="*60)
        for i, doc in enumerate(result['context'], 1):
            print(f"\n[{i}] Source: {doc.metadata.get('source', 'Unknown')}")
            content = doc.page_content
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"Content: {content}")
            print("-" * 60)
    
    print("\n" + "="*60)
    print("QUERY COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
