"""
FastAPI Web Application for RAG Chatbot

This provides a web interface and REST API for the RAG system:
- REST API endpoint for asking questions
- Simple HTML/JavaScript chat interface
- CORS support for frontend development
- Health check endpoint

Usage:
    python app.py
    Then visit: http://localhost:8000
"""

import os
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, HTTPException  # type: ignore
from fastapi.responses import HTMLResponse  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from pydantic import BaseModel

# Our RAG system
from src.rag_chain import RAGChain


# Global variable to hold RAG chain instance
rag_chain: Optional[RAGChain] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    
    What is a lifespan?
    - Runs code when the app starts (before accepting requests)
    - Runs code when the app shuts down (cleanup)
    
    Why use it?
    - Load heavy models once at startup (not per request)
    - Expensive: Loading FAISS index, embedding model, etc.
    - Fast responses: Models already in memory
    
    Yields:
        Control back to FastAPI to handle requests
    """
    global rag_chain
    
    # Startup: Initialize RAG chain
    print("\n" + "="*60)
    print("INITIALIZING RAG SYSTEM")
    print("="*60)
    
    try:
        rag_chain = RAGChain(config_path="configs/rag.yaml")
        print("\nRAG system ready!")
    except Exception as e:
        print(f"\nError initializing RAG system: {e}")
        print("Make sure:")
        print("1. Vector index exists (run src/ingest.py first)")
        print("2. OPENAI_API_KEY is set in environment")
        raise
    
    print("="*60 + "\n")
    
    yield  # App runs and handles requests here
    
    # Shutdown: Cleanup if needed
    print("\nShutting down RAG system...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="RAG Chatbot API",
    description="Retrieval-Augmented Generation chatbot for document Q&A",
    version="1.0.0",
    lifespan=lifespan
)


# Add CORS middleware
# What is CORS?
# - Cross-Origin Resource Sharing
# - Allows frontend apps on different domains to call this API
# - Important for development (frontend on localhost:3000, API on localhost:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QuestionRequest(BaseModel):
    """
    Request model for asking questions.
    
    Pydantic models provide:
    - Automatic validation (ensures 'question' is a string)
    - Type checking
    - API documentation (shows up in /docs)
    """
    question: str
    method: str = "similarity"  # Default to similarity search
    

class AnswerResponse(BaseModel):
    """Response model for answers."""
    answer: str
    sources: list[str]
    method_used: str


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """
    Root endpoint - provides basic API information.
    
    What is this?
    - The "home page" of the API
    - Returns JSON with API metadata
    
    Why have a root endpoint?
    - Quick way to check if API is running
    - Provides discovery of other endpoints
    - Shows version information
    
    Why async?
    - FastAPI best practice - all endpoints should be async
    - Allows handling multiple requests concurrently
    - Even simple endpoints benefit from async
    
    What does it return?
    - API name and version
    - Status (always "running" if you get a response)
    - Links to important endpoints
    
    Try it:
    - Visit http://localhost:8000/
    - Or curl http://localhost:8000/
    """
    return {
        "name": "RAG Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",        # Interactive API documentation
            "health": "/health",    # Health check for monitoring
            "ask": "/ask"           # Main Q&A endpoint
        }
    }


# We'll add more endpoints in the next steps
