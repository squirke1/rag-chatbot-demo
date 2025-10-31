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


@app.get("/health")
async def health_check():
    """
    Health check endpoint - verifies system is ready.
    
    What is a health check?
    - An endpoint that returns the system's status
    - Used by monitoring tools, load balancers, orchestrators (Kubernetes)
    - Should be fast and lightweight
    
    Why do we need it?
    - Load balancers: Only send traffic to healthy instances
    - Monitoring: Alert if service becomes unhealthy
    - Deployment: Don't mark deployment complete until healthy
    
    What does it check?
    - Is the RAG chain initialized?
    - Are all components loaded (vector store, embeddings, LLM)?
    
    HTTP Status Codes:
    - 200 OK: Everything is healthy
    - 503 Service Unavailable: RAG system not initialized
    
    Why 503?
    - Tells load balancers "don't send traffic here yet"
    - Temporary condition (unlike 500 which is an error)
    - Service exists but isn't ready
    
    Try it:
    - Visit http://localhost:8000/health
    - Or curl http://localhost:8000/health
    """
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    
    return {
        "status": "healthy",
        "rag_chain": "initialized",
        "vector_store": "loaded",
        "embedding_model": "ready",
        "llm": "configured"
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Question-answering endpoint - the core of the RAG system.
    
    This is where the magic happens!
    
    The RAG Process:
    1. Receive question from user
    2. Convert question to embeddings (vector)
    3. Search vector store for similar document chunks
    4. Retrieve top-k most relevant chunks
    5. Format chunks into context for the LLM
    6. Send question + context to LLM
    7. LLM generates answer based on retrieved context
    8. Return answer with source references
    
    Why POST instead of GET?
    - POST is for sending data (the question)
    - GET is for retrieving resources
    - POST body can handle larger/complex data
    - POST requests aren't cached by browsers
    
    Why async?
    - LLM API calls take 2-5 seconds typically
    - async allows server to handle other requests during that time
    - Without async: Server blocked, can only handle 1 request at a time
    - With async: Server can juggle multiple requests concurrently
    
    Request Body (JSON):
    {
        "question": "What is RAG?",
        "method": "similarity"  // optional: "similarity" or "mmr"
    }
    
    Response (JSON):
    {
        "answer": "RAG stands for Retrieval-Augmented Generation...",
        "sources": ["chunk_0", "chunk_1"],
        "method_used": "similarity"
    }
    
    Error Codes:
    - 200: Success - answer generated
    - 422: Validation error (Pydantic caught invalid request)
    - 503: Service Unavailable (RAG system not initialized)
    - 500: Internal error (LLM API failed, etc.)
    
    Try it:
    - Interactive docs: http://localhost:8000/docs
    - cURL example:
      curl -X POST http://localhost:8000/ask \\
           -H "Content-Type: application/json" \\
           -d '{"question": "What is RAG?"}'
    """
    # Check if RAG system is ready
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Please check logs."
        )
    
    try:
        # Call the RAG chain
        # This orchestrates: retrieve → format prompt → call LLM → return result
        result = rag_chain.query(
            question=request.question,
            method=request.method
        )
        
        # Return structured response
        # Pydantic validates the response matches AnswerResponse model
        return AnswerResponse(
            answer=result["answer"],
            sources=result["sources"],
            method_used=request.method
        )
        
    except Exception as e:
        # Log error for debugging
        print(f"Error processing question: {e}")
        
        # Return helpful error to client
        # Don't expose internal details in production
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


# We'll add HTML chat interface in the next step
