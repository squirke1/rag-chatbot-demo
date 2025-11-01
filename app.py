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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# FastAPI imports
from fastapi import FastAPI, HTTPException, Request  # type: ignore
from fastapi.responses import HTMLResponse  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from fastapi.templating import Jinja2Templates  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from pydantic import BaseModel

# Our RAG system
from src.rag_chain import RAGChain
from src.configuration import resolve_config_path


# Global variable to hold RAG chain instance
rag_chain: Optional[RAGChain] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Initializes RAG chain on startup and cleans up on shutdown.
    """
    global rag_chain
    
    # Startup: Initialize RAG chain
    print("\n" + "="*60)
    print("INITIALIZING RAG SYSTEM")
    print("="*60)
    
    try:
        config_path = resolve_config_path()
        print(f"Using configuration: {config_path}")
        rag_chain = RAGChain(config_path=config_path)
        print("\nRAG system ready!")
    except Exception as e:
        print(f"\nError initializing RAG system: {e}")
        print("Make sure:")
        print("1. Vector index exists (run src/ingest.py first)")
        print("2. OPENAI_API_KEY is set in environment")
        raise
    
    print("="*60 + "\n")
    
    yield
    
    # Shutdown: Cleanup if needed
    print("\nShutting down RAG system...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="RAG Chatbot API",
    description="Retrieval-Augmented Generation chatbot for document Q&A",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    question: str
    method: str = "similarity"
    

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
    """Root endpoint - returns API information and available endpoints."""
    return {
        "name": "RAG Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "ask": "/ask"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    Returns 200 if system is healthy, 503 if not initialized.
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
    Main question-answering endpoint using RAG.
    
    Retrieves relevant documents and generates answers using LLM.
    Returns answer with source references.
    """
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Please check logs."
        )
    
    try:
        result = rag_chain.query(
            question=request.question,
            method=request.method
        )
        
        return AnswerResponse(
            answer=result["answer"],
            sources=result["sources"],
            method_used=request.method
        )
        
    except Exception as e:
        print(f"Error processing question: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


@app.get("/chat", response_class=HTMLResponse)
async def chat_interface(request: Request):
    """Serve the HTML chat interface."""
    return templates.TemplateResponse("chat.html", {"request": request})


# We'll add server startup in the next step


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
