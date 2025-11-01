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
async def chat_interface():
    """Serve the HTML chat interface."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Chatbot</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            
            .chat-container {
                background: white;
                border-radius: 12px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                width: 100%;
                max-width: 800px;
                height: 600px;
                display: flex;
                flex-direction: column;
            }
            
            .chat-header {
                background: #667eea;
                color: white;
                padding: 20px;
                border-radius: 12px 12px 0 0;
            }
            
            .chat-header h1 {
                font-size: 24px;
                font-weight: 600;
            }
            
            .chat-header p {
                font-size: 14px;
                opacity: 0.9;
                margin-top: 5px;
            }
            
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                display: flex;
                flex-direction: column;
                gap: 15px;
            }
            
            .message {
                display: flex;
                gap: 10px;
                max-width: 80%;
                animation: slideIn 0.3s ease;
            }
            
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .message.user {
                align-self: flex-end;
                flex-direction: row-reverse;
            }
            
            .message-avatar {
                width: 36px;
                height: 36px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 18px;
                flex-shrink: 0;
            }
            
            .message.user .message-avatar {
                background: #667eea;
            }
            
            .message.assistant .message-avatar {
                background: #e0e0e0;
            }
            
            .message-content {
                background: #f5f5f5;
                padding: 12px 16px;
                border-radius: 12px;
                line-height: 1.5;
            }
            
            .message.user .message-content {
                background: #667eea;
                color: white;
            }
            
            .message-sources {
                margin-top: 8px;
                padding-top: 8px;
                border-top: 1px solid rgba(0,0,0,0.1);
                font-size: 12px;
                opacity: 0.8;
            }
            
            .chat-input-container {
                padding: 20px;
                border-top: 1px solid #e0e0e0;
                display: flex;
                gap: 10px;
            }
            
            #questionInput {
                flex: 1;
                padding: 12px 16px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 14px;
                outline: none;
                transition: border-color 0.3s;
            }
            
            #questionInput:focus {
                border-color: #667eea;
            }
            
            #sendButton {
                padding: 12px 24px;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
                cursor: pointer;
                transition: background 0.3s;
            }
            
            #sendButton:hover:not(:disabled) {
                background: #5568d3;
            }
            
            #sendButton:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            
            .loading {
                display: inline-block;
            }
            
            .loading::after {
                content: '...';
                animation: dots 1.5s steps(4, end) infinite;
            }
            
            @keyframes dots {
                0%, 20% { content: '.'; }
                40% { content: '..'; }
                60%, 100% { content: '...'; }
            }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <h1>RAG Chatbot</h1>
                <p>Ask questions about your documents</p>
            </div>
            
            <div class="chat-messages" id="chatMessages">
                <div class="message assistant">
                    <div class="message-avatar">🤖</div>
                    <div class="message-content">
                        Hello! I'm your RAG chatbot. Ask me anything about the documents in the knowledge base.
                    </div>
                </div>
            </div>
            
            <div class="chat-input-container">
                <input 
                    type="text" 
                    id="questionInput" 
                    placeholder="Type your question here..."
                    onkeypress="if(event.key === 'Enter') sendMessage()"
                />
                <button id="sendButton" onclick="sendMessage()">Send</button>
            </div>
        </div>
        
        <script>
            const chatMessages = document.getElementById('chatMessages');
            const questionInput = document.getElementById('questionInput');
            const sendButton = document.getElementById('sendButton');
            
            function addMessage(text, isUser, sources = null) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
                
                const avatar = document.createElement('div');
                avatar.className = 'message-avatar';
                avatar.textContent = isUser ? '👤' : '🤖';
                
                const content = document.createElement('div');
                content.className = 'message-content';
                content.textContent = text;
                
                if (sources && sources.length > 0) {
                    const sourcesDiv = document.createElement('div');
                    sourcesDiv.className = 'message-sources';
                    sourcesDiv.textContent = `Sources: ${sources.join(', ')}`;
                    content.appendChild(sourcesDiv);
                }
                
                messageDiv.appendChild(avatar);
                messageDiv.appendChild(content);
                chatMessages.appendChild(messageDiv);
                
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            async function sendMessage() {
                const question = questionInput.value.trim();
                if (!question) return;
                
                // Add user message
                addMessage(question, true);
                questionInput.value = '';
                
                // Disable input while processing
                sendButton.disabled = true;
                questionInput.disabled = true;
                sendButton.innerHTML = '<span class="loading">Thinking</span>';
                
                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            question: question,
                            method: 'similarity'
                        })
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    const data = await response.json();
                    addMessage(data.answer, false, data.sources);
                    
                } catch (error) {
                    console.error('Error:', error);
                    addMessage('Sorry, there was an error processing your question. Please try again.', false);
                } finally {
                    sendButton.disabled = false;
                    questionInput.disabled = false;
                    sendButton.textContent = 'Send';
                    questionInput.focus();
                }
            }
            
            // Focus input on load
            questionInput.focus();
        </script>
    </body>
    </html>
    """


# We'll add server startup in the next step


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
