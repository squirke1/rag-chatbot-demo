"""
Prompt Template Management

This module handles prompt engineering for the RAG system:
1. System prompts that define the AI's behavior
2. User prompts that inject context and questions
3. Context formatting to structure retrieved documents

Usage:
    from prompt import create_rag_prompt
    prompt = create_rag_prompt(question, context_docs)
"""

from typing import List
from langchain_core.documents import Document


# System prompt defines the AI assistant's role and behavior
SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context.

Your responsibilities:
- Answer questions using ONLY the information from the context provided
- If the context doesn't contain enough information, say so honestly
- Be concise and precise in your answers
- Cite specific parts of the context when possible
- Do not make up information or use outside knowledge

Guidelines:
- If the answer is not in the context, respond: "I don't have enough information in the provided context to answer that question."
- If the context is relevant but incomplete, provide what you can and acknowledge the limitations
- Maintain a professional and helpful tone"""


def format_context(documents: List[Document]) -> str:
    """
    Format retrieved documents into a context string.
    
    Takes the list of retrieved documents and formats them into
    a readable context block for the LLM.
    
    Why format context?
    - Provides clear structure for the LLM
    - Numbers each chunk for reference
    - Includes source metadata when available
    - Makes it easy for the LLM to cite sources
    
    Args:
        documents: List of Document objects from the retriever
        
    Returns:
        Formatted context string
    """
    if not documents:
        return "No relevant context found."
    
    context_parts = []
    for i, doc in enumerate(documents, 1):
        # Get source information if available
        source = doc.metadata.get('source', 'Unknown source')
        
        # Format each chunk with number and source
        chunk = f"[{i}] Source: {source}\n{doc.page_content}"
        context_parts.append(chunk)
    
    # Join all chunks with separators
    return "\n\n---\n\n".join(context_parts)


# We'll add create_rag_prompt() in the next step
