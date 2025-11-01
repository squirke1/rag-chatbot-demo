# RAG Chatbot Demo

A modular implementation of a Retrieval-Augmented Generation (RAG) system for question-answering over custom document collections.

## Overview

This project implements a complete RAG pipeline with the following capabilities:
- Multi-format document ingestion (PDF, TXT, Markdown, HTML, DOCX)
- Semantic search using vector embeddings
- Multiple retrieval strategies (similarity, MMR, hybrid)
- Context-aware response generation
- Performance evaluation framework
- RESTful API and web interface

## Features

- Multi-format document support (PDF, TXT, Markdown, HTML, DOCX)
- Advanced retrieval strategies: similarity search, MMR, and hybrid approaches
- YAML-based configuration system
- Built-in evaluation metrics
- FastAPI backend with RESTful endpoints
- Modular, extensible architecture

## Project Structure

```
rag-chatbot-demo/
├── configs/
│   └── rag.yaml              # System configuration
├── data/                     # Document storage
├── notebooks/                # Jupyter notebooks for experimentation
├── src/
│   ├── ingest.py            # Document loading and indexing
│   ├── retriever.py         # Search strategies (top-k, MMR, hybrid)
│   ├── prompt.py            # Prompt templates
│   ├── rag_chain.py         # RAG pipeline orchestration
│   └── eval.py              # Evaluation metrics
├── app.py                   # FastAPI web application
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip or conda for package management

### Installation

Clone the repository:
```bash
git clone https://github.com/squirke1/rag-chatbot-demo.git
cd rag-chatbot-demo
```

Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies (requirements file to be added):
```bash
pip install -r requirements.txt
```

Configure API keys:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Configuration

Environment-specific configuration files live under `configs/`:

- `configs/rag.dev.yaml` – default for local development, verbose logging, localhost binding.
- `configs/rag.prod.yaml` – tuned for production deployments, stricter thresholds, public binding.

The active file is resolved in this order:

1. Explicit `--config /path/to/file.yaml` CLI argument.
2. `RAG_CONFIG_PATH=/absolute/or/relative/path.yaml`
3. `RAG_ENV=dev|prod` (defaults to `dev`).

Examples:

```bash
# Use the production settings
export RAG_ENV=prod

# Point to a completely custom configuration
export RAG_CONFIG_PATH=/opt/rag/configs/custom.yaml
```

Feel free to duplicate one of the provided YAML files if you need additional environments (e.g., `rag.test.yaml`).

## Usage

### Document Ingestion

Place documents in the `data/` directory and run ingestion:
```bash
python src/ingest.py --data-dir data
```

This process:
- Loads documents from the specified directory
- Splits text into chunks with configured size and overlap
- Generates embeddings using the configured model
- Creates a FAISS vector index for similarity search

### Running the Application

*Note: Web interface is currently in development. Use the CLI for now.*

Query via command line:
```bash
python src/rag_chain.py --question "What is RAG?"
```

Once the web app is complete, you'll be able to start the server:
```bash
uvicorn app:app --reload
```

## Branching Strategy

Local branches `dev` and `test` are available alongside `main`:

- Develop features on `dev`, promote stabilized changes to `test`, and release from `main`.
- Push the new branches upstream once authenticated:  
  `git push -u origin dev` and `git push -u origin test`
- Protect the branches in GitHub as desired (e.g., required reviews or checks).

And access the web interface at `http://localhost:8000`

### Command Line Interface

Query the RAG system directly:
```bash
python src/rag_chain.py --question "What is RAG?"
```

Use MMR retrieval for diversity:
```bash
python src/rag_chain.py --question "What is RAG?" --method mmr
```

Test retrieval only:
```bash
python src/retriever.py --query "RAG systems" --k 5 --method similarity
```

Show retrieved context:
```bash
python src/rag_chain.py --question "What is RAG?" --show-context
```

## Architecture

```
Documents → Ingest → Vector DB → Retriever → RAG Chain → Response
             ↓          ↓           ↓           ↓
           Split     FAISS     Similarity   LLM + Context
          Embed     Index       Search      Generation
```

## Implementation Status

**Completed:**
- Project structure and configuration system
- Document ingestion pipeline (`src/ingest.py`)
  - Multi-format document loading
  - Text chunking with overlap
  - Batch embedding generation
  - FAISS vector index creation
- Retrieval system (`src/retriever.py`)
  - Similarity search (top-k)
  - MMR (Maximal Marginal Relevance)
  - CLI for testing retrieval
- Prompt management (`src/prompt.py`)
  - System and user prompt templates
  - Context formatting
  - OpenAI messages format support
- RAG chain (`src/rag_chain.py`)
  - End-to-end question answering
  - Retrieve → Compose → Generate pipeline
  - CLI for testing Q&A

**In Progress:**
- Web application (`app.py`)
  - FastAPI setup with lifespan management
  - CORS middleware
  - Request/response models

**Planned:**
- Web interface (HTML/JavaScript UI)
- Evaluation framework (`src/eval.py`) - optional

## Technical Details

### Document Processing
- Chunk size: 1000 characters
- Chunk overlap: 200 characters
- Supported formats: PDF, TXT, MD, HTML, DOCX

### Embeddings
- Model: sentence-transformers/all-MiniLM-L6-v2
- Vector dimension: 384
- Batch size: 32

### Vector Store
- Implementation: FAISS (Facebook AI Similarity Search)
- Similarity metric: Cosine similarity
- Storage: Local filesystem

### Retrieval
- Default strategy: Similarity search
- Top-k: 5 documents
- Score threshold: 0.7
- Alternative strategies: MMR, hybrid

### Language Model
- Default provider: OpenAI
- Model: GPT-3.5-turbo
- Temperature: 0.1
- Max tokens: 1000

## Development Approach

This project follows an incremental development process:

1. Configuration - Define system parameters
2. Document ingestion - Text processing and embedding generation
3. Retrieval - Vector search implementation
4. Prompt engineering - Context formatting
5. Chain integration - End-to-end pipeline
6. Evaluation - Performance metrics
7. API development - Web interface

## References

- [LangChain Documentation](https://python.langchain.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [FastAPI](https://fastapi.tiangolo.com/)
- [RAG Papers and Research](https://arxiv.org/abs/2005.11401)

## Future Enhancements

- Support for more LLM providers (Anthropic, local models)
- Advanced chunking strategies
- Multi-modal document support (images, tables)
- Conversation memory and follow-up questions
- Cloud vector database integration
- Docker containerization
- Comprehensive test suite

## Contact

For questions or feedback, please open an issue on GitHub.

---

*Last Updated: October 29, 2025*  
*Status: Core RAG pipeline complete, web interface in progress*
