"""Unit tests for RAGChain.query() error handling."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.rag_chain import RAGChain


@pytest.fixture
def chain():
    """A RAGChain with retrieval and the LLM client stubbed out."""
    rag = RAGChain.__new__(RAGChain)
    rag.llm_config = {"model": "test-model"}
    rag.retrieve = MagicMock(
        return_value=[Document(page_content="context", metadata={"source": "a.txt"})]
    )
    rag.client = MagicMock()
    return rag


def test_llm_error_is_returned_as_answer_by_default(chain):
    chain.client.chat.completions.create.side_effect = RuntimeError("boom")

    result = chain.query("What is RAG?")

    assert result["answer"] == "Error: Could not generate answer. boom"
    assert result["sources"] == ["a.txt"]


def test_llm_error_propagates_when_raise_on_error_is_set(chain):
    chain.client.chat.completions.create.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        chain.query("What is RAG?", raise_on_error=True)


def test_successful_answer_is_unaffected_by_raise_on_error(chain):
    message = SimpleNamespace(content="An answer")
    chain.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=message)]
    )

    assert chain.query("What is RAG?")["answer"] == "An answer"
    assert chain.query("What is RAG?", raise_on_error=True)["answer"] == "An answer"
