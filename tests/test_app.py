"""Unit tests for FastAPI application."""
import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


def test_chat_page_loads(client):
    """Test that chat page loads successfully."""
    response = client.get("/chat")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_ask_endpoint_with_message(client):
    """Test that /ask endpoint responds (may need RAG initialized)."""
    response = client.post("/ask", json={"question": "Hello"})
    # May return 503 if RAG not initialized, 200 if working
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)


def test_ask_endpoint_requires_question(client):
    """Test that /ask POST requires a question."""
    response = client.post("/ask", json={})
    # Should return 422 (validation error)
    assert response.status_code == 422


def test_static_css_exists(client):
    """Test that CSS file is accessible."""
    response = client.get("/static/css/style.css")
    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]


def test_static_js_exists(client):
    """Test that JavaScript file is accessible."""
    response = client.get("/static/js/chat.js")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
