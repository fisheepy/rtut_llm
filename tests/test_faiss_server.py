import importlib


def load_server_module():
    return importlib.import_module("faiss_server")


def test_status_endpoint_returns_shape(monkeypatch):
    server = load_server_module()
    client = server.app.test_client()

    response = client.get("/status")

    assert response.status_code == 200
    payload = response.get_json()
    assert {"num_vectors", "num_policies", "index_size", "sample_metadata"}.issubset(payload.keys())


def test_chat_requires_question():
    server = load_server_module()
    client = server.app.test_client()

    response = client.post("/chat", json={})

    assert response.status_code == 400
    assert response.get_json()["error"] == "Missing question"


def test_search_requires_query():
    server = load_server_module()
    client = server.app.test_client()

    response = client.post("/search", json={})

    assert response.status_code == 400
    assert response.get_json()["error"] == "Missing search query"


def test_chat_happy_path_with_basic_question(monkeypatch):
    server = load_server_module()
    client = server.app.test_client()

    server.metadata_store = [{"name": "policy.docx", "text": "Employees may request leave in advance."}]

    class DummyIndex:
        def search(self, _query_embedding, k=3):
            return [[0.01]], [[0]]

    class DummyMessage:
        content = "You can request leave by submitting a leave form."

    class DummyChoice:
        message = DummyMessage()

    class DummyResponse:
        choices = [DummyChoice()]

    monkeypatch.setattr(server, "index", DummyIndex())
    monkeypatch.setattr(server, "get_embedding", lambda _text: [0.0] * 1536)
    monkeypatch.setattr(server.openai.chat.completions, "create", lambda **_kwargs: DummyResponse())

    response = client.post(
        "/chat",
        json={"question": "How do I request leave?", "style": "friendly"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert "answer" in payload
    assert "Roy Bot" in payload["answer"]
    assert payload["referenced_documents"] == ["policy.docx"]
