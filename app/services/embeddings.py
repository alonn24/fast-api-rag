import voyageai


class EmbeddingService:
    """Wraps a voyageai.Client behind a small interface so tests can inject a fake client
    instead of hitting the real Voyage API."""

    def __init__(self, client: voyageai.Client, model: str = "voyage-3"):
        self._client = client
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        result = self._client.embed(texts, model=self._model, input_type="document")
        return result.embeddings

    def embed_query(self, text: str) -> list[float]:
        result = self._client.embed([text], model=self._model, input_type="query")
        return result.embeddings[0]
