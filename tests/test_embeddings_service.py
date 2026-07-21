from unittest.mock import Mock

from app.services.embeddings import EmbeddingService


def test_embed_documents_calls_client_with_document_input_type():
    fake_client = Mock()
    fake_client.embed.return_value = Mock(embeddings=[[0.1, 0.2], [0.3, 0.4]])
    service = EmbeddingService(fake_client, model="voyage-3")

    result = service.embed_documents(["a", "b"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    fake_client.embed.assert_called_once_with(["a", "b"], model="voyage-3", input_type="document")


def test_embed_documents_empty_list_returns_empty_without_calling_client():
    fake_client = Mock()
    service = EmbeddingService(fake_client)
    assert service.embed_documents([]) == []
    fake_client.embed.assert_not_called()


def test_embed_query_calls_client_with_query_input_type():
    fake_client = Mock()
    fake_client.embed.return_value = Mock(embeddings=[[0.5, 0.6]])
    service = EmbeddingService(fake_client, model="voyage-3")

    result = service.embed_query("hello")

    assert result == [0.5, 0.6]
    fake_client.embed.assert_called_once_with(["hello"], model="voyage-3", input_type="query")
