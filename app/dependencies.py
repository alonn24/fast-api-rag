import anthropic
import voyageai

from app.config import get_settings
from app.services.embeddings import EmbeddingService


def get_embedding_service_dep() -> EmbeddingService:
    settings = get_settings()
    client = voyageai.Client(api_key=settings.voyage_api_key)
    return EmbeddingService(client, model=settings.embedding_model)


def get_anthropic_client_dep() -> anthropic.Anthropic:
    settings = get_settings()
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)
