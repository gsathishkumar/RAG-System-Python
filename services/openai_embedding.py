import openai

from app_settings import settings


class OpenAIEmbeddingService:
    """Wrapper around OpenAI Embeddings API that returns vectors for input text."""

    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise ValueError('OpenAI API key must be configured before using the embedding service')
        self._client = openai
        self._client.api_key = settings.openai_api_key

    def embed_content(
        self,
        content: str,
        *,
        model: str = 'text-embedding-ada-002'
    ) -> list[float]:
        embeddings = self.embed_contents([content], model=model)
        if not embeddings:
            raise ValueError('OpenAI returned no embeddings for the provided content')
        return embeddings[0]

    def embed_contents(
        self,
        contents: list[str],
        *,
        model: str = 'text-embedding-ada-002'
    ) -> list[list[float]]:
        if not contents:
            return []
        response = self._client.Embedding.create(
            model=model,
            input=contents
        )
        data = response.get('data', [])
        embeddings = [item['embedding'] for item in data]
        if len(embeddings) != len(contents):
            raise ValueError('OpenAI did not return embeddings for every requested input')
        return embeddings
