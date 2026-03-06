from google import genai
from google.genai import types

from app_settings import settings


class GeminiEmbeddingService:
    """Wrapper around GenAI SDK that returns vector embeddings."""

    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)

    def embed_content(
        self,
        content: str,
        *,
        model: str = 'gemini-embedding-001',
        output_dimensionality: int = 1024
    ) -> list[float]:
        embeddings = self.embed_contents(
            [content],
            model=model,
            output_dimensionality=output_dimensionality
        )
        if not embeddings:
            raise ValueError('Gemini returned no embeddings for the provided content')
        return embeddings[0]

    def embed_contents(
        self,
        contents: list[str],
        *,
        model: str = 'gemini-embedding-001',
        output_dimensionality: int = 1024
    ) -> list[list[float]]:
        if not contents:
            return []
        response = self._client.models.embed_content(
            model=model,
            contents=contents,
            config=types.EmbedContentConfig(output_dimensionality=output_dimensionality)
        )
        embeddings = [embedding.values for embedding in response.embeddings]
        if len(embeddings) != len(contents):
            raise ValueError("Gemini did not return embeddings for all provided contents")
        return embeddings
