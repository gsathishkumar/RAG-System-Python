from app_settings import settings
from openai import AsyncOpenAI


class RAGAnswerService:
    """Formats context and delegates completion to OpenAI."""

    def __init__(self, client: AsyncOpenAI | None = None, model: str = "gpt-4o-mini") -> None:
        self._client = client or AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = model

    async def answer(self, query: str, chunks: list[dict]) -> str:
        context_block = "\n\n".join(
            f"[{chunk['file_name']}#p{chunk['page_no']}] {chunk['content']}" for chunk in chunks
        ) or "No relevant context found."

        completion = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a RAG assistant. Answer the user's question strictly using the provided context. "
                        "If the context is insufficient, say you don't have enough information and avoid guessing."
                    )
                },
                {
                    "role": "user",
                    "content": f"Query:\n{query}\n\nContext:\n{context_block}"
                },
            ],
        )

        return completion.choices[0].message.content if completion.choices else ""
