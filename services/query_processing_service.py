from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.chunks import ChunkInfo


class QueryProcessingService:
    """Encapsulates chunk search queries and response shaping."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def search_chunks(self, input_text_embedding: list[float], limit: int = 5) -> list[dict]:
        stmt = (
            select(
                ChunkInfo.chunk_id,
                ChunkInfo.file_name,
                ChunkInfo.page_no,
                ChunkInfo.content_type,
                ChunkInfo.content
            )
            .order_by(ChunkInfo.embedding.cosine_distance(input_text_embedding))
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return [
            {
                "chunk_id": f'chunk_{row.chunk_id:02d}',
                "file_name": row.file_name,
                "page_no": row.page_no,
                "content_type": row.content_type,
                "content": row.content,
            }
            for row in result.all()
        ]
