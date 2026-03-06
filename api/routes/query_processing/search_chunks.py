from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from db.dependencies import get_db_session
from services.genai_embedding import GeminiEmbeddingService
from services.query_processing_service import QueryProcessingService
from services.rag_answer_service import RAGAnswerService

router = APIRouter(prefix="/query-processing", tags=["Search Operations"])
embedding_service = GeminiEmbeddingService()
rag_answer_service = RAGAnswerService()

@router.get("/search-chunks", response_model=None)
async def search_chunks(input_text: str =  Query(..., min_length=3),
                               session: AsyncSession = Depends(get_db_session)):  # Pydantic request body via Form
  print('Connecting gemini via Genai-Client SDK --> ')
  input_text_embedding = embedding_service.embed_content(input_text)
  print('Completed retrieving embeddings--> ')
  query_service = QueryProcessingService(session)
  retrieved_chunks = await query_service.search_chunks(input_text_embedding)
  return {'query': input_text, 'chunks': retrieved_chunks}


@router.get("/answer-query", response_model=None)
async def answer_query(
    user_query: str = Query(..., min_length=3),
    session: AsyncSession = Depends(get_db_session)
):
    """Generate an answer using retrieved chunks as RAG context."""
    query_service = QueryProcessingService(session)

    # 1) Embed the incoming query
    query_embedding = embedding_service.embed_content(user_query)

    # 2) Retrieve the most relevant chunks
    retrieved_chunks = await query_service.search_chunks(query_embedding)

    # 3) Generate answer through the RAG answer service
    answer = await rag_answer_service.answer(user_query, retrieved_chunks)

    return {"query": user_query, "answer": answer}
