from fastapi import APIRouter

from .routes.data_ingestion import process_files, upload_file
from .routes.query_processing import search_chunks

api_router = APIRouter()
api_router.include_router(upload_file.router)
api_router.include_router(process_files.router)
api_router.include_router(search_chunks.router)