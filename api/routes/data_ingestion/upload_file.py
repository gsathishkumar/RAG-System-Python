import os
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, Form, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app_settings import settings
from db.dependencies import get_db_session
from schemas.file_upload import FileUploadSchema
from services.data_ingestion_service import DataInjestionService

router = APIRouter(prefix="/data-ingest", tags=["File Operations"])
UPLOAD_DIR = Path(settings.file.upload_path)
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload-file", response_model=None)
async def multi_part_form_data(request: Request,
                               form_data: FileUploadSchema = Form(..., media_type="multipart/form-data"),
                               session: AsyncSession = Depends(get_db_session)):  # Pydantic request body via Form
    service = DataInjestionService(session, request.app.state.executor)
    file_exist_in_db = await service.check_file_info_exists_in_db(form_data.input_file.filename)
    if file_exist_in_db:
        return {'message' : 'File Information Already exists in Database'}
    
    file_path = UPLOAD_DIR / form_data.input_file.filename
    try:
        async with aiofiles.open(file_path, "wb") as out_file:
            while content := await form_data.input_file.read(1024 * 1024):  # 1MB chunks
                await out_file.write(content)
                
        file_info = await service.add_file_info_in_db(form_data.input_file.filename, form_data.uploaded_by)
    except Exception as e:
        raise e
    finally:
        await form_data.input_file.close()
    return file_info

