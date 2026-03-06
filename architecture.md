# Architecture Overview

This FastAPI-based Retrieval-Augmented Generation (RAG) system has two main pipelines: a **data ingestion** path that chunks and embeds PDF content into PostgreSQL (with `pgvector`), and a **query** path that retrieves similar chunks and generates answers with OpenAI using the retrieved context only.

## Core Components

- **API Layer** (`main.py`, `api/router.py`, `api/routes/*`): FastAPI app with routes for file upload/processing and query answering.
- **Lifespan Setup** (`fastapi_lifespan.py`): Initializes the async SQLAlchemy engine and a `ProcessPoolExecutor` stored on `app.state` for CPU-bound PDF parsing.
- **Database** (`models/chunks.py`, `models/file_info.py`): PostgreSQL tables `chunk_info` (stores chunk text/table data + `VECTOR(1024)` embedding) and `file_info` (ingestion status tracking).
- **Embedding Services** (`services/genai_embedding.py`): Gemini embeddings for both ingestion chunks and user queries.
- **Data Ingestion** (`services/data_ingestion_service.py`, `services/process_worker.py`, `api/routes/data_ingestion/*.py`):
  - Upload endpoint saves PDFs, records `file_info`.
  - Process endpoint dispatches `process_file` into the process pool to extract text/tables with `pdfplumber` + `pandas`, batch-embed with Gemini, and bulk insert chunks.
- **Query & RAG** (`services/query_processing_service.py`, `services/rag_answer_service.py`, `api/routes/query_processing/search_chunks.py`):
  - Query endpoint embeds the user query, retrieves nearest chunks via cosine distance in `pgvector`.
  - `RAGAnswerService` formats context and calls OpenAI chat (`gpt-4o-mini`), instructing the model to answer strictly from provided context.

## End-to-End Flows

- **Ingestion Flow**
  1. `POST /data-ingest/upload-file` saves the PDF and writes `file_info`.
  2. `GET /data-ingest/process-files` marks files `IN_PROGRESS`, dispatches `process_file` workers.
  3. Workers extract text/tables → Gemini `embed_contents` → bulk insert into `chunk_info` (with embeddings) → update `file_info` status.
- **Query Flow**
  1. `GET /query-processing/answer-query?user_query=...` embeds the query with Gemini.
  2. `QueryProcessingService` orders chunks by `embedding.cosine_distance` and returns top matches.
  3. `RAGAnswerService` builds a context block and calls OpenAI chat; response returned with selected chunks.
