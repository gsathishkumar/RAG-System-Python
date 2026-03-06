# Retrieval and Generation System

(Using postgres db with pgvector extension)

#### Concept:

Build a basic Retrieval system that will perform splitting on given pdf documents along with tables and convert them in embedding and store embedding in database. Perform a search operation for the given query from the OpenAI Model.

## Core Features:

There are two kinds of pipelines in this system

### Data Ingestion pipeline

Load the pdf document and apply chunking over it.  
The document has both text and tables.  
Make sure each table should be stored as one chunk.

Convert each chunk of both text and table into embedding and store in postgres db.  
Use pgvector extension to enable vector search.

You can use some embedding model like openai or Google api

### Search Engine

Build a search engine where user can have a query and system should bring the relavant answer.

## Expected output of Search Engine

For give query, it should bring top 10 similar chunks
If query has row info then it should bring that table chunks

## End-to-end flow

Upload File -> Text/Table(Chunks) → Embedding model → Vector → Store in PostgreSQL (pgvector)

Search Query -> Text → Embedding model → Vector → PostgreSQL (pgvector) -> Similarity search -> Generation model -> final answer

## Libraries Used

- FastAPI to expose async API's
- Pydantic for data validation
- Pydantic_Settings to load app settings based on environment (dev/qa/prod)
- Sqlalchemy ORM to persist the model in database
- Asynchronous PostgreSQL client library(asyncpg) for Python that is specifically designed to work with Python's asyncio framework.
- Uvicorn as an ASGI web server implementation for Python
- Concurrent Futures to Spawn Process for PDF Processing(CPU-Bound Task)
- Google GenAI client SDK to generate embedding
- Pdfplumber to extract text and tables from PDF files
- Panda to convert the tablular data to DataFrames
- Tabulate to convert the DataFrame to tabular text data
- OpenAI LLM to generate answer from the context
