"""
EchoNotes API Module
====================

FastAPI-based REST API for speech-to-notes conversion.

Usage:
    # Run the server
    cd echonotes
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    
    # Or use Python
    python -m api.main

API Documentation:
    - Swagger UI: http://localhost:8000/docs
    - ReDoc: http://localhost:8000/redoc

Endpoints:
    POST /api/upload          - Upload audio file
    POST /api/transcribe      - Transcribe audio (async)
    POST /api/transcribe/sync - Transcribe audio (sync)
    POST /api/analyze         - Analyze text with NLP
    POST /api/generate        - Generate document
    POST /api/process         - Full pipeline (async)
    POST /api/process/sync    - Full pipeline (sync)
    GET  /api/status/{job_id} - Check job status
    GET  /api/download/{job_id} - Download document
    GET  /api/jobs            - List all jobs
    GET  /health              - Health check
"""

from .main import app, settings

__all__ = ['app', 'settings']
__version__ = '1.0.0'
