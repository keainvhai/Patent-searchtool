@echo off
call .\.venv\Scripts\activate
uvicorn app.api:app --host 127.0.0.1 --port 8000
