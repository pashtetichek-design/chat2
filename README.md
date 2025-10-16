# Pet Hotel Chat Backend (FastAPI)

Backend для чат-бота отеля для животных. Совместим с фронтенд-виджетом.

## Локальный запуск

```bash
python -m venv venv
# Windows PowerShell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Создайте .env по образцу .env.example
# Запустите
uvicorn main:APP --reload --host 0.0.0.0 --port 8000
# Health check:
# http://localhost:8000/health
