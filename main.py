import os
import uuid
import requests
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Загрузка .env
load_dotenv()

# Конфигурация
ROUTE_LLM_API_KEY = os.getenv("ROUTE_LLM_API_KEY")
if not ROUTE_LLM_API_KEY:
    raise RuntimeError("Переменная окружения ROUTE_LLM_API_KEY не задана")

# ВАЖНО: проверьте актуальный endpoint на странице RouteLLM APIs
# Ниже два популярных варианта интерфейса. Оставьте тот, что работает у вас.
# Вариант A (унифицированный чат‑эндпоинт Abacus Route LLM):
ROUTE_LLM_URL = "https://api.abacus.ai/route-llm/v1/chat"
# Вариант B (если используется совместимый формат OpenAI Chat Completions):
# ROUTE_LLM_URL = "https://api.abacus.ai/route-llm/v1/openai/chat/completions"

# Базовые подсказки (адаптируйте под ваш отель)
SYSTEM_PROMPT = """Вы — виртуальный консультант отеля для животных.
Отвечайте кратко, дружелюбно и по делу. 
Задачи: FAQ, предварительная оценка стоимости, сбор данных для заявки.
Всегда уточняйте: вид животного, вес/размер, даты, тип номера, доп. услуги, контакты.
Не подтверждайте бронирование и цену окончательно — это делает администратор.
Если не уверены — предложите соединить с администратором.
"""

# Простейшая "база знаний" (позже замените на RAG)
KNOWLEDGE_BASE = """Прайс (пример):
- Собаки до 10 кг: 800₽/сутки
- Собаки 10–25 кг: 1200₽/сутки
- Собаки 25+ кг: 1500₽/сутки
- Кошки: 600₽/сутки
Доп. услуги: выгул (2 раза/день) +300₽, индивидуальное кормление +200₽, видеоотчеты +500₽/нед.
Требования: ветпаспорт с прививками (бешенство обязательно), обработка от паразитов до 3 мес.
Важно: цены ориентировочные, подтверждаются администратором перед оплатой.
"""

# Инициализация FastAPI
APP = FastAPI(title="Pet Hotel Chat Backend")

# CORS (в продакшене ограничьте allow_origins конкретным доменом сайта)
APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# In-memory хранилище сессий (в продакшене используйте Redis/БД)
SESSIONS: dict[str, List[dict]] = {}

# Модели запрос/ответ
class ChatRequest(BaseModel):
    message: str
    sessionId: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    sessionId: str

def call_route_llm(messages: List[dict]) -> str:
    """
    Вызов RouteLLM. 
    Если используете совместимый с OpenAI endpoint, раскомментируйте формат ниже (variant='openai').
    """
    headers = {
        "Authorization": f"Bearer {ROUTE_LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    # Вариант A: универсальный роутер Abacus RouteLLM (чаще всего messages и model)
    payload = {
        "model": "gpt-4o-mini",   # выберите модель из списка в RouteLLM
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 500,
        "stream": False
    }

    # Вариант B (совместимый с OpenAI format):
    # payload = {
    #     "model": "gpt-4o-mini",
    #     "messages": messages,
    #     "temperature": 0.3,
    #     "max_tokens": 500
    # }

    try:
        r = requests.post(ROUTE_LLM_URL, json=payload, headers=headers, timeout=45)
        r.raise_for_status()
        data = r.json()

        # Попробуем несколько форматов ответа, в зависимости от выбранного роутера/модели:
        # 1) OpenAI-совместимый:
        choice = None
        if isinstance(data, dict) and "choices" in data and data["choices"]:
            choice = data["choices"][0]
            # OpenAI-like
            msg = choice.get("message", {})
            if isinstance(msg, dict) and "content" in msg:
                return msg["content"]

        # 2) Упрощенный формат {"content": "..."}
        if "content" in data and isinstance(data["content"], str):
            return data["content"]

        # 3) Fallback — попытаемся достать текст из разных полей
        return str(data)
    except Exception as e:
        print("RouteLLM error:", e)
        return "Извините, сервис временно недоступен. Попробуйте позднее или свяжитесь с администратором."

@APP.get("/health")
def health():
    return {"status": "ok"}

@APP.post("/api/pet-hotel-chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Создаём/получаем sessionId
    session_id = req.sessionId or str(uuid.uuid4())

    # Инициализация истории
    if session_id not in SESSIONS:
        SESSIONS[session_id] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"Справочная информация:\n{KNOWLEDGE_BASE}"}
        ]

    history = SESSIONS[session_id]

    # Добавляем пользовательское сообщение
    history.append({"role": "user", "content": req.message})

    # Вызов LLM
    reply = call_route_llm(history)

    # Сохраняем ответ
    history.append({"role": "assistant", "content": reply})

    # Ограничиваем рост истории
    SESSIONS[session_id] = history[-24:]

    return ChatResponse(reply=reply, sessionId=session_id)

# Локальный запуск: uvicorn main:APP --reload --host 0.0.0.0 --port 8000