"""KronaAI — Telegram assistant backed by a local Ollama model."""
from __future__ import annotations

import asyncio
import base64
import html
import json
import logging
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

import httpx
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("kronai")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b").strip()
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1").strip()
DB_PATH = Path(os.getenv("DATABASE_PATH", "/app/data/kronai.sqlite3"))
HISTORY_LIMIT = max(2, int(os.getenv("HISTORY_LIMIT", "12")))
TELEGRAM_TEXT_LIMIT = 3900

SYSTEM_PROMPT = (
    "Ты KronaAI — полезный русскоязычный AI-ассистент в Telegram. "
    "Отвечай точно и дружелюбно на языке пользователя. Не утверждай, что "
    "выполнил действие во внешней системе, если этого не делал. "
    "Если запрос явно просит написать код, дай полный рабочий пример в fenced code block. "
    "Не выполняй и не запускай сгенерированный код."
)

TABLE_RE = re.compile(r"\b(таблиц\w*|xlsx|excel|электронн\w+\s+таблиц\w*|spreadsheet)\b", re.I)
IMAGE_RE = re.compile(
    r"(нарисуй|создай\s+(?:изображение|картинк)|сгенерируй\s+(?:изображение|картинк)|"
    r"сделай\s+(?:изображение|картинк)|generate\s+(?:an?\s+)?image|draw\s+)",
    re.I,
)
CODE_RE = re.compile(
    r"(напиши|создай|сгенерируй|сделай|покажи)\s+(?:мне\s+)?(?:код|скрипт|программ)|"
    r"(write|create|generate)\s+(?:me\s+)?(?:some\s+)?(?:code|a\s+script|a\s+program)",
    re.I,
)
CODE_EXTENSIONS = {
    "python": "py", "py": "py", "javascript": "js", "js": "js",
    "typescript": "ts", "ts": "ts", "html": "html", "css": "css",
    "sql": "sql", "json": "json", "bash": "sh", "sh": "sh",
    "go": "go", "java": "java", "csharp": "cs", "cs": "cs",
    "rust": "rs", "cpp": "cpp", "c++": "cpp", "php": "php",
    "ruby": "rb", "kotlin": "kt", "swift": "swift",
}


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class KronaBot:
    def __init__(self) -> None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._model_lock = asyncio.Lock()
        self._available_models: set[str] = set()
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(DB_PATH, timeout=20)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _init_db(self) -> None:
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                """CREATE TABLE IF NOT EXISTS users (
                    telegram_id INTEGER PRIMARY KEY,
                    username TEXT,
                    display_name TEXT NOT NULL,
                    active_chat_id INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            db.execute(
                """CREATE TABLE IF NOT EXISTS chats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    telegram_id INTEGER NOT NULL,
                    title TEXT NOT NULL DEFAULT 'Новый диалог',
                    archived INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            db.execute(
                """CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )"""
            )
            db.execute("CREATE INDEX IF NOT EXISTS idx_chats_owner ON chats(telegram_id, archived, updated_at)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id)")

    def register_user(self, tg_user: Any) -> tuple[sqlite3.Row, int]:
        stamp = now_iso()
        name = " ".join(part for part in [tg_user.first_name, tg_user.last_name] if part).strip()
        name = name or tg_user.username or "Пользователь Telegram"
        with self._connect() as db:
            user = db.execute("SELECT * FROM users WHERE telegram_id = ?", (tg_user.id,)).fetchone()
            if user is None:
                db.execute(
                    """INSERT INTO users (telegram_id, username, display_name, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (tg_user.id, tg_user.username, name, stamp, stamp),
                )
                cursor = db.execute(
                    """INSERT INTO chats (telegram_id, title, created_at, updated_at)
                       VALUES (?, 'Первый диалог', ?, ?)""",
                    (tg_user.id, stamp, stamp),
                )
                chat_id = cursor.lastrowid
                db.execute(
                    "UPDATE users SET active_chat_id = ? WHERE telegram_id = ?",
                    (chat_id, tg_user.id),
                )
            else:
                db.execute(
                    "UPDATE users SET username = ?, display_name = ?, updated_at = ? WHERE telegram_id = ?",
                    (tg_user.username, name, stamp, tg_user.id),
                )
            user = db.execute("SELECT * FROM users WHERE telegram_id = ?", (tg_user.id,)).fetchone()
            if user["active_chat_id"] is None:
                chat_id = self.create_chat(tg_user.id, "Новый диалог")
            else:
                chat_id = int(user["active_chat_id"])
                exists = db.execute(
                    "SELECT id FROM chats WHERE id = ? AND telegram_id = ? AND archived = 0",
                    (chat_id, tg_user.id),
                ).fetchone()
                if not exists:
                    chat_id = self.create_chat(tg_user.id, "Новый диалог")
            user = db.execute("SELECT * FROM users WHERE telegram_id = ?", (tg_user.id,)).fetchone()
            return user, chat_id

    def create_chat(self, telegram_id: int, title: str = "Новый диалог") -> int:
        stamp = now_iso()
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO chats (telegram_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (telegram_id, title[:60], stamp, stamp),
            )
            chat_id = int(cursor.lastrowid)
            db.execute(
                "UPDATE users SET active_chat_id = ?, updated_at = ? WHERE telegram_id = ?",
                (chat_id, stamp, telegram_id),
            )
            return chat_id

    def set_active_chat(self, telegram_id: int, chat_id: int) -> bool:
        with self._connect() as db:
            valid = db.execute(
                "SELECT id FROM chats WHERE id = ? AND telegram_id = ?",
                (chat_id, telegram_id),
            ).fetchone()
            if not valid:
                return False
            db.execute(
                "UPDATE chats SET archived = 0, updated_at = ? WHERE id = ?",
                (now_iso(), chat_id),
            )
            db.execute(
                "UPDATE users SET active_chat_id = ?, updated_at = ? WHERE telegram_id = ?",
                (chat_id, now_iso(), telegram_id),
            )
            return True

    def archive_chat(self, telegram_id: int, chat_id: int) -> bool:
        with self._connect() as db:
            chat = db.execute(
                "SELECT id FROM chats WHERE id = ? AND telegram_id = ?",
                (chat_id, telegram_id),
            ).fetchone()
            if not chat:
                return False
            db.execute("UPDATE chats SET archived = 1, updated_at = ? WHERE id = ?", (now_iso(), chat_id))
            user = db.execute("SELECT active_chat_id FROM users WHERE telegram_id = ?", (telegram_id,)).fetchone()
            if user and user["active_chat_id"] == chat_id:
                cursor = db.execute(
                    """INSERT INTO chats (telegram_id, title, created_at, updated_at)
                       VALUES (?, 'Новый диалог', ?, ?)""",
                    (telegram_id, now_iso(), now_iso()),
                )
                db.execute(
                    "UPDATE users SET active_chat_id = ? WHERE telegram_id = ?",
                    (cursor.lastrowid, telegram_id),
                )
            return True

    def list_chats(self, telegram_id: int, archived: bool, limit: int = 8) -> list[sqlite3.Row]:
        with self._connect() as db:
            return db.execute(
                """SELECT c.id, c.title, c.updated_at,
                          (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id) AS message_count
                   FROM chats c WHERE c.telegram_id = ? AND c.archived = ?
                   ORDER BY c.updated_at DESC LIMIT ?""",
                (telegram_id, int(archived), limit),
            ).fetchall()

    def append_message(self, chat_id: int, role: str, content: str) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (chat_id, role, content, now_iso()),
            )
            db.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now_iso(), chat_id))
            if role == "user":
                first = db.execute(
                    "SELECT content FROM messages WHERE chat_id = ? AND role = 'user' ORDER BY id LIMIT 1",
                    (chat_id,),
                ).fetchone()
                if first:
                    title = re.sub(r"\s+", " ", first["content"]).strip()[:42] or "Новый диалог"
                    db.execute("UPDATE chats SET title = ? WHERE id = ? AND title IN ('Первый диалог', 'Новый диалог')", (title, chat_id))

    def get_history(self, chat_id: int) -> list[dict[str, str]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
                (chat_id, HISTORY_LIMIT),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]

    def delete_account(self, telegram_id: int) -> None:
        with self._connect() as db:
            chats = db.execute("SELECT id FROM chats WHERE telegram_id = ?", (telegram_id,)).fetchall()
            for chat in chats:
                db.execute("DELETE FROM messages WHERE chat_id = ?", (chat["id"],))
            db.execute("DELETE FROM chats WHERE telegram_id = ?", (telegram_id,))
            db.execute("DELETE FROM users WHERE telegram_id = ?", (telegram_id,))

    async def ensure_model(self, model: str, progress_message: Any | None = None) -> None:
        if model in self._available_models:
            return
        async with self._model_lock:
            if model in self._available_models:
                return
            try:
                async with httpx.AsyncClient(timeout=20) as client:
                    response = await client.get(f"{OLLAMA_URL}/api/tags")
                    response.raise_for_status()
                    installed = {
                        item.get("name", "")
                        for item in response.json().get("models", [])
                    }
                    if model not in installed:
                        if progress_message:
                            await progress_message.edit_text(
                                f"Скачиваю лёгкую модель <b>{html.escape(model)}</b> в Ollama. "
                                "Первый запуск может занять несколько минут.",
                                parse_mode="HTML",
                            )
                        async with client.stream(
                            "POST",
                            f"{OLLAMA_URL}/api/pull",
                            json={"name": model, "stream": True},
                            timeout=None,
                        ) as stream:
                            stream.raise_for_status()
                            last_percent = -1
                            async for line in stream.aiter_lines():
                                if not line:
                                    continue
                                event = json.loads(line)
                                total = event.get("total") or 0
                                completed = event.get("completed") or 0
                                percent = int(completed * 100 / total) if total else None
                                if progress_message and percent is not None and percent >= last_percent + 20:
                                    last_percent = percent
                                    await progress_message.edit_text(
                                        f"Устанавливаю модель <b>{html.escape(model)}</b>… {percent}%",
                                        parse_mode="HTML",
                                    )
                    self._available_models.add(model)
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                log.warning("Model setup failed for %s: %s", model, exc)
                raise RuntimeError(
                    "Не удалось подключиться к Ollama. Проверьте, что Docker Compose запущен, "
                    "а контейнер ollama доступен."
                ) from exc

    async def ask_ollama(self, messages: list[dict[str, Any]], model: str = MODEL) -> str:
        payload = {"model": model, "messages": messages, "stream": False}
        async with httpx.AsyncClient(timeout=httpx.Timeout(240, connect=15)) as client:
            response = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            response.raise_for_status()
            return response.json()["message"]["content"].strip()


BOT = KronaBot()


def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("💬 Новый диалог", callback_data="chat:new"),
             InlineKeyboardButton("🗂 Архив", callback_data="chat:archive")],
            [InlineKeyboardButton("🧠 Модель", callback_data="model:info"),
             InlineKeyboardButton("❓ Помощь", callback_data="menu:help")],
            [InlineKeyboardButton("🗑 Удалить мои данные", callback_data="account:confirm_delete")],
        ]
    )


def back_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("↩️ Главное меню", callback_data="menu:home")]])


async def send_text(update: Update, text: str, reply_markup: InlineKeyboardMarkup | None = None) -> None:
    """Send plain text in safe Telegram-sized chunks."""
    message = update.effective_message
    if not message:
        return
    text = text or " "
    chunks = [text[i:i + TELEGRAM_TEXT_LIMIT] for i in range(0, len(text), TELEGRAM_TEXT_LIMIT)]
    for index, chunk in enumerate(chunks):
        await message.reply_text(chunk, reply_markup=reply_markup if index == len(chunks) - 1 else None)


def parse_table(raw: str) -> tuple[list[str], list[list[Any]]] | None:
    candidate = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", candidate, flags=re.S | re.I)
    if fenced:
        candidate = fenced.group(1).strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(candidate[start:end + 1])
            columns = data.get("columns") or data.get("headers")
            rows = data.get("rows") or data.get("data")
            if isinstance(columns, list) and isinstance(rows, list):
                headings = [str(value)[:80] for value in columns]
                normalized: list[list[Any]] = []
                for row in rows[:5000]:
                    if isinstance(row, dict):
                        normalized.append([row.get(column, "") for column in columns])
                    elif isinstance(row, (list, tuple)):
                        normalized.append(list(row[:len(headings)]) + [""] * max(0, len(headings) - len(row)))
                if headings:
                    return headings, normalized
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
    # Accept a standard Markdown table as a fallback if the model ignored the JSON instruction.
    lines = [line.strip() for line in raw.splitlines() if "|" in line]
    if len(lines) < 2:
        return None

    def cells(line: str) -> list[str]:
        return [value.strip() for value in line.strip().strip("|").split("|")]

    headings = cells(lines[0])
    if not headings or not all(re.fullmatch(r":?-{3,}:?", item) for item in cells(lines[1])):
        return None
    rows = [cells(line) for line in lines[2:502]]
    rows = [row[:len(headings)] + [""] * max(0, len(headings) - len(row)) for row in rows]
    return headings, rows


def make_xlsx(columns: list[str], rows: list[list[Any]]) -> BytesIO:
    book = Workbook()
    sheet = book.active
    sheet.title = "Таблица"
    sheet.append(columns)
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="5B5BD6")
        cell.alignment = Alignment(vertical="top", wrap_text=True)
    for row in rows:
        sheet.append([str(value)[:32000] if value is not None else "" for value in row])
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for cells in sheet.columns:
        letter = cells[0].column_letter
        width = min(36, max(12, max((len(str(c.value or "")) for c in cells[:100]), default=12) + 2))
        sheet.column_dimensions[letter].width = width
    output = BytesIO()
    book.save(output)
    output.seek(0)
    return output


def extract_code(raw: str) -> tuple[str, str] | None:
    match = re.search(r"```([a-zA-Z0-9+#.-]*)\s*\n(.*?)```", raw, flags=re.S)
    if not match:
        return None
    language = match.group(1).strip().lower()
    extension = CODE_EXTENSIONS.get(language, "txt")
    code = match.group(2).strip()
    if not code:
        return None
    return extension, code


def looks_like_image_request(text: str) -> bool:
    return bool(IMAGE_RE.search(text))


def is_table_request(text: str) -> bool:
    return bool(TABLE_RE.search(text))


def is_code_request(text: str) -> bool:
    return bool(CODE_RE.search(text))


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user or not update.effective_message:
        return
    user, _ = BOT.register_user(update.effective_user)
    greeting = await update.effective_message.reply_text(
        f"Привет, {user['display_name']}! Я KronaAI — ваш AI-ассистент.\n\n"
        "Регистрация создана по Telegram ID. История диалогов хранится в базе этого Docker-проекта. "
        "Модель работает локально через Ollama; для генерации изображений нужен отдельный API-ключ.",
        reply_markup=main_menu(),
    )
    try:
        await BOT.ensure_model(MODEL, greeting)
        await greeting.edit_text(
            f"Привет, {user['display_name']}! Модель <b>{html.escape(MODEL)}</b> готова. "
            "Напишите сообщение или выберите действие в меню.",
            parse_mode="HTML",
            reply_markup=main_menu(),
        )
    except RuntimeError as exc:
        await greeting.edit_text(
            f"Аккаунт создан, но модель пока недоступна.\n{exc}\n\n"
            "Запустите проект командой <code>docker compose up -d</code> и нажмите /start ещё раз.",
            parse_mode="HTML",
            reply_markup=main_menu(),
        )


async def handle_new_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user or not update.effective_message:
        return
    BOT.register_user(update.effective_user)
    BOT.create_chat(update.effective_user.id)
    await send_text(
        update,
        "Новый диалог создан. Напишите сообщение — я отвечу с учётом его истории.",
        reply_markup=main_menu(),
    )


async def show_archive(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user:
        return
    BOT.register_user(update.effective_user)
    chats = BOT.list_chats(update.effective_user.id, archived=True)
    if not chats:
        text = "Архив пока пуст. Диалог можно отправить в архив командой /archive."
        if update.callback_query:
            await update.callback_query.edit_message_text(text, reply_markup=back_menu())
        else:
            await send_text(update, text, reply_markup=back_menu())
        return
    buttons = []
    for chat in chats:
        label = f"📄 {chat['title'][:32]} · {chat['message_count']}"
        buttons.append([InlineKeyboardButton(label, callback_data=f"chat:open:{chat['id']}")])
    buttons.append([InlineKeyboardButton("↩️ Главное меню", callback_data="menu:home")])
    markup = InlineKeyboardMarkup(buttons)
    text = "🗂 Архив диалогов\nВыберите диалог, чтобы восстановить его."
    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=markup)
    else:
        await send_text(update, text, reply_markup=markup)


async def handle_archive_current(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user:
        return
    user, chat_id = BOT.register_user(update.effective_user)
    if BOT.archive_chat(update.effective_user.id, chat_id):
        text = "Текущий диалог отправлен в архив. Создан новый диалог."
    else:
        text = "Не удалось найти текущий диалог."
    await send_text(update, text, reply_markup=main_menu())


async def show_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Как пользоваться KronaAI:\n"
        "• Напишите вопрос — модель ответит и сохранит историю диалога.\n"
        "• Попросите составить таблицу — отправлю файл Excel (.xlsx).\n"
        "• Попросите написать код — приложу исходник отдельным файлом.\n"
        "• Попросите создать изображение — отправлю его, если настроен OPENAI_API_KEY.\n"
        "• Отправьте фото — для анализа настройте необязательную модель зрения.\n\n"
        "Команды: /start, /new, /archive, /model, /help, /delete_me."
    )
    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=back_menu())
    else:
        await send_text(update, text, reply_markup=main_menu())


async def show_model_info(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        f"Основная модель: {MODEL}\n"
        f"Провайдер: Ollama в Docker (локально)\n"
        f"Модель зрения: {VISION_MODEL or 'не настроена'}\n"
        f"Генерация изображений: {'настроена' if OPENAI_API_KEY else 'нужен OPENAI_API_KEY'}\n\n"
        "Модель загружается автоматически в Ollama при первом /start. "
        "Промпты отправляются локальной модели; ключ для картинок необязателен."
    )
    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=back_menu())
    else:
        await send_text(update, text, reply_markup=main_menu())


async def request_delete_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    markup = InlineKeyboardMarkup(
        [[InlineKeyboardButton("Удалить историю и профиль", callback_data="account:delete"),
          InlineKeyboardButton("Отмена", callback_data="menu:home")]]
    )
    text = "Удалить регистрацию и все ваши диалоги без возможности восстановления?"
    if update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=markup)
    else:
        await send_text(update, text, reply_markup=markup)


async def delete_account(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_user:
        return
    BOT.delete_account(update.effective_user.id)
    text = "Ваш профиль и сохранённые диалоги удалены. Нажмите /start, чтобы зарегистрироваться снова."
    if update.callback_query:
        await update.callback_query.edit_message_text(text)
    else:
        await send_text(update, text)


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return
    await query.answer()
    data = query.data or ""
    if data == "menu:home":
        await query.edit_message_text("Главное меню KronaAI", reply_markup=main_menu())
    elif data == "menu:help":
        await show_help(update, context)
    elif data == "model:info":
        await show_model_info(update, context)
    elif data == "chat:new":
        if update.effective_user:
            BOT.register_user(update.effective_user)
            BOT.create_chat(update.effective_user.id)
        await query.edit_message_text(
            "Новый диалог создан. Напишите сообщение — я отвечу с учётом его истории.",
            reply_markup=main_menu(),
        )
    elif data == "chat:archive":
        await show_archive(update, context)
    elif data == "account:confirm_delete":
        await request_delete_confirmation(update, context)
    elif data == "account:delete":
        await delete_account(update, context)
    elif data.startswith("chat:open:") and update.effective_user:
        try:
            chat_id = int(data.rsplit(":", 1)[1])
        except ValueError:
            await query.edit_message_text("Не удалось открыть этот диалог.", reply_markup=back_menu())
            return
        if BOT.set_active_chat(update.effective_user.id, chat_id):
            await query.edit_message_text("Диалог восстановлен. Продолжайте писать сообщения.", reply_markup=main_menu())
        else:
            await query.edit_message_text("Диалог не найден.", reply_markup=back_menu())


async def create_image(prompt: str) -> BytesIO:
    if not OPENAI_API_KEY:
        raise RuntimeError("Генерация изображений не настроена. Добавьте OPENAI_API_KEY в файл .env и перезапустите Docker.")
    async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=20)) as client:
        response = await client.post(
            "https://api.openai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={"model": IMAGE_MODEL, "prompt": prompt, "size": "1024x1024"},
        )
        if response.is_error:
            raise RuntimeError(f"Сервис изображений вернул ошибку {response.status_code}. Проверьте ключ и доступ к модели.")
        data = response.json()["data"][0]
        if data.get("b64_json"):
            image_bytes = base64.b64decode(data["b64_json"])
        elif data.get("url"):
            image_response = await client.get(data["url"])
            image_response.raise_for_status()
            image_bytes = image_response.content
        else:
            raise RuntimeError("Сервис изображений не вернул файл.")
    image = BytesIO(image_bytes)
    image.name = "kronaai-image.png"
    image.seek(0)
    return image


async def reply_to_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> None:
    user = update.effective_user
    message = update.effective_message
    if not user or not message:
        return
    _, chat_id = BOT.register_user(user)
    if looks_like_image_request(text):
        status = await message.reply_text("🎨 Создаю изображение…")
        try:
            image = await create_image(text)
            await message.reply_photo(photo=image, caption="Готово — изображение создано по вашему описанию.")
            await status.delete()
        except (RuntimeError, httpx.HTTPError, ValueError, KeyError) as exc:
            await status.edit_text(str(exc))
        return

    BOT.append_message(chat_id, "user", text)
    status = await message.reply_text("Думаю…")
    try:
        await context.bot.send_chat_action(chat_id=user.id, action=ChatAction.TYPING)
        history = BOT.get_history(chat_id)
        model = MODEL
        is_table = is_table_request(text)
        if is_table:
            history = history[:-1] + [{
                "role": "user",
                "content": (
                    f"{text}\n\nВерни результат только как JSON без markdown и пояснений: "
                    '{"columns":["столбец 1"],"rows":[["значение"]]}. '
                    "Сделай таблицу заполненной и полезной; максимум 100 строк."
                ),
            }]
        answer = await BOT.ask_ollama([{"role": "system", "content": SYSTEM_PROMPT}, *history], model)
        BOT.append_message(chat_id, "assistant", answer)
        await status.delete()
        parsed = parse_table(answer) if is_table else None
        if parsed:
            columns, rows = parsed
            preview = " | ".join(columns) + "\n" + " | ".join(["—"] * len(columns))
            preview_rows = [" | ".join(str(value) for value in row) for row in rows[:12]]
            preview_text = f"📊 Таблица готова: {len(rows)} строк.\n\n" + preview + "\n" + "\n".join(preview_rows)
            await message.reply_text(preview_text[:TELEGRAM_TEXT_LIMIT])
            xlsx = make_xlsx(columns, rows)
            await message.reply_document(
                document=xlsx,
                filename="kronaai_table.xlsx",
                caption="Таблица Excel — можно открыть и редактировать.",
            )
        else:
            await send_text(update, answer)
            if is_code_request(text) or "```" in answer:
                code = extract_code(answer)
                if code:
                    extension, source = code
                    doc = BytesIO(source.encode("utf-8"))
                    doc.name = f"kronaai_code.{extension}"
                    await message.reply_document(
                        document=doc,
                        caption=f"Исходный код ({extension}). Файл создан как текст, код не запускался.",
                    )
    except (httpx.HTTPError, RuntimeError, KeyError, ValueError) as exc:
        log.exception("AI request failed")
        await status.edit_text(
            "Не получилось получить ответ от модели. Убедитесь, что Ollama работает, "
            "и повторите запрос через минуту."
        )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message or not update.effective_message.text:
        return
    await reply_to_prompt(update, context, update.effective_message.text.strip())


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    user = update.effective_user
    if not message or not user or not message.photo:
        return
    BOT.register_user(user)
    if not VISION_MODEL:
        await message.reply_text(
            "Чтобы анализировать фото, задайте OLLAMA_VISION_MODEL (например, llava:7b) в .env. "
            "Это отдельная, более тяжёлая модель; обычный чат уже работает."
        )
        return
    status = await message.reply_text(f"Подготавливаю модель зрения {VISION_MODEL}…")
    try:
        await BOT.ensure_model(VISION_MODEL, status)
        photo_file = await message.photo[-1].get_file()
        image_bytes = await photo_file.download_as_bytearray()
        caption = message.caption or "Опиши это изображение и ответь на вопросы по нему."
        _, chat_id = BOT.register_user(user)
        BOT.append_message(chat_id, "user", f"[Фото] {caption}")
        history = BOT.get_history(chat_id)
        image_message = {
            "role": "user",
            "content": caption,
            "images": [base64.b64encode(bytes(image_bytes)).decode("ascii")],
        }
        history = history[:-1] + [image_message]
        answer = await BOT.ask_ollama(
            [{"role": "system", "content": SYSTEM_PROMPT}, *history],
            VISION_MODEL,
        )
        BOT.append_message(chat_id, "assistant", answer)
        await status.delete()
        await send_text(update, answer)
    except (RuntimeError, httpx.HTTPError, ValueError, KeyError) as exc:
        log.exception("Image analysis failed")
        await status.edit_text(f"Не удалось обработать фото: {exc}")


async def handle_model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await show_model_info(update, context)


async def handle_delete_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await request_delete_confirmation(update, context)


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.error("Unhandled Telegram handler error", exc_info=context.error)


def build_application() -> Application:
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан. Создайте .env по образцу .env.example.")
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(CommandHandler("new", handle_new_chat))
    app.add_handler(CommandHandler("archive", handle_archive_current))
    app.add_handler(CommandHandler("model", handle_model_command))
    app.add_handler(CommandHandler("help", show_help))
    app.add_handler(CommandHandler("delete_me", handle_delete_command))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_error_handler(on_error)
    return app


if __name__ == "__main__":
    application = build_application()
    log.info("KronaAI is starting; default model: %s", MODEL)
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=False)
