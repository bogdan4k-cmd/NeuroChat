import os
import tempfile
import asyncio
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import (
    Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton,
    LabeledPrice, PreCheckoutQuery, ContentType
)
from aiogram.filters import Command, CommandStart
from groq import Groq
import pytesseract
from PIL import Image
import logging
import sys
from typing import Optional, Iterable, List
import sqlite3
from datetime import datetime
import datetime as pydt

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Bot / clients ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "").strip()
_admins_raw = os.getenv("ADMIN_IDS", "").strip()
PROVIDER_TOKEN = os.getenv("PROVIDER_TOKEN", "").strip()

if not TELEGRAM_TOKEN:
    logger.error("TELEGRAM_TOKEN не задан в .env")
    sys.exit(1)
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY не задан в .env")
    sys.exit(1)

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
router = Router()
groq_client = Groq(api_key=GROQ_API_KEY)

# admin ids
if _admins_raw:
    try:
        ADMINS = set(map(int, filter(None, (s.strip() for s in _admins_raw.split(",")))))
    except Exception:
        ADMINS = set()
else:
    ADMINS = set()

# Tesseract config (from .env or default Windows path)
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()
if not TESSERACT_CMD:
    default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default):
        TESSERACT_CMD = default
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# --- Groq helpers ---
def _extract_model_names(resp) -> Iterable[str]:
    names = []
    try:
        if hasattr(resp, "data"):
            data = resp.data
        else:
            data = resp
        if isinstance(data, (list, tuple)):
            for item in data:
                if isinstance(item, dict):
                    for key in ("name", "id", "model"):
                        if key in item:
                            names.append(item[key])
                            break
                else:
                    for attr in ("name", "id", "model"):
                        if hasattr(item, attr):
                            names.append(getattr(item, attr))
                            break
        elif isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, dict):
                    for k in ("name", "id", "model"):
                        if k in val:
                            names.append(val[k])
                            break
                else:
                    names.append(key)
    except Exception:
        pass
    return [str(n) for n in dict.fromkeys(filter(None, names))]

async def list_available_models(client: Groq) -> List[str]:
    names = []
    try:
        resp = client.models.list()
        names = _extract_model_names(resp)
    except Exception:
        pass
    if not names:
        try:
            resp = client.list_models()
            names = _extract_model_names(resp)
        except Exception:
            pass
    if not names:
        try:
            if hasattr(client, "get"):
                resp = client.get("/models")
                names = _extract_model_names(resp)
        except Exception:
            pass
    return names

def is_model_not_found_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return ("model_not_found" in s) or ("does not exist" in s) or ("404" in s) or ("model" in s and "not" in s)

if not GROQ_MODEL:
    try:
        auto = None
        try:
            resp = groq_client.models.list()
            auto = next(iter(_extract_model_names(resp)), None)
        except Exception:
            pass
        if auto:
            GROQ_MODEL = auto
            logger.info("GROQ_MODEL автоматически установлен в '%s'", GROQ_MODEL)
        else:
            logger.error("GROQ_MODEL не задан. Установите GROQ_MODEL в .env")
            sys.exit(1)
    except Exception:
        logger.error("GROQ_MODEL не задан. Установите GROQ_MODEL в .env")
        sys.exit(1)

def groq_create_completion(messages, max_tokens=1024, temperature=0.7):
    return groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

# --- localization & assistant identity ---
LOCALE_GREETING = {
    "ru": "Привет! Я — бот NeuroChat. Выберите вкладку или напишите сообщение.",
    "en": "Hi! I'm NeuroChat. Choose a tab or send a message.",
    "uk": "Привіт! Я — NeuroChat. Оберіть вкладку або напишіть повідомлення.",
    "es": "¡Hola! Soy NeuroChat. Elige una pestaña o envía un mensaje.",
}

def user_language_from_msg(msg: Message) -> str:
    code = getattr(msg.from_user, "language_code", "") or ""
    code = code.lower()
    if code.startswith("ru"): return "ru"
    if code.startswith("uk"): return "uk"
    if code.startswith("es"): return "es"
    if code.startswith("en"): return "en"
    return "ru"

def make_system_instruction(lang_code: str) -> str:
    return (
        "You are NeuroChat, a helpful virtual assistant developed by BARRSIKE312 Studio. "
        "You are not ChatGPT and are not affiliated with OpenAI. "
        f"Default response language: {lang_code}. If the user explicitly requests another language, respond in that language."
    )

# --- database (persistent memory & chats) ---
DB_PATH = os.path.join(os.path.dirname(__file__), "memory.db")

def init_db(path: str):
    conn = sqlite3.connect(path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            role TEXT,
            content TEXT,
            ts TEXT,
            chat_id INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT,
            ts TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            user_id INTEGER PRIMARY KEY,
            current_chat_id INTEGER
        )
    """)
    conn.commit()
    return conn

db_conn = init_db(DB_PATH)

def create_chat(user_id: int, name: Optional[str] = None) -> int:
    if not name:
        name = f"Чат {datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    cur = db_conn.cursor()
    cur.execute("INSERT INTO chats (user_id, name, ts) VALUES (?, ?, ?)",
                (user_id, name, datetime.utcnow().isoformat()))
    db_conn.commit()
    chat_id = cur.lastrowid
    set_current_chat(user_id, chat_id)
    return chat_id

def list_chats(user_id: int):
    cur = db_conn.cursor()
    cur.execute("SELECT id, name, ts FROM chats WHERE user_id = ? ORDER BY id DESC", (user_id,))
    return cur.fetchall()

def set_current_chat(user_id: int, chat_id: int):
    cur = db_conn.cursor()
    cur.execute("INSERT OR REPLACE INTO sessions (user_id, current_chat_id) VALUES (?, ?)", (user_id, chat_id))
    db_conn.commit()

def get_current_chat(user_id: int) -> int:
    cur = db_conn.cursor()
    cur.execute("SELECT current_chat_id FROM sessions WHERE user_id = ?", (user_id,))
    r = cur.fetchone()
    if r and r[0]:
        return int(r[0])
    return create_chat(user_id, "Новый чат")

def save_message(user_id: int, role: str, content: str, chat_id: int = None):
    if not content:
        return
    if chat_id is None:
        chat_id = get_current_chat(user_id)
    try:
        cur = db_conn.cursor()
        cur.execute(
            "INSERT INTO messages (user_id, role, content, ts, chat_id) VALUES (?, ?, ?, ?, ?)",
            (user_id, role, content, datetime.utcnow().isoformat(), chat_id),
        )
        db_conn.commit()
    except Exception as e:
        logger.debug("save_message error: %s", e)

def load_memory(user_id: int, limit: int = 50, chat_id: int = None):
    try:
        cur = db_conn.cursor()
        if chat_id is None:
            chat_id = get_current_chat(user_id)
        cur.execute(
            "SELECT role, content FROM messages WHERE user_id = ? AND chat_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, chat_id, limit),
        )
        rows = cur.fetchall()
        rows.reverse()
        return [{"role": row[0], "content": row[1]} for row in rows]
    except Exception as e:
        logger.debug("load_memory error: %s", e)
        return []

def delete_chat(user_id: int, chat_id: int):
    try:
        cur = db_conn.cursor()
        cur.execute("DELETE FROM messages WHERE user_id = ? AND chat_id = ?", (user_id, chat_id))
        cur.execute("DELETE FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id))
        cur.execute("SELECT current_chat_id FROM sessions WHERE user_id = ?", (user_id,))
        r = cur.fetchone()
        if r and r[0] == chat_id:
            cur.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        db_conn.commit()
        return True
    except Exception as e:
        logger.debug("delete_chat error: %s", e)
        return False

# --- Subscriptions & free trials ---
subscriptions = {}  # user_id -> expiry_timestamp
free_trials = {}    # user_id -> trial_end_timestamp

def is_subscribed(user_id: int) -> bool:
    if user_id in subscriptions:
        return subscriptions[user_id] > pydt.datetime.utcnow().timestamp()
    return False

def add_subscription(user_id: int, days: int = 30):
    expiry = pydt.datetime.utcnow().timestamp() + days * 86400
    subscriptions[user_id] = expiry

def has_access(user_id: int) -> bool:
    if user_id in ADMINS:
        return True
    if is_subscribed(user_id):
        return True
    if user_id in free_trials:
        return free_trials[user_id] > pydt.datetime.utcnow().timestamp()
    # grant 7 days free on first check
    free_trials[user_id] = pydt.datetime.utcnow().timestamp() + 7 * 86400
    return True

def access_needed_text() -> str:
    return (
        "Доступ ограничен. Оплатите подписку на 1 месяц — 50₽, "
        "или получите 7 дней бесплатного доступа. "
        "После пробной недели потребуется оформить подписку."
    )

# --- Payments handlers ---
@router.message(Command("subscribe"))
async def subscribe_handler(msg: Message):
    user_id = msg.from_user.id
    # allow admins to 'use' subscribe command too (they may skip paying)
    if is_subscribed(user_id):
        await msg.answer("✅ У тебя уже есть активная подписка!")
        return
    prices = [LabeledPrice(label="Подписка на 1 месяц", amount=5000)]  # 50.00 RUB -> 5000 kopeek
    if not PROVIDER_TOKEN:
        await msg.answer("Платёжный провайдер не настроен. Обратитесь к администратору.")
        return
    try:
        await bot.send_invoice(
            chat_id=msg.chat.id,
            title="Доступ к ИИ-боту",
            description="Подписка на 1 месяц. После оплаты — полный доступ к боту.",
            payload=f"subscription_month:{msg.from_user.id}",
            provider_token=PROVIDER_TOKEN,
            currency="RUB",
            prices=prices,
            start_parameter="subscribe",
            need_name=False,
            need_phone_number=False,
            need_email=False,
            need_shipping_address=False,
            is_flexible=False
        )
    except Exception as e:
        logger.exception("send_invoice error: %s", e)
        await msg.answer("Не удалось инициировать оплату. Проверьте настройки провайдера.")

@router.pre_checkout_query()
async def pre_checkout_query(pre_checkout_q: PreCheckoutQuery):
    await bot.answer_pre_checkout_query(pre_checkout_q.id, ok=True)

@router.message(F.content_type == ContentType.SUCCESSFUL_PAYMENT)
async def successful_payment(msg: Message):
    user_id = msg.from_user.id
    add_subscription(user_id, days=30)
    await msg.answer(
        "🎉 Оплата прошла успешно!\n\n"
        "Теперь у тебя есть полный доступ к боту на 30 дней."
    )

# --- /start handler (shows tab buttons) ---
@router.message(CommandStart())
async def cmd_start(msg: Message):
    user_id = msg.from_user.id
    if not has_access(user_id):
        await msg.answer(access_needed_text())
        # show quick buttons including subscription
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="💳 Подписка", callback_data="tab:subscribe"),
             InlineKeyboardButton(text="🧾 /subscribe", callback_data="tab:subscribe")],
        ])
        await msg.answer("Выберите:", reply_markup=kb)
        return
    lang = user_language_from_msg(msg)
    greeting = LOCALE_GREETING.get(lang, LOCALE_GREETING["ru"])
    cur_chat = get_current_chat(user_id)
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📝 Новый чат", callback_data="tab:new_chat"),
         InlineKeyboardButton(text="📂 Прошлые чаты", callback_data="tab:all_chats")],
        [InlineKeyboardButton(text="💳 Подписка", callback_data="tab:subscribe"),
         InlineKeyboardButton(text="🧹 Очистить чат", callback_data="tab:clear_chat")],
        [InlineKeyboardButton(text="❌ Удалить все", callback_data="tab:clear_all")]
    ])
    await msg.answer(f"{greeting}\nТекущий чат: #{cur_chat}", reply_markup=kb)

# --- general text handler (non-command) ---
@router.message(F.text)
async def text_handler(msg: Message):
    # avoid intercepting commands handled by Command filters
    if isinstance(msg.text, str) and msg.text.startswith("/"):
        return

    user_id = msg.from_user.id
    if not has_access(user_id):
        await msg.answer(access_needed_text())
        return

    text = (msg.text or "").strip()
    lang = user_language_from_msg(msg)

    # save user msg
    save_message(user_id, "user", text)

    # build context from memory for current chat
    history = load_memory(user_id, limit=40)
    system_msg = make_system_instruction(lang)
    messages = [{"role": "system", "content": system_msg}] + history + [{"role": "user", "content": text}]

    try:
        resp = groq_create_completion(messages=messages)
        try:
            content = resp.choices[0].message.content
        except Exception:
            content = str(resp)
        save_message(user_id, "assistant", content)
        await msg.answer(content)
    except Exception as e:
        logger.exception("Ошибка при создании completion")
        if is_model_not_found_error(e):
            try:
                available = await list_available_models(groq_client)
            except Exception:
                available = []
            hint = ""
            if available:
                hint = "Доступные модели (первые):\n" + "\n".join(available[:10]) + "\n\n"
            hint += "Обновите GROQ_MODEL в .env или проверьте ключ API."
            if user_id in ADMINS:
                await msg.answer("Ошибка модели Groq: указанная модель недоступна или не найдена.\n\n" + hint)
            else:
                await msg.answer("Ошибка модели Groq: указанная модель недоступна. Свяжитесь с администратором.")
        else:
            await msg.answer(f"Ошибка Groq: {str(e)}")

# photo handler (downloads, OCR, uses current chat memory)
@router.message(F.photo)
async def photo_handler(msg: Message):
    user_id = msg.from_user.id
    if not has_access(user_id):
        await msg.answer(access_needed_text())
        return

    lang = user_language_from_msg(msg)

    try:
        photo = msg.photo[-1]
        tmp_path = None
        try:
            if hasattr(photo, "download"):
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    await photo.download(destination_file=tmp_path)
                except TypeError:
                    await photo.download(tmp_path)
            else:
                raise AttributeError("no photo.download")
        except Exception:
            try:
                file_info = await bot.get_file(photo.file_id)
                file_path = file_info.file_path
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    await bot.download_file(file_path, tmp_path)
                except TypeError:
                    data = await bot.download_file(file_path)
                    with open(tmp_path, "wb") as f:
                        f.write(data)
            except Exception as e:
                logger.exception("Не удалось скачать фото: %s", e)
                await msg.answer("Не удалось скачать фото.")
                if tmp_path and os.path.exists(tmp_path):
                    try: os.remove(tmp_path)
                    except Exception: pass
                return

        try:
            image = Image.open(tmp_path)
            try:
                text = pytesseract.image_to_string(image, lang="rus+eng").strip()
            finally:
                image.close()
        finally:
            try: os.remove(tmp_path)
            except Exception: pass

        if not text:
            await msg.answer("Не удалось распознать текст на фото.")
            return

        save_message(user_id, "user", text)
        history = load_memory(user_id, limit=40)
        system_msg = make_system_instruction(lang)
        user_prompt = f"Распознан текст с фото:\n{text}\n\nПожалуйста, ответь или объясни."
        messages = [{"role": "system", "content": system_msg}] + history + [{"role": "user", "content": user_prompt}]

        try:
            resp = groq_create_completion(messages=messages)
        except Exception as e:
            logger.exception("Ошибка Groq при обработке фото")
            if is_model_not_found_error(e):
                try:
                    available = await list_available_models(groq_client)
                except Exception:
                    available = []
                hint = ""
                if available:
                    hint = "Доступные модели (первые):\n" + "\n".join(available[:10]) + "\n\n"
                hint += "Проверьте GROQ_MODEL в .env."
                if user_id in ADMINS:
                    await msg.answer("Ошибка модели Groq: указанная модель недоступна или не найдена.\n\n" + hint)
                else:
                    await msg.answer("Ошибка модели Groq: указанная модель недоступна. Свяжитесь с администратором.")
            else:
                await msg.answer(f"Ошибка Groq: {str(e)}")
            return

        try:
            answer = resp.choices[0].message.content
        except Exception:
            answer = str(resp)

        save_message(user_id, "assistant", answer)
        await msg.answer(f"Распознано:\n```\n{text}\n```\n\nОтвет:\n{answer}", parse_mode="Markdown")
    except Exception as e:
        logger.exception("Ошибка обработки фото")
        await msg.answer(f"Ошибка обработки: {str(e)}")

# command helpers (tabs) - keep commands too
@router.message(Command("tabs"))
async def cmd_tabs(msg: Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📝 Новый чат", callback_data="tab:new_chat"),
         InlineKeyboardButton(text="📂 Прошлые чаты", callback_data="tab:all_chats")],
        [InlineKeyboardButton(text="💳 Подписка", callback_data="tab:subscribe")]
    ])
    await msg.answer("Выберите вкладку:", reply_markup=kb)

@router.message(Command("new_chat"))
async def cmd_new_chat(msg: Message):
    args = (msg.get_args() or "").strip()
    name = args if args else None
    chat_id = create_chat(msg.from_user.id, name)
    await msg.answer(f"Создан новый чат #{chat_id}. Переключено на этот чат.")

@router.message(Command("all_chats"))
async def cmd_list_chats(msg: Message):
    rows = list_chats(msg.from_user.id)
    if not rows:
        await msg.answer("У вас нет сохранённых чатов. Создайте новый через /new_chat или кнопку 'Новый чат'.")
        return
    kb = InlineKeyboardMarkup(row_width=1)
    for r in rows:
        cid, name, ts = r
        kb.add(InlineKeyboardButton(text=f"{name} (#{cid})", callback_data=f"chat:select:{cid}"))
    await msg.answer("Ваши чаты:", reply_markup=kb)

# callbacks for tabs and chats
@router.callback_query(F.data.startswith("tab:"))
async def cb_tab(query: CallbackQuery):
    data = query.data
    await query.answer()
    uid = query.from_user.id
    if data == "tab:new_chat":
        chat_id = create_chat(uid, None)
        await query.message.answer(f"Создан новый чат #{chat_id}.")
    elif data == "tab:all_chats":
        rows = list_chats(uid)
        if not rows:
            await query.message.answer("У вас нет чатов.")
            return
        kb = InlineKeyboardMarkup(row_width=1)
        for r in rows:
            cid, name, ts = r
            kb.add(InlineKeyboardButton(text=f"{name} (#{cid})", callback_data=f"chat:select:{cid}"))
        await query.message.answer("Ваши чаты:", reply_markup=kb)
    elif data == "tab:subscribe":
        # show subscription info and optionally create invoice
        text = (
            "Оплатите подписку на 1 месяц — 50₽, или получите 7 дней бесплатного доступа. "
            "После пробной недели потребуется оформить подписку."
        )
        if not PROVIDER_TOKEN:
            await query.message.answer(text + "\n\nПлатёжный провайдер не настроен.")
            return
        prices = [LabeledPrice(label="Подписка на 1 месяц", amount=5000)]
        try:
            # send invoice to this chat
            await bot.send_invoice(
                chat_id=query.message.chat.id,
                title="Подписка NeuroChat — 1 месяц",
                description="Подписка на 1 месяц (50₽).",
                payload=f"subscription_month_cb:{uid}",
                provider_token=PROVIDER_TOKEN,
                currency="RUB",
                prices=prices,
                start_parameter="subscribe"
            )
        except Exception as e:
            logger.exception("send_invoice from callback error: %s", e)
            await query.message.answer(text + "\n\nНе удалось инициировать оплату.")
    elif data == "tab:clear_chat":
        cur = get_current_chat(uid)
        if not cur:
            await query.message.answer("Нет активного чата для очистки.")
            return
        try:
            curc = db_conn.cursor()
            curc.execute("SELECT COUNT(*) FROM messages WHERE user_id = ? AND chat_id = ?", (uid, cur))
            to_delete = curc.fetchone()[0] or 0
            curc.execute("DELETE FROM messages WHERE user_id = ? AND chat_id = ?", (uid, cur))
            db_conn.commit()
            await query.message.answer(f"Чат #{cur} очищен. Удалено сообщений: {to_delete}.")
        except Exception as e:
            logger.exception("clear_chat error: %s", e)
            await query.message.answer("Не удалось очистить чат.")
    elif data == "tab:clear_all":
        try:
            curc = db_conn.cursor()
            curc.execute("DELETE FROM messages WHERE user_id = ?", (uid,))
            curc.execute("DELETE FROM chats WHERE user_id = ?", (uid,))
            curc.execute("DELETE FROM sessions WHERE user_id = ?", (uid,))
            db_conn.commit()
            # if chats table is now empty globally, reset autoincrement sequence for chats
            try:
                curc.execute("SELECT COUNT(*) FROM chats")
                total = curc.fetchone()[0] or 0
                if total == 0:
                    # reset sqlite_sequence so next chat id starts from 1
                    try:
                        curc.execute("DELETE FROM sqlite_sequence WHERE name='chats'")
                        db_conn.commit()
                    except Exception:
                        # some SQLite builds may not allow direct manipulation; ignore
                        logger.debug("Не удалось сбросить sqlite_sequence для 'chats'")
            except Exception:
                pass
            new_id = create_chat(uid, "Новый чат")
            await query.message.answer(f"Все чаты удалены. Создан новый чат #{new_id}.")
        except Exception as e:
            logger.exception("clear_all error: %s", e)
            await query.message.answer("Не удалось удалить все чаты.")

@router.callback_query(F.data.startswith("chat:select:"))
async def cb_select_chat(query: CallbackQuery):
    await query.answer()
    try:
        cid = int(query.data.split(":")[-1])
    except Exception:
        await query.message.answer("Неправильный id чата.")
        return
    mem = load_memory(query.from_user.id, limit=20, chat_id=cid)
    preview_lines = []
    for m in mem[-20:]:
        role = "Вы" if m["role"] == "user" else "Бот"
        txt = m["content"].replace("```", "`")
        if len(txt) > 800: txt = txt[:800] + "…"
        preview_lines.append(f"{role}: {txt}")
    preview = "\n".join(preview_lines) if preview_lines else "(пусто)"
    kb = InlineKeyboardMarkup(row_width=2)
    kb.add(InlineKeyboardButton(text="↩️ Активировать", callback_data=f"chat:choose:{cid}"),
           InlineKeyboardButton(text="🗑️ Удалить", callback_data=f"chat:delete:{cid}"))
    await query.message.answer(f"Чат #{cid}\n\nПоследние сообщения:\n{preview}", reply_markup=kb)

@router.callback_query(F.data.startswith("chat:choose:"))
async def cb_choose_chat(query: CallbackQuery):
    await query.answer()
    try:
        cid = int(query.data.split(":")[-1])
    except Exception:
        await query.message.answer("Неправильный id чата.")
        return
    set_current_chat(query.from_user.id, cid)
    await query.message.answer(f"Теперь активный чат #{cid}.")

@router.callback_query(F.data.startswith("chat:delete:"))
async def cb_delete_chat(query: CallbackQuery):
    await query.answer()
    try:
        cid = int(query.data.split(":")[-1])
    except Exception:
        await query.message.answer("Неправильный id чата.")
        return
    ok = delete_chat(query.from_user.id, cid)
    if ok:
        cur = get_current_chat(query.from_user.id)
        await query.message.answer(f"Чат #{cid} удалён. Текущий чат: #{cur}.")
    else:
        await query.message.answer("Не удалось удалить чат.")

# --- main ---
async def main():
    dp.include_router(router)
    logger.info("Бот запущен. Начинаю polling.")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        try:
            asyncio.run(bot.session.close())
        except Exception:
            pass