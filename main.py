import os
import re
import httpx
import logging
import asyncio
import sqlite3
import json
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, constants, BotCommand, BotCommandScopeAllPrivateChats, BotCommandScopeAllGroupChats
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from telegram.error import BadRequest

# ==============================================================================
# --- 1. CONFIGURATION ---
# ==============================================================================

# Load environment variables from a .env file
load_dotenv()

# --- Basic Setup & Configuration ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- API Keys ---
# Fetches your secret keys from environment variables.
try:
    TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
except KeyError:
    logger.critical("FATAL: Environment variables TELEGRAM_BOT_TOKEN or GEMINI_API_KEY not found.")
    logger.critical("Please create a .env file and add your keys to it.")
    exit()

# --- Model Definitions ---
AVAILABLE_MODELS = {
    "pro": ("Pro 🧠 (Smartest)", "gemini-2.5-pro", 2500),
    "flash": ("Flash ✨ (Balanced)", "gemini-2.5-flash", 2500),
    "flash_lite": ("Flash-Lite ⚡️ (Fastest)", "gemini-2.5-flash-lite", 2500),
}

# --- Bot Personality ---
# System instruction for standard AI behavior
BOT_SYSTEM_INSTRUCTION = {
    "role": "model",
    "parts": [{
        "text": "You are a helpful and friendly AI assistant."
    }]
}

# --- Database Configuration ---
DB_FILE = "gemini_bot.db"


# ==============================================================================
# --- 2. DATABASE ---
# ==============================================================================

def init_db():
    """Initializes the database and creates tables if they don't exist."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_settings (
                chat_id INTEGER PRIMARY KEY,
                model_id TEXT,
                token_limit INTEGER
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_histories (
                chat_id INTEGER PRIMARY KEY,
                history TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_stats (
                chat_id INTEGER PRIMARY KEY,
                stats TEXT
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully.")
    except sqlite3.Error as e:
        logger.error(f"Database error during initialization: {e}")

def save_model_selection(chat_id: int, model_name: str, token_limit: int):
    """Saves the chosen model and token limit."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("REPLACE INTO chat_settings (chat_id, model_id, token_limit) VALUES (?, ?, ?)",
                       (chat_id, model_name, token_limit))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        logger.error(f"Failed to save model selection for chat {chat_id}: {e}")

def save_json_data(table_name: str, chat_id: int, data: dict or list):
    """Saves JSON-serializable data (like history or stats) to the specified table."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        column_name = "history" if table_name == "chat_histories" else "stats"
        cursor.execute(f"REPLACE INTO {table_name} (chat_id, {column_name}) VALUES (?, ?)",
                       (chat_id, json.dumps(data)))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        logger.error(f"Failed to save JSON data to '{table_name}' for chat {chat_id}: {e}")

def load_chat_data(chat_id: int) -> dict:
    """Loads all data for a a chat from the DB."""
    data = {
        "model_selection": None,
        "history": [],
        "stats": {"message_count": 0, "model_usage": {}}
    }
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        cursor.execute("SELECT model_id, token_limit FROM chat_settings WHERE chat_id = ?", (chat_id,))
        settings = cursor.fetchone()
        if settings:
            data["model_selection"] = (settings[0], settings[1])

        cursor.execute("SELECT history FROM chat_histories WHERE chat_id = ?", (chat_id,))
        history_data = cursor.fetchone()
        if history_data:
            data["history"] = json.loads(history_data[0])

        cursor.execute("SELECT stats FROM chat_stats WHERE chat_id = ?", (chat_id,))
        stats_data = cursor.fetchone()
        if stats_data:
            data["stats"] = json.loads(stats_data[0])

        conn.close()
        logger.info(f"Successfully loaded data for chat {chat_id} from database.")
    except sqlite3.Error as e:
        logger.error(f"Failed to load data for chat {chat_id}: {e}")

    return data


# ==============================================================================
# --- 3. UTILITIES ---
# ==============================================================================

def format_text_for_telegram(text: str) -> str:
    """Converts various markdown styles from Gemini to Telegram-supported HTML."""
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text, flags=re.DOTALL)

    def escape_code_block(match):
        language = match.group(1) or ""
        code = match.group(2)
        code = code.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        return f'<pre><code class="language-{language}">{code}</code></pre>'
    text = re.sub(r'```(\w*?)\n(.*?)```', escape_code_block, text, flags=re.DOTALL)

    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    text = re.sub(r'^\s*\*\s+', '• ', text, flags=re.MULTILINE)
    return text


# ==============================================================================
# --- 4. HANDLERS ---
# ==============================================================================

# --- State Management Caches ---
chat_model_selections = {}
chat_histories = {}
chat_stats = {}

async def load_chat_data_into_cache(chat_id: int):
    """Loads all data for a chat from the DB into the cache if not already present."""
    if chat_id in chat_model_selections:
        return

    data = load_chat_data(chat_id)
    chat_model_selections[chat_id] = data["model_selection"]
    chat_histories[chat_id] = data["history"]
    chat_stats[chat_id] = data["stats"]

# --- Command Handlers ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for the /start command. Presents model selection and clears history."""
    chat_id = update.message.chat_id
    chat_histories[chat_id] = []
    save_json_data("chat_histories", chat_id, [])
    logger.info(f"Chat {chat_id} history cleared by /start command.")
    await model_command(update, context) # Delegate to model selection

async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for the /model command. Presents model selection."""
    keyboard = [
        [InlineKeyboardButton(display_name, callback_data=f"model_{unique_id}")]
        for unique_id, (display_name, _, _) in AVAILABLE_MODELS.items()
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "Please select a model to begin.",
        reply_markup=reply_markup
    )

async def newchat_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for the /newchat command."""
    chat_id = update.message.chat_id
    chat_histories[chat_id] = []
    save_json_data("chat_histories", chat_id, [])
    logger.info(f"Chat {chat_id} history cleared by /newchat command.")
    await update.message.reply_text("New chat session started. The conversation history has been cleared.")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Displays usage stats for the current chat."""
    chat_id = update.message.chat_id
    await load_chat_data_into_cache(chat_id)
    stats = chat_stats.get(chat_id, {"message_count": 0, "model_usage": {}})
    message_count = stats.get("message_count", 0)
    model_usage = stats.get("model_usage", {})
    if not message_count:
        await update.message.reply_text("No messages have been sent in this chat yet.")
        return
    usage_text = "\n".join([f"    - {model}: {count} times" for model, count in model_usage.items()])
    stats_message = f"""
📊 **Chat Statistics** 📊
💬 *Messages Sent:* {message_count}
🤖 *Model Usage:*
{usage_text if usage_text else "    - No models used yet."}
    """
    await update.message.reply_text(stats_message, parse_mode=constants.ParseMode.HTML)

# --- Callback Query Handlers (Button Clicks) ---
async def main_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """The main router for all button clicks."""
    query = update.callback_query
    try:
        await query.answer()
    except BadRequest as e:
        logger.warning(f"Failed to answer callback query, it might be old: {e}")
        return
    callback_data = query.data
    if callback_data.startswith("model_"):
        await model_selection_handler(update, context)

async def model_selection_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the selection of a new model."""
    query = update.callback_query
    chat_id = query.message.chat_id
    selected_id = query.data.split("model_")[1]
    display_name, model_name, token_limit = AVAILABLE_MODELS[selected_id]
    chat_model_selections[chat_id] = (model_name, token_limit)
    save_model_selection(chat_id, model_name, token_limit)
    await query.edit_message_text(
        text=f"Model set to: <b>{display_name}</b>.",
        parse_mode=constants.ParseMode.HTML
    )

# --- Main Message Logic ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """The main message handler, now without any tool-use capabilities."""
    chat_id = update.message.chat_id
    user_text = update.message.text
    chat_type = update.message.chat.type

    await load_chat_data_into_cache(chat_id)

    # --- Group Chat Logic ---
    should_reply = False
    if chat_type == 'private':
        should_reply = True
    elif chat_type in ['group', 'supergroup']:
        bot_username = context.bot.username
        if f"@{bot_username}" in user_text:
            user_text = user_text.replace(f"@{bot_username}", "").strip()
            should_reply = True
        elif update.message.reply_to_message and update.message.reply_to_message.from_user.username == bot_username:
            should_reply = True

    if not should_reply:
        return
    # --- End of Group Chat Logic ---

    if chat_id not in chat_model_selections or not chat_model_selections[chat_id]:
        await update.message.reply_text(
            "Please select a model first using the /start or /model command."
        )
        return

    if not user_text:
        await update.message.reply_text("Please provide a prompt.")
        return

    await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)

    model_name, token_limit = chat_model_selections[chat_id]
    history = chat_histories.get(chat_id, [])
    history.append({"role": "user", "parts": [{"text": user_text}]})

    # Construct the message payload with system instruction
    contents = [BOT_SYSTEM_INSTRUCTION] + history

    try:
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
        
        payload = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": token_limit},
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, json=payload, timeout=300.0)
            response.raise_for_status()

        data = response.json()
        candidate = data.get('candidates', [{}])[0]
        
        if candidate.get('finishReason') in ('SAFETY', 'RECITATION'):
            await update.message.reply_text("The response was blocked due to safety concerns.")
            return

        part = candidate.get('content', {}).get('parts', [{}])[0]

        if 'text' in part:
            bot_response = part['text']
            history.append({"role": "model", "parts": [{"text": bot_response}]})
            chat_histories[chat_id] = history
            
            stats = chat_stats.get(chat_id, {"message_count": 0, "model_usage": {}})
            stats["message_count"] += 1
            model_display_name = next((val[0] for key, val in AVAILABLE_MODELS.items() if val[1] == model_name), "Unknown")
            stats["model_usage"][model_display_name] = stats["model_usage"].get(model_display_name, 0) + 1
            chat_stats[chat_id] = stats
            
            save_json_data("chat_histories", chat_id, history)
            save_json_data("chat_stats", chat_id, stats)

            formatted_response = format_text_for_telegram(bot_response)

            if len(formatted_response) > 4096:
                chunks = [formatted_response[i:i + 4096] for i in range(0, len(formatted_response), 4096)]
                for i, chunk in enumerate(chunks):
                    await update.message.reply_text(chunk, parse_mode=constants.ParseMode.HTML)
                    if i < len(chunks) - 1:
                        await asyncio.sleep(1)
            else:
                await update.message.reply_text(formatted_response, parse_mode=constants.ParseMode.HTML)
        
        else:
             logger.warning(f"Gemini API returned an unexpected response structure for chat {chat_id}. Response: {data}")
             await update.message.reply_text("An unexpected error occurred while processing the response.")

    except httpx.HTTPStatusError as http_err:
        logger.error(f"HTTP error occurred: {http_err} - {http_err.response.text}")
        await update.message.reply_text(f"An HTTP error occurred: {http_err}")
    except httpx.TimeoutException:
        logger.error(f"Request to Gemini API timed out for chat {chat_id}.")
        await update.message.reply_text("The request timed out. Please try again.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        await update.message.reply_text(f"An unexpected error occurred: {e}")

async def post_init(application: Application) -> None:
    """A function to run after the bot has been initialized, to set the bot commands."""
    commands = [
        BotCommand("start", "Start a new chat & select model"),
        BotCommand("model", "Change AI model"),
        BotCommand("newchat", "Start a new chat with current settings"),
        BotCommand("stats", "Show your usage stats"),
    ]
    await application.bot.set_my_commands(commands, scope=BotCommandScopeAllPrivateChats())
    await application.bot.set_my_commands(commands, scope=BotCommandScopeAllGroupChats())
    logger.info("Custom commands have been set for private and group chats.")

# ==============================================================================
# --- 5. MAIN ---
# ==============================================================================

def main() -> None:
    """The main function that initializes and starts the bot."""
    init_db()
    logger.info("Starting bot...")

    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    # --- Register all handlers ---
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("model", model_command))
    application.add_handler(CommandHandler("newchat", newchat_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CallbackQueryHandler(main_callback_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()
    logger.info("Bot stopped.")

if __name__ == "__main__":
    main()
