"""
Web3 Security Mentor Bot - FastAPI + Telegram + OpenAI
Single-file implementation: main.py

Features:
- Telegram webhook receiver
- Basic conversational state per user (presence check, awaiting explanation, audit tracker flows)
- Uses OpenAI (Chat Completions / Responses) to evaluate user explanations and generate: corrections, 3 Solidity examples, 1 Foundry test, 1 exploit sketch, related blog links, 5-question quiz
- Simple SQLite persistence for user state and audit tracker

Requirements:
- Python 3.10+
- pip install fastapi uvicorn httpx python-dotenv

Environment variables (create a .env file):
- TELEGRAM_TOKEN=<your_telegram_bot_token>
- OPENAI_API_KEY=<your_openai_api_key>
- BASE_URL=<https://yourdomain.com> (public url where webhook will be set)

Quick start (local testing with ngrok):
1. Start app: uvicorn main:app --host 0.0.0.0 --port 8000
2. Expose with ngrok: ngrok http 8000
3. Set BASE_URL to the https://...ngrok.io value
4. Call /set_webhook to register webhook (or use setWebhook via curl)

Notes:
- This is a template and intentionally pragmatic. For production, secure the endpoint, use long-term DB, background jobs, rate-limiting, and better error handling.
- The OpenAI usage asks the model to be strict and produce thorough technical output. Keep an eye on token usage.

"""

import os
import sqlite3
import json
import asyncio
from typing import Optional
from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")  # e.g. https://xxxx.ngrok.io

if not TELEGRAM_TOKEN or not OPENAI_API_KEY:
    raise RuntimeError("Set TELEGRAM_TOKEN and OPENAI_API_KEY in environment (.env)")

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

app = FastAPI()

DB_PATH = "mentor_bot.db"

# Initialize DB
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    chat_id INTEGER PRIMARY KEY,
    state TEXT,
    awaiting_field TEXT,
    last_topic TEXT
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS audits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER,
    name TEXT,
    urls TEXT,
    status TEXT,
    issues_found TEXT,
    missed_issues TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# --- Helpers ---
async def telegram_send_message(chat_id: int, text: str, parse_mode: str = "Markdown"):
    async with httpx.AsyncClient() as client:
        await client.post(f"{TELEGRAM_API}/sendMessage", json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        })

async def set_webhook():
    url = f"{BASE_URL}/webhook"
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{TELEGRAM_API}/setWebhook", json={"url": url})
        return r.json()

# OpenAI chat helper
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

async def ask_openai_system(user_topic: str, user_explanation: str) -> str:
    """
    Sends a detailed system/user prompt to OpenAI and returns a markdown response.
    The model is instructed to evaluate, correct, and produce examples, test, exploit, blogs, and quiz.
    """
    system_prompt = (
        "You are an expert Web3 security mentor and senior auditor. \n"
        "When given a user's explanation of a topic, do the following precisely:\n"
        "1) Evaluate the explanation briefly, list mistakes if any.\n"
        "2) Provide a corrected, concise explanation with important details.\n"
        "3) Provide three small Solidity coding examples the learner can run to recall the topic (label Example 1/2/3).\n"
        "4) Provide one Foundry-style test skeleton (Rust/forge style) to validate a key invariant related to the topic.\n"
        "5) Provide one exploit sketch / step-by-step POC (high-level) showing how an attacker would abuse the issue.\n"
        "6) Provide 2-3 reputable blog/article links discussing the attack vector. Keep links short.\n"
        "7) Provide a 5-question micro-quiz (questions only, no answers).\n"
        "Respond in clear Markdown sections with headers: Evaluation, Corrected Explanation, Examples, Foundry Test, Exploit Sketch, Links, Quiz. Keep code blocks short and runnable. Aim for concision and technical accuracy.\n"
    )

    user_prompt = (
        f"Topic: {user_topic}\n\nUser explanation:\n{user_explanation}\n\n"
        "Follow the instructions in the system prompt and produce the sections."
    )

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "gpt-4o-mini",  # adjust model as needed
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 1200,
        "temperature": 0.1
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(OPENAI_CHAT_URL, headers=headers, json=body)
        r.raise_for_status()
        data = r.json()
        ans = data["choices"][0]["message"]["content"]
        return ans

# --- State helpers ---

def get_user(chat_id: int):
    cur.execute("SELECT chat_id, state, awaiting_field, last_topic FROM users WHERE chat_id=?", (chat_id,))
    row = cur.fetchone()
    if row:
        return {"chat_id": row[0], "state": row[1], "awaiting_field": row[2], "last_topic": row[3]}
    # create default
    cur.execute("INSERT OR REPLACE INTO users (chat_id, state, awaiting_field, last_topic) VALUES (?,?,?,?)",
                (chat_id, "idle", None, None))
    conn.commit()
    return {"chat_id": chat_id, "state": "idle", "awaiting_field": None, "last_topic": None}


def update_user(chat_id: int, state: Optional[str] = None, awaiting_field: Optional[str] = None, last_topic: Optional[str] = None):
    user = get_user(chat_id)
    new_state = state if state is not None else user["state"]
    new_await = awaiting_field if awaiting_field is not None else user["awaiting_field"]
    new_topic = last_topic if last_topic is not None else user["last_topic"]
    cur.execute("UPDATE users SET state=?, awaiting_field=?, last_topic=? WHERE chat_id=?",
                (new_state, new_await, new_topic, chat_id))
    conn.commit()

# --- Audit tracker helpers ---

def add_audit(chat_id: int, name: str, urls: str, status: str, issues: str, missed: str, notes: str):
    cur.execute("INSERT INTO audits (chat_id, name, urls, status, issues_found, missed_issues, notes) VALUES (?,?,?,?,?,?,?)",
                (chat_id, name, urls, status, issues, missed, notes))
    conn.commit()


def get_audits(chat_id: int):
    cur.execute("SELECT id, name, urls, status, issues_found, missed_issues, notes, created_at FROM audits WHERE chat_id=? ORDER BY created_at DESC",
                (chat_id,))
    rows = cur.fetchall()
    return rows

# --- Telegram message parsing flow ---

@app.post("/webhook")
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    update = await request.json()
    # Basic safety: ensure it is a message update
    if "message" not in update:
        return {"ok": True}

    message = update["message"]
    chat_id = message["chat"]["id"]
    text = message.get("text", "").strip()

    user = get_user(chat_id)

    # Commands
    if text.startswith("/start"):
        await telegram_send_message(chat_id, "I am your Web3 Security Mentor Bot.\nAre you here today for your security practice? (Reply 'yes' or 'no')")
        update_user(chat_id, state="awaiting_presence")
        return {"ok": True}

    # Show audit tracker
    if text.lower() == "show audit tracker":
        rows = get_audits(chat_id)
        if not rows:
            await telegram_send_message(chat_id, "No audits tracked yet. Use 'Add audit' to create one.")
            return {"ok": True}
        msg = "*Your Audit Tracker*\n"
        for r in rows:
            msg += f"\n*{r[1]}* (id:{r[0]})\nURLs: {r[2]}\nStatus: {r[3]}\nIssues: {r[4]}\nMissed: {r[5]}\nNotes: {r[6]}\n"
        await telegram_send_message(chat_id, msg)
        return {"ok": True}

    # Add audit flow
    if text.lower() == "add audit":
        await telegram_send_message(chat_id, "Okay — let's add a new audit. What is the *Audit Name*? (Send name)")
        update_user(chat_id, state="adding_audit", awaiting_field="name")
        return {"ok": True}

    # If user is in adding_audit state, handle fields sequentially
    if user["state"] == "adding_audit":
        awaiting = user["awaiting_field"]
        # We'll store temporary data in a very simple way: use last_topic as JSON for the flow
        temp = {}
        if user["last_topic"]:
            try:
                temp = json.loads(user["last_topic"])
            except Exception:
                temp = {}
        if awaiting == "name":
            temp["name"] = text
            update_user(chat_id, last_topic=json.dumps(temp), awaiting_field="urls")
            await telegram_send_message(chat_id, "Send Contract URL(s) (comma separated)")
            return {"ok": True}
        if awaiting == "urls":
            temp["urls"] = text
            update_user(chat_id, last_topic=json.dumps(temp), awaiting_field="status")
            await telegram_send_message(chat_id, "Status (e.g., in-progress, completed)")
            return {"ok": True}
        if awaiting == "status":
            temp["status"] = text
            update_user(chat_id, last_topic=json.dumps(temp), awaiting_field="issues")
            await telegram_send_message(chat_id, "Issues found (short summary)")
            return {"ok": True}
        if awaiting == "issues":
            temp["issues"] = text
            update_user(chat_id, last_topic=json.dumps(temp), awaiting_field="missed")
            await telegram_send_message(chat_id, "Post-evaluation: Missed issues (if any). If none, write 'none'.")
            return {"ok": True}
        if awaiting == "missed":
            temp["missed"] = text
            update_user(chat_id, last_topic=json.dumps(temp), awaiting_field="notes")
            await telegram_send_message(chat_id, "Add any notes (or type 'none')")
            return {"ok": True}
        if awaiting == "notes":
            temp["notes"] = text
            # Save to DB
            add_audit(chat_id, temp.get("name",""), temp.get("urls",""), temp.get("status",""), temp.get("issues",""), temp.get("missed",""), temp.get("notes",""))
            update_user(chat_id, state="idle", awaiting_field=None, last_topic=None)
            await telegram_send_message(chat_id, f"Audit '{temp.get('name')}' added to your tracker.")
            return {"ok": True}

    # Presence flow
    if user["state"] == "awaiting_presence":
        if text.lower() in ["yes", "y", "i am here", "here"]:
            await telegram_send_message(chat_id, "Great! What topic did you learn today? Please explain it in your own words.")
            update_user(chat_id, state="awaiting_explanation", awaiting_field=None)
            return {"ok": True}
        else:
            await telegram_send_message(chat_id, "No worries. What happened today? If you'd like a short revision quiz instead, type 'revision'.")
            update_user(chat_id, state="idle")
            return {"ok": True}

    # If expecting an explanation
    if user["state"] == "awaiting_explanation":
        # capture topic from first line if user wrote 'Topic: ...' else ask
        user_text = text
        # Try to extract topic name heuristically (first line or fallback to 'general')
        first_line = user_text.splitlines()[0].strip()
        topic = first_line if len(first_line.split()) < 8 else "general smart contract topic"
        update_user(chat_id, state="idle", last_topic=topic)
        # Call OpenAI in background
        background_tasks.add_task(handle_explanation_and_reply, chat_id, topic, user_text)
        await telegram_send_message(chat_id, "Thanks — evaluating your explanation and preparing examples. I'll send detailed feedback shortly.")
        return {"ok": True}

    # Fallback small commands
    if text.lower() in ["help", "/help"]:
        help_msg = (
            "Commands:\n/start - start practice\nAdd audit - add an audit to tracker\nShow audit tracker - display audits\nHelp - show this message\n"
        )
        await telegram_send_message(chat_id, help_msg)
        return {"ok": True}

    # Default reply
    await telegram_send_message(chat_id, "I didn't understand that. Send /start to begin a practice session or 'Add audit' to track an audit.")
    return {"ok": True}


async def handle_explanation_and_reply(chat_id: int, topic: str, user_explanation: str):
    try:
        ans = await ask_openai_system(topic, user_explanation)
    except Exception as e:
        await telegram_send_message(chat_id, f"OpenAI call failed: {e}\nTry again later.")
        return
    # send the long markdown answer in chunks so Telegram doesn't reject it
    chunks = split_text_chunks(ans, 3800)
    for c in chunks:
        await telegram_send_message(chat_id, c)


def split_text_chunks(text: str, max_len: int = 3500):
    parts = []
    while text:
        if len(text) <= max_len:
            parts.append(text)
            break
        # split at last newline before max_len
        idx = text.rfind('\n', 0, max_len)
        if idx == -1:
            idx = max_len
        parts.append(text[:idx])
        text = text[idx:]
    return parts


@app.get("/set_webhook")
async def route_set_webhook():
    if not BASE_URL:
        return {"error": "Set BASE_URL environment variable to your public URL."}
    result = await set_webhook()
    return result


# Run with: uvicorn main:app --reload

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

