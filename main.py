"""
Web3 Security Mentor Bot - FastAPI + Telegram + OpenAI
Rewritten main.py (improved):

Improvements made:
- Robust conversation state machine (explicit states + temp storage)
- Accepts free-form multi-line topic + explanation (no strict formatting)
- SQLite persistence for users + audits (temp JSON stored in users.temp)
- OpenAI requests with exponential backoff and 429 handling
- Concurrency limit on OpenAI calls to avoid rate limits
- Better error handling and logging
- /start, /help, /reset, Add audit, Show audit tracker implemented
- /set_webhook helper route
- Healthcheck route

Usage:
- Set env: TELEGRAM_TOKEN, OPENAI_API_KEY, BASE_URL (optional for set_webhook), DATABASE_URL (optional, default sqlite file)
- Run: uvicorn main:app --host 0.0.0.0 --port 8000

"""

import os
import sqlite3
import json
import asyncio
import time
from typing import Optional
from fastapi import FastAPI, Request, BackgroundTasks
import httpx
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")  # optional
DATABASE_URL = os.getenv("DATABASE_URL") or "sqlite:///./mentor_bot.db"

if not TELEGRAM_TOKEN or not OPENAI_API_KEY:
    raise RuntimeError("Set TELEGRAM_TOKEN and OPENAI_API_KEY in environment (.env)")

# Derive sqlite file path if DATABASE_URL is sqlite
if DATABASE_URL.startswith("sqlite:///"):
    DB_PATH = DATABASE_URL.replace("sqlite:///", "")
else:
    # fallback default
    DB_PATH = "mentor_bot.db"

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

app = FastAPI()

# concurrency limiter for OpenAI calls (avoid bursts)
OPENAI_SEMAPHORE = asyncio.Semaphore(2)

# initialize sqlite
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    chat_id INTEGER PRIMARY KEY,
    state TEXT DEFAULT 'idle',
    temp TEXT DEFAULT NULL,
    last_topic TEXT DEFAULT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

# --- helpers ---
async def telegram_send_message(chat_id: int, text: str, parse_mode: str = "Markdown"):
    # safe send with retries
    payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode, "disable_web_page_preview": True}
    async with httpx.AsyncClient() as client:
        for attempt in range(3):
            try:
                r = await client.post(f"{TELEGRAM_API}/sendMessage", json=payload, timeout=30.0)
                if r.status_code == 200:
                    return True
            except Exception:
                await asyncio.sleep(1 + attempt)
    return False

async def set_webhook():
    if not BASE_URL:
        raise RuntimeError("BASE_URL not set")
    url = f"{BASE_URL}/webhook"
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{TELEGRAM_API}/setWebhook", json={"url": url})
        return r.json()

# --- OpenAI with retries and backoff ---
async def openai_chat_with_retries(system_prompt: str, user_prompt: str, max_retries: int = 4) -> str:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 1200,
        "temperature": 0.12
    }

    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        await OPENAI_SEMAPHORE.acquire()
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(OPENAI_CHAT_URL, headers=headers, json=body)
            if r.status_code == 200:
                data = r.json()
                return data["choices"][0]["message"]["content"]
            # handle rate limit
            if r.status_code in (429, 503, 502):
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            # propagate non-retryable errors
            raise
        except Exception as e:
            # network or other error -> retry
            if attempt == max_retries:
                raise
            await asyncio.sleep(backoff)
            backoff *= 2
        finally:
            OPENAI_SEMAPHORE.release()
    raise RuntimeError("OpenAI failed after retries")

async def ask_openai_system(user_topic: str, user_explanation: str) -> str:
    system_prompt = (
        "You are an expert Web3 security mentor and senior auditor.
"
        "When given a user's explanation of a topic, do the following precisely:
"
        "1) Evaluate the explanation briefly, list mistakes if any.
"
        "2) Provide a corrected, concise explanation with important details.
"
        "3) Provide three small Solidity coding examples the learner can run to recall the topic (label Example 1/2/3).
"
        "4) Provide one Foundry-style test skeleton to validate a key invariant related to the topic.
"
        "5) Provide one exploit sketch / step-by-step POC (high-level) showing how an attacker would abuse the issue.
"
        "6) Provide 2-3 reputable blog/article links discussing the attack vector. Keep links short.
"
        "7) Provide a 5-question micro-quiz (questions only, no answers).
"
        "Respond in clear Markdown sections with headers: Evaluation, Corrected Explanation, Examples, Foundry Test, Exploit Sketch, Links, Quiz. Keep code blocks short and runnable. Be concise and precise.
"
    )

    user_prompt = f"Topic: {user_topic}

User explanation:
{user_explanation}

Follow the instructions in the system prompt and produce the sections."
    return await openai_chat_with_retries(system_prompt, user_prompt)

# --- DB helpers ---

def get_user(chat_id: int):
    cur.execute("SELECT chat_id, state, temp, last_topic FROM users WHERE chat_id=?", (chat_id,))
    row = cur.fetchone()
    if row:
        temp = None
        if row[2]:
            try:
                temp = json.loads(row[2])
            except Exception:
                temp = None
        return {"chat_id": row[0], "state": row[1] or "idle", "temp": temp, "last_topic": row[3]}
    cur.execute("INSERT OR REPLACE INTO users (chat_id, state, temp, last_topic) VALUES (?,?,?,?)", (chat_id, "idle", None, None))
    conn.commit()
    return {"chat_id": chat_id, "state": "idle", "temp": None, "last_topic": None}


def update_user(chat_id: int, state: Optional[str] = None, temp: Optional[dict] = None, last_topic: Optional[str] = None):
    user = get_user(chat_id)
    new_state = state if state is not None else user["state"]
    new_temp = json.dumps(temp) if temp is not None else (json.dumps(user["temp"]) if user["temp"] is not None else None)
    new_topic = last_topic if last_topic is not None else user["last_topic"]
    cur.execute("UPDATE users SET state=?, temp=?, last_topic=?, updated_at=CURRENT_TIMESTAMP WHERE chat_id=?", (new_state, new_temp, new_topic, chat_id))
    conn.commit()


def add_audit(chat_id: int, name: str, urls: str, status: str, issues: str, missed: str, notes: str):
    cur.execute("INSERT INTO audits (chat_id, name, urls, status, issues_found, missed_issues, notes) VALUES (?,?,?,?,?,?,?)",
                (chat_id, name, urls, status, issues, missed, notes))
    conn.commit()


def get_audits(chat_id: int):
    cur.execute("SELECT id, name, urls, status, issues_found, missed_issues, notes, created_at FROM audits WHERE chat_id=? ORDER BY created_at DESC", (chat_id,))
    return cur.fetchall()

# --- message processing ---

@app.get("/health")
async def health():
    return {"ok": True, "db": os.path.exists(DB_PATH)}

@app.post("/webhook")
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    payload = await request.json()
    if "message" not in payload:
        # ignore non-message updates
        return {"ok": True}

    message = payload["message"]
    chat_id = message["chat"]["id"]
    text = message.get("text") or ""
    text = text.strip()

    user = get_user(chat_id)

    # commands
    lower = text.lower()
    if lower.startswith("/start"):
        await telegram_send_message(chat_id, "I am your Web3 Security Mentor Bot.
Are you here today for your security practice? (reply 'yes' or 'no')")
        update_user(chat_id, state="awaiting_presence", temp=None)
        return {"ok": True}

    if lower in ("/help", "help"):
        help_msg = (
            "Commands:
/start - begin practice session
/add audit - add an audit to tracker
/show audits - view audits
/reset - reset conversation state
/help - show this help
"
        )
        await telegram_send_message(chat_id, help_msg)
        return {"ok": True}

    if lower in ("/reset", "reset"):
        update_user(chat_id, state="idle", temp=None, last_topic=None)
        await telegram_send_message(chat_id, "State reset. Send /start to begin a new session.")
        return {"ok": True}

    # show audits
    if lower in ("show audits", "show audit tracker", "show audit"):
        rows = get_audits(chat_id)
        if not rows:
            await telegram_send_message(chat_id, "No audits tracked yet. Use 'add audit' to create one.")
            return {"ok": True}
        msg = "*Your Audit Tracker*
"
        for r in rows:
            msg += f"
*{r[1]}* (id:{r[0]})
URLs: {r[2]}
Status: {r[3]}
Issues: {r[4]}
Missed: {r[5]}
Notes: {r[6]}
"
        await telegram_send_message(chat_id, msg)
        return {"ok": True}

    # add audit initiation
    if lower.startswith("add audit") or lower == "add audit":
        await telegram_send_message(chat_id, "Okay — let's add a new audit. What is the *Audit Name*? (send name)")
        update_user(chat_id, state="adding_audit", temp={})
        return {"ok": True}

    # flow: adding audit (multi-step)
    if user["state"] == "adding_audit":
        temp = user["temp"] or {}
        # determine next missing field
        if "name" not in temp:
            temp["name"] = text
            update_user(chat_id, state="adding_audit", temp=temp)
            await telegram_send_message(chat_id, "Send Contract URL(s) (comma separated)")
            return {"ok": True}
        if "urls" not in temp:
            temp["urls"] = text
            update_user(chat_id, state="adding_audit", temp=temp)
            await telegram_send_message(chat_id, "Status (e.g., in-progress, completed)")
            return {"ok": True}
        if "status" not in temp:
            temp["status"] = text
            update_user(chat_id, state="adding_audit", temp=temp)
            await telegram_send_message(chat_id, "Issues found (short summary)")
            return {"ok": True}
        if "issues" not in temp:
            temp["issues"] = text
            update_user(chat_id, state="adding_audit", temp=temp)
            await telegram_send_message(chat_id, "Post-evaluation: Missed issues (if any). If none, write 'none'.")
            return {"ok": True}
        if "missed" not in temp:
            temp["missed"] = text
            update_user(chat_id, state="adding_audit", temp=temp)
            await telegram_send_message(chat_id, "Add any notes (or type 'none')")
            return {"ok": True}
        # final notes
        temp["notes"] = text
        add_audit(chat_id, temp.get("name",""), temp.get("urls",""), temp.get("status",""), temp.get("issues",""), temp.get("missed",""), temp.get("notes",""))
        update_user(chat_id, state="idle", temp=None)
        await telegram_send_message(chat_id, f"Audit '{temp.get('name')}' added to your tracker.")
        return {"ok": True}

    # presence flow
    if user["state"] == "awaiting_presence":
        if lower in ["yes", "y", "i am here", "here"]:
            await telegram_send_message(chat_id, "Great! What topic did you learn today? Please explain it in your own words.
(You can send multi-line explanations.)")
            update_user(chat_id, state="awaiting_explanation", temp=None)
            return {"ok": True}
        else:
            await telegram_send_message(chat_id, "No worries. What happened today? If you'd like a short revision quiz instead, type 'revision'.")
            update_user(chat_id, state="idle", temp=None)
            return {"ok": True}

    # explanation flow: accept ANY text as explanation while in awaiting_explanation
    if user["state"] == "awaiting_explanation":
        user_text = text
        # attempt to auto-detect a short topic from first line
        first_line = user_text.splitlines()[0].strip() if user_text else ""
        topic_guess = first_line if first_line and len(first_line.split()) <= 6 else "general smart contract topic"
        update_user(chat_id, state="processing_explanation", temp=None, last_topic=topic_guess)
        # call OpenAI in background
        background_tasks.add_task(process_explanation_background, chat_id, topic_guess, user_text)
        await telegram_send_message(chat_id, "Thanks — evaluating your explanation and preparing examples. I'll send detailed feedback shortly.")
        return {"ok": True}

    # default fallback
    await telegram_send_message(chat_id, "I didn't understand that. Send /start to begin a practice session or 'add audit' to track an audit.")
    return {"ok": True}


async def process_explanation_background(chat_id: int, topic: str, user_explanation: str):
    try:
        res = await ask_openai_system(topic, user_explanation)
    except Exception as e:
        # If OpenAI fails, put user back to awaiting_explanation so they can retry
        update_user(chat_id, state="awaiting_explanation", temp=None)
        await telegram_send_message(chat_id, f"OpenAI call failed: {e}
Please try again in a moment. You can paste your explanation again or shorten it.")
        return
    # send response in safe-sized chunks
    for chunk in split_text_chunks(res, 3500):
        await telegram_send_message(chat_id, chunk)
    # after sending full response, move user to idle and store last_topic
    update_user(chat_id, state="idle", temp=None, last_topic=topic)


def split_text_chunks(text: str, max_len: int = 3500):
    parts = []
    while text:
        if len(text) <= max_len:
            parts.append(text)
            break
        idx = text.rfind("
", 0, max_len)
        if idx == -1:
            idx = max_len
        parts.append(text[:idx])
        text = text[idx:].lstrip()
    return parts


@app.get("/set_webhook")
async def route_set_webhook():
    if not BASE_URL:
        return {"error": "Set BASE_URL environment variable to your public URL (render/ngrok/railway)."}
    try:
        result = await set_webhook()
        return result
    except Exception as e:
        return {"error": str(e)}


# run with: uvicorn main:app --reload

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

