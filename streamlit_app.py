#!/usr/bin/env python3
"""
OpenRouter Streamlit Chat — Full Edition (Redesigned UI - Polished & Error Handled)
"""

# ───────────────────────── Imports ─────────────────────────
import json
import logging # Keep this basic for now
import os
import sys
import subprocess
import time
import requests
import re
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo
import streamlit as st

# ─────────────────────────── Logging Setup (Simplified for Cloud) ─────────────────
# Basic configuration at the top level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s:%(lineno)d | %(message)s", # Added module and lineno
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True # Try to force override any other configurations
)
logger = logging.getLogger(__name__)


# ────────────────────────── Configuration ───────────────────
OPENROUTER_API_KEY  = "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" # Replace
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT     = 120

FALLBACK_MODEL_ID = "deepseek/deepseek-chat-v3-0324:free"
FALLBACK_MODEL_KEY = "_FALLBACK_"
FALLBACK_MODEL_EMOJI = "🆓"
FALLBACK_MODEL_MAX_TOKENS = 8000

MODEL_MAP = {
    "A": "google/gemini-2.5-pro-preview",
    "B": "openai/o4-mini",
    "C": "openai/chatgpt-4o-latest",
    "D": "deepseek/deepseek-r1",
    "F": "google/gemini-2.5-flash-preview"
}
ROUTER_MODEL_ID = "mistralai/mistral-7b-instruct:free"

MAX_TOKENS = {"A": 16_000, "B": 8_000, "C": 16_000, "D": 8_000, "F": 8_000}
PLAN = { # (Daily, Weekly, Monthly)
    "A": (10, 70, 300), "B": (5, 35, 150), "C": (1, 7, 30),
    "D": (4, 28, 120), "F": (180, 500, 2000)
}
EMOJI = {"A": "🌟", "B": "🔷", "C": "🟥", "D": "🟢", "F": "🌀"}
MODEL_DESCRIPTIONS = {
    "A": "🌟 (gemini-2.5-pro-preview) – Top-quality, creative, expensive.",
    "B": "🔷 (o4-mini) – Mid-stakes reasoning, cost-effective.",
    "C": "🟥 (chatgpt-4o-latest) – Polished/empathetic, pricier.",
    "D": "🟢 (deepseek-r1) – Cheap factual reasoning.",
    "F": "🌀 (gemini-2.5-flash-preview) – Quick, general purpose."
}

TZ = ZoneInfo("Australia/Sydney")
DATA_DIR   = Path(__file__).parent
SESS_FILE  = DATA_DIR / "chat_sessions.json"
QUOTA_FILE = DATA_DIR / "quotas.json"

# ──────────────────────── Helper Functions ───────────────────────
def _load(path: Path, default):
    try: return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError): return default
def _save(path: Path, obj): path.write_text(json.dumps(obj, indent=2))
def _today(): return date.today().isoformat()
def _yweek(): return datetime.now(TZ).strftime("%G-%V")
def _ymonth(): return datetime.now(TZ).strftime("%Y-%m")

# ───────────────────── Quota Management ────────────────────────
def _reset(block: dict, key: str, stamp: str, zeros: dict):
    active_zeros = {k: 0 for k in MODEL_MAP}
    if block.get(key) != stamp:
        block[key] = stamp
        block[f"{key}_u"] = active_zeros.copy()

def _load_quota():
    zeros = {k: 0 for k in MODEL_MAP}
    q = _load(QUOTA_FILE, {})
    for period_usage_key in ("d_u", "w_u", "m_u"):
        if period_usage_key in q:
            current_usage_dict = q[period_usage_key]
            keys_to_remove = [k for k in current_usage_dict if k not in MODEL_MAP]
            for k_rem in keys_to_remove:
                del current_usage_dict[k_rem]
                logger.info(f"Removed old model key '{k_rem}' from quota usage '{period_usage_key}'.")
    _reset(q, "d", _today(), zeros); _reset(q, "w", _yweek(), zeros); _reset(q, "m", _ymonth(), zeros)
    _save(QUOTA_FILE, q); return q
quota = _load_quota()

def remaining(key: str):
    ud = quota.get("d_u", {}).get(key, 0); uw = quota.get("w_u", {}).get(key, 0); um = quota.get("m_u", {}).get(key, 0)
    if key not in PLAN: logger.error(f"Unknown key for remaining: {key}"); return 0,0,0
    ld, lw, lm = PLAN[key]; return ld - ud, lw - uw, lm - um

def record_use(key: str):
    if key not in MODEL_MAP: logger.warning(f"Unknown model key for record_use: {key}"); return
    for blk_key in ("d_u", "w_u", "m_u"):
        if blk_key not in quota: quota[blk_key] = {k: 0 for k in MODEL_MAP}
        quota[blk_key][key] = quota[blk_key].get(key, 0) + 1
    _save(QUOTA_FILE, quota)

# ───────────────────── Session Management ───────────────────────
def _delete_unused_blank_sessions(keep_sid: str = None):
    sids_to_delete = [sid for sid, data in sessions.items() if sid != keep_sid and data.get("title") == "New chat" and not data.get("messages")]
    if sids_to_delete:
        for sid_del in sids_to_delete: logger.info(f"Auto-deleting blank session: {sid_del}"); del sessions[sid_del]
        return True
    return False
sessions = _load(SESS_FILE, {})
def _new_sid():
    _delete_unused_blank_sessions(keep_sid=None)
    sid = str(int(time.time() * 1000))
    sessions[sid] = {"title": "New chat", "messages": []}; return sid
def _autoname(seed: str) -> str:
    words = seed.strip().split(); cand = " ".join(words[:3]) or "Chat"
    return (cand[:25] + "…") if len(cand) > 25 else cand

# ────────────────────────── API Calls ──────────────────────────
def api_post(payload: dict, *, stream: bool=False, timeout: int=DEFAULT_TIMEOUT):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type":  "application/json"}
    logger.info(f"POST /chat/completions → model={payload.get('model')}, stream={stream}, max_tokens={payload.get('max_tokens')}")
    return requests.post(f"{OPENROUTER_API_BASE}/chat/completions", headers=headers, json=payload, stream=stream, timeout=timeout)

def streamed(model: str, messages: list, max_tokens_out: int):
    payload = {"model": model, "messages": messages, "stream": True, "max_tokens": max_tokens_out}
    with api_post(payload, stream=True) as r:
        try: r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            text = r.text; logger.error(f"Stream HTTPError {e.response.status_code}: {text}"); yield None, f"HTTP {e.response.status_code}: {text}"; return
        for line in r.iter_lines():
            if not line: continue
            line_str = line.decode("utf-8")
            if line_str.startswith(": OPENROUTER PROCESSING"):
                logger.info(f"OpenRouter PING: {line_str.strip()}")
                continue
            if not line_str.startswith("data: "):
                logger.warning(f"Unexpected non-event-stream line (decoded): {line_str.strip()}")
                continue
            data = line_str[6:].strip()
            if data == "[DONE]": break
            try: chunk = json.loads(data)
            except json.JSONDecodeError: logger.error(f"Bad JSON chunk: {data}"); yield None, "Error decoding response chunk"; return

            if "error" in chunk:
                msg_obj = chunk["error"]
                msg = "Unknown API error in stream chunk"
                if isinstance(msg_obj, dict) and "message" in msg_obj: msg = msg_obj["message"]
                logger.error(f"API stream chunk error: {msg}"); yield None, msg; return

            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            if delta is not None: yield delta, None

# ───────────────────────── Model Routing ───────────────────────
def route_choice(user_msg: str, allowed: list[str]) -> str:
    if not allowed:
        logger.warning("Router: No models allowed, defaulting to F or first available.")
        return "F" if "F" in MODEL_MAP else (list(MODEL_MAP.keys())[0] if MODEL_MAP else "F")
    if len(allowed) == 1:
        logger.info(f"Router: Only one model allowed ({allowed[0]}), selecting it directly.")
        return allowed[0]

    system_lines = ["You are an intelligent model-routing assistant.", "Select ONLY one letter from the following available models:"]
    for k in allowed:
        if k in MODEL_DESCRIPTIONS:
            desc_for_router = MODEL_DESCRIPTIONS[k].split('–')[1].strip() if '–' in MODEL_DESCRIPTIONS[k] else MODEL_DESCRIPTIONS[k]
            system_lines.append(f"- {k}: {MODEL_MAP[k].split('/')[-1]} ({desc_for_router})")
        else:
            system_lines.append(f"- {k}: {MODEL_MAP[k].split('/')[-1]}")

    system_lines.extend(["Based on the user's query, choose the letter that best balances quality, speed, and cost-sensitivity.", "Respond with ONLY the single capital letter. No extra text."])

    router_messages = [{"role": "system", "content": "\n".join(system_lines)}, {"role": "user", "content": user_msg}]
    payload_r = {"model": ROUTER_MODEL_ID, "messages": router_messages, "max_tokens": 10, "temperature": 0.1}

    try:
        r = api_post(payload_r); r.raise_for_status()
        r_json = r.json()

        if "error" in r_json:
            logger.error(f"Router API returned an error object: {r_json['error']}")
        elif "choices" not in r_json or not r_json["choices"] or "message" not in r_json["choices"][0] or "content" not in r_json["choices"][0]["message"]:
            logger.error(f"Router API response malformed: {r_json}")
        else:
            raw_text = r_json["choices"][0]["message"]["content"].strip().upper()
            logger.info(f"Router raw response: '{raw_text}'")

            for letter_allowed in sorted(allowed):
                if re.search(rf"\b{re.escape(letter_allowed)}\b", raw_text):
                    logger.info(f"Router selected model: '{letter_allowed}' (standalone regex match).")
                    return letter_allowed
            for char_code in raw_text:
                if char_code in allowed:
                    logger.info(f"Router selected model: '{char_code}' (first character match).")
                    return char_code
            logger.warning(f"Router response '{raw_text}' did not contain an identifiable allowed model from {allowed}. Falling back.")
    except requests.exceptions.RequestException as e: logger.error(f"Router API call failed (RequestException): {e}")
    except json.JSONDecodeError as e: logger.error(f"Router API response not valid JSON: {e}")
    except Exception as e: logger.error(f"Unexpected error during router call: {e}")

    fallback_choice = "F" if "F" in allowed else allowed[0]
    logger.warning(f"Router falling back to model: {fallback_choice}"); return fallback_choice

# ───────────────────── Credits Endpoint ───────────────────────
def get_credits():
    try:
        r = requests.get(f"{OPENROUTER_API_BASE}/credits", headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}, timeout=10)
        r.raise_for_status(); d = r.json()["data"]; return d["total_credits"], d["total_usage"], d["total_credits"] - d["total_usage"]
    except Exception as e: logger.warning(f"Could not fetch /credits: {e}"); return None, None, None

# ───────────────────────── UI Styling ──────────────────────────
def load_custom_css():
    css = """
    <style>
        /* General Styles */
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
        }
        [data-testid="stAppViewContainer"] > .main {
            background-color: #171923; /* Main chat area background */
        }
        html[data-theme="light"] [data-testid="stAppViewContainer"] > .main {
            background-color: #FFFFFF;
        }
        [data-testid="stAppViewContainer"] > .main > .block-container {
            padding-top: 2rem;
            max-width: 860px;
            padding-bottom: calc(2rem + 80px); /* Adjust if chat input height changes significantly */
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #1A202C;
            padding: 1.25rem 1rem;
            border-right: 1px solid #2D3748;
        }
        html[data-theme="light"] [data-testid="stSidebar"] {
            background-color: #F7FAFC;
            border-right: 1px solid #E2E8F0;
        }

        [data-testid="stSidebar"] > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) { /* Sidebar Header container */
            display: flex !important; align-items: center !important; gap: 10px;
            margin-bottom: 0 !important;
            padding-bottom: 0;
            border-bottom: none;
        }
        [data-testid="stSidebar"] .stImage > img { /* Logo in sidebar */
            border-radius: 6px; width: 38px !important; height: 38px !important;
        }
        [data-testid="stSidebar"] h1 { /* "OpenRouter Chat" title in sidebar */
            font-size: 1.3rem !important; color: #E2E8F0;
            font-weight: 600; margin-bottom: 0;
        }
        html[data-theme="light"] [data-testid="stSidebar"] h1 { color: #2D3748; }

        [data-testid="stSidebar"] .stButton > button[kind="primary"] { /* e.g. New Chat button */
             font-weight: 500;
        }

        [data-testid="stSidebar"] h3 { /* Styling for st.subheader like "CHATS" */
            font-size: 0.7rem !important; text-transform: uppercase;
            font-weight: 600; color: #A0AEC0;
            margin-top: 1.5rem; margin-bottom: 0.75rem; letter-spacing: 0.05em;
            padding-left: 0.1rem;
        }
        html[data-theme="light"] [data-testid="stSidebar"] h3 { color: #718096; }

        [data-testid="stSidebar"] .stExpander {
            border: none !important;
            margin-left: -0.5rem;
            margin-right: -0.5rem;
        }
        [data-testid="stSidebar"] .stExpander header {
            padding: 0.75rem 0.5rem !important;
            font-size: 0.7rem !important; text-transform: uppercase;
            font-weight: 600; color: #A0AEC0;
            letter-spacing: 0.05em;
            border-bottom: none !important;
            background-color: transparent !important;
        }
        html[data-theme="light"] [data-testid="stSidebar"] .stExpander header { color: #718096; }
        [data-testid="stSidebar"] .stExpander header:hover {
            background-color: rgba(255,255,255,0.05) !important;
        }
        html[data-theme="light"] [data-testid="stSidebar"] .stExpander header:hover {
            background-color: rgba(0,0,0,0.03) !important;
        }
        [data-testid="stSidebar"] .stExpander div[data-testid="stExpanderDetails"] {
            padding: 0rem 0.5rem 0.75rem 0.5rem;
        }

        .model-usage-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.3rem 0.1rem;
            margin-bottom: 0.1rem;
            font-size: 0.85rem;
        }
        .model-info { display: flex; align-items: center; gap: 7px; }
        .model-emoji { font-size: 1rem; }
        .model-key-name { color: #CBD5E0; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 150px;}
        html[data-theme="light"] .model-key-name { color: #4A5568; }
        .quota-text {
            font-weight: 500; color: #A0AEC0; font-size: 0.8rem;
            background-color: #2D3748;
            padding: 2px 6px;
            border-radius: 4px;
        }
        html[data-theme="light"] .quota-text { color: #4A5568; background-color: #E2E8F0; }

        .progress-bar-container {
            height: 6px;
            background-color: #2D3748;
            border-radius: 3px;
            margin-bottom: 0.4rem;
            overflow: hidden;
        }
        html[data-theme="light"] .progress-bar-container { background-color: #E2E8F0; }
        .progress-bar-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.3s ease-in-out, background-color 0.3s ease-in-out;
        }

        [data-testid="stSidebar"] button[data-testid*="stPopover"] {
            font-size: 0.75rem !important;
            color: #718096 !important;
            padding: 0.1rem 0.4rem !important;
            margin-top: -0.1rem !important;
            background: transparent !important;
            border: 1px solid #4A5568 !important;
            border-radius: 4px !important;
            line-height: 1.2 !important;
            min-height: auto !important;
            width: 100%;
        }
        [data-testid="stSidebar"] button[data-testid*="stPopover"]:hover {
            color: var(--primary-color) !important;
            border-color: var(--primary-color) !important;
            background-color: color-mix(in srgb, var(--primary-color) 10%, transparent) !important;
        }

        [data-testid="stSidebar"] .stButton > button:not([kind="primary"]) {
            border-radius: 6px; border: none;
            padding: 0.6rem 0.75rem; font-size: 0.875rem; font-weight: 400;
            background-color: transparent; color: #CBD5E0;
            transition: background-color 0.2s, color 0.2s, border-left-color 0.2s;
            width: 100%; margin-bottom: 0.2rem;
            display: flex; align-items: center; gap: 8px;
            border-left: 3px solid transparent;
            justify-content: flex-start !important;
            text-align: left !important;
        }
        html[data-theme="light"] [data-testid="stSidebar"] .stButton > button:not([kind="primary"]) { color: #4A5568; }

        [data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover {
            background-color: #2D3748; color: #F7FAFC;
            border-left-color: #4A5568;
        }
        html[data-theme="light"] [data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover {
            background-color: #E2E8F0; color: var(--primary-color);
            border-left-color: #CBD5E0;
        }

        [data-testid="stSidebar"] .stButton > button:not([kind="primary"]):has(span:contains("🔹")) {
            color: var(--primary-color) !important;
            background-color: color-mix(in srgb, var(--primary-color) 10%, transparent);
            border-left: 3px solid var(--primary-color);
            font-weight: 500;
            justify-content: flex-start !important;
            text-align: left !important;
        }

        [data-testid="stSidebar"] .stCaption {
            color: #718096; font-size: 0.8rem; text-align: left;
            padding: 0.2rem 0.1rem 1rem 0.1rem; line-height: 1.4;
        }
        html[data-theme="light"] [data-testid="stSidebar"] .stCaption { color: #6B7280; }

        [data-testid="stSidebar"] hr {
            margin: 1.25rem -1rem;
            border: 0;
            border-top: 1px solid #2D3748;
        }
        html[data-theme="light"] [data-testid="stSidebar"] hr { border-top-color: #E2E8F0; }

        .empty-chat-container {
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            min-height: 65vh; text-align: center; padding: 2rem;
        }
        .empty-chat-container img.logo-main {
            width: 72px; height: 72px; border-radius: 12px; margin-bottom: 1.75rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        }
        .empty-chat-container h2 {
            font-size: 1.75rem; font-weight: 600; margin-bottom: 0.7rem; color: var(--text-color);
        }
        .empty-chat-container p {
            font-size: 1rem; color: var(--text-color-secondary); max-width: 450px; line-height: 1.6;
        }

        /* --- REMOVED OLD CHAT INPUT STYLES ---
        [data-testid="stChatInput"] {
            background-color: #1A202C !important;
            border: 1px solid #2D3748 !important;
            padding: 0.75rem 1rem !important;
            position: sticky; bottom: 0; left:0; right:0;
            z-index: 100;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        html[data-theme="light"] [data-testid="stChatInput"] {
            background-color: #F0F2F6 !important;
            border-color: #D1D7E0 !important;
        }

        [data-testid="stChatInput"] textarea {
            flex-grow: 1;
            background-color: #2D3748 !important;
            border: 1px solid #4A5568 !important;
            color: #E2E8F0 !important;
            padding: 10px 14px !important;
            line-height: 1.5 !important;
            box-shadow: none !important;
            margin: 0 !important;
            border-radius: 8px !important;
            outline: none !important;
            transition: border-color 0.2s, box-shadow 0.2s;
            min-height: 40px;
            max-height: 200px;
            resize: none;
        }
        html[data-theme="light"] [data-testid="stChatInput"] textarea {
            background-color: #FFFFFF !important;
            border-color: #CBD5E0 !important;
            color: #2D3748 !important;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color: #718096;
        }
        html[data-theme="light"] [data-testid="stChatInput"] textarea::placeholder {
            color: #A0AEC0;
        }
        [data-testid="stChatInput"] textarea:focus {
             border-color: var(--primary-color) !important;
             box-shadow: 0 0 0 2px color-mix(in srgb, var(--primary-color) 25%, transparent) !important;
        }

        [data-testid="stChatInput"] button {
             height: 40px;
             border-radius: 8px !important;
        }
        [data-testid="stChatInput"] button svg { fill: #A0AEC0; }
        [data-testid="stChatInput"] button:hover svg { fill: var(--primary-color); }
        [data-testid="stChatInput"] button:disabled svg { fill: #4A5568; }
        --- END REMOVED OLD CHAT INPUT STYLES --- */


        /* +++ NEW CHAT INPUT STYLES +++ */
        [data-testid="stChatInput"] {
            background-color: #2D3748 !important; /* Darker gray for the bar background in dark mode */
            border: 1px solid #4A5568 !important; /* Slightly lighter gray border for the bar in dark mode */
            border-radius: 10px !important; /* Rounded corners for the bar */
            padding: 0.5rem 0.75rem !important;
            margin: 0.5rem 1rem 1rem 1rem !important; /* Margin around the bar */
            position: sticky !important; /* Ensure it's sticky */
            bottom: 1rem !important; /* Stick to bottom with some space from viewport edge */
            left: 0 !important; /* Ensure it spans */
            right: 0 !important; /* Ensure it spans */
            width: calc(100% - 2rem) !important; /* Adjust width considering margin */
            max-width: calc(860px - 2rem); /* Align with main content max-width considering margin */
            margin-left: auto !important; /* Center it if block-container is centered */
            margin-right: auto !important; /* Center it if block-container is centered */
            z-index: 100 !important;
            display: flex !important;
            align-items: center !important;
            gap: 0.75rem !important; /* Increased gap */
            box-shadow: 0 -2px 10px rgba(0,0,0,0.2) !important;
        }

        html[data-theme="light"] [data-testid="stChatInput"] {
            background-color: #F3F4F6 !important; /* Light gray for bar background */
            border-color: #D1D5DB !important; /* Medium gray border for bar */
            box-shadow: 0 -2px 10px rgba(0,0,0,0.05) !important;
        }

        [data-testid="stChatInput"] textarea {
            flex-grow: 1 !important;
            background-color: #1F2937 !important; /* Even darker gray for textarea in dark mode */
            border: 1px solid #374151 !important; /* Darker border for textarea in dark mode */
            color: #E5E7EB !important; /* Light text color */
            padding: 10px 14px !important;
            line-height: 1.5 !important;
            box-shadow: none !important;
            margin: 0 !important;
            border-radius: 8px !important;
            outline: none !important;
            transition: border-color 0.2s, box-shadow 0.2s;
            min-height: 42px !important; /* Standard height */
            max-height: 200px !important;
            resize: none !important;
        }

        html[data-theme="light"] [data-testid="stChatInput"] textarea {
            background-color: #FFFFFF !important; /* White background for textarea */
            border-color: #D1D5DB !important; /* Medium gray border */
            color: #111827 !important; /* Dark text color */
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: #6B7280 !important; /* Dark mode placeholder */
        }
        html[data-theme="light"] [data-testid="stChatInput"] textarea::placeholder {
            color: #9CA3AF !important; /* Light mode placeholder */
        }

        [data-testid="stChatInput"] textarea:focus {
             border-color: var(--primary-color) !important;
             box-shadow: 0 0 0 2px color-mix(in srgb, var(--primary-color) 25%, transparent) !important;
        }

        [data-testid="stChatInput"] button {
             height: 42px !important; /* Match textarea height */
             width: 42px !important;
             min-width: 42px !important; /* Ensure it doesn't shrink too much */
             border-radius: 8px !important;
             background-color: #374151 !important; /* Dark mode button bg */
             border: none !important;
             padding: 0 !important;
             display: flex !important;
             align-items: center !important;
             justify-content: center !important;
             cursor: pointer !important;
             transition: background-color 0.2s !important;
        }
        html[data-theme="light"] [data-testid="stChatInput"] button {
            background-color: #E5E7EB !important; /* Light mode button bg */
        }

        [data-testid="stChatInput"] button svg {
            fill: #9CA3AF !important; /* Dark mode icon color */
            width: 20px !important;
            height: 20px !important;
        }
        html[data-theme="light"] [data-testid="stChatInput"] button svg {
            fill: #4B5563 !important; /* Light mode icon color */
        }

        [data-testid="stChatInput"] button:hover:not(:disabled) {
            background-color: #4B5563 !important; /* Dark mode hover */
        }
        html[data-theme="light"] [data-testid="stChatInput"] button:hover:not(:disabled) {
            background-color: #D1D5DB !important; /* Light mode hover */
        }

        [data-testid="stChatInput"] button:hover:not(:disabled) svg {
            fill: var(--primary-color) !important;
        }

        [data-testid="stChatInput"] button:disabled {
            background-color: #374151 !important;
            opacity: 0.6 !important;
            cursor: not-allowed !important;
        }
        html[data-theme="light"] [data-testid="stChatInput"] button:disabled {
            background-color: #E5E7EB !important;
            opacity: 0.6 !important;
        }
        [data-testid="stChatInput"] button:disabled svg {
            fill: #6B7280 !important; /* Dark mode disabled icon */
        }
        html[data-theme="light"] [data-testid="stChatInput"] button:disabled svg {
            fill: #9CA3AF !important; /* Light mode disabled icon */
        }
        /* +++ END NEW CHAT INPUT STYLES +++ */


        [data-testid="stChatMessage"] {
            border-radius: 10px; padding: 12px 18px; margin-bottom: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: none; max-width: 78%; line-height: 1.6;
        }
        [data-testid^="stChatMessageUser"] {
            background-color: var(--primary-color); color: white; margin-left: auto;
            border-bottom-right-radius: 4px;
        }
        [data-testid^="stChatMessageUser"] .stMarkdown p,
        [data-testid^="stChatMessageUser"] .stMarkdown li,
        [data-testid^="stChatMessageUser"] .stMarkdown ol,
        [data-testid^="stChatMessageUser"] .stMarkdown ul {
            color: white !important;
        }

        [data-testid^="stChatMessageAssistant"] {
            background-color: #2D3748; color: #E2E8F0;
            border-bottom-left-radius: 4px;
            margin-right: auto;
        }
        html[data-theme="light"] [data-testid^="stChatMessageAssistant"] {
            background-color: #E9ECF2; color: #2D3748;
        }
        [data-testid^="stChatMessageAssistant"] .stMarkdown p,
        [data-testid^="stChatMessageAssistant"] .stMarkdown li,
        [data-testid^="stChatMessageAssistant"] .stMarkdown ol,
        [data-testid^="stChatMessageAssistant"] .stMarkdown ul {
            color: inherit !important;
        }

        .stExpander {
            border: 1px solid #2D3748; border-radius: 8px; margin-bottom: 1rem;
            background-color: transparent;
        }
        html[data-theme="light"] .stExpander { border-color: #CBD5E0; }
        .stExpander header {
            font-weight: 500; font-size: 0.8rem; padding: 0.5rem 0.8rem !important;
            background-color: rgba(45, 55, 72, 0.5);
            border-bottom: 1px solid #2D3748;
            border-top-left-radius: 7px; border-top-right-radius: 7px;
            color: #A0AEC0;
        }
        html[data-theme="light"] .stExpander header { background-color: rgba(226, 232, 240, 0.5); border-color: #CBD5E0; color: #4A5568; }
        .stExpander header:hover { background-color: #2D3748; }
        html[data-theme="light"] .stExpander header:hover { background-color: #E2E8F0; }
        .stExpander div[data-testid="stExpanderDetails"] { padding: 0.75rem 1rem; background-color: transparent; }

        .main::-webkit-scrollbar { width: 8px; }
        .main::-webkit-scrollbar-track { background: transparent; }
        .main::-webkit-scrollbar-thumb {
            background-color: #4A5568;
            border-radius: 10px;
            border: 2px solid #171923;
            background-clip: content-box;
        }
        .main::-webkit-scrollbar-thumb:hover { background-color: #718096; }

        [data-testid="stSidebar"] > div:nth-child(1) {
            scrollbar-width: thin;
            scrollbar-color: #4A5568 #1A202C;
        }
        [data-testid="stSidebar"] > div:nth-child(1)::-webkit-scrollbar {
            width: 8px;
        }
        [data-testid="stSidebar"] > div:nth-child(1)::-webkit-scrollbar-track {
            background: transparent;
        }
        [data-testid="stSidebar"] > div:nth-child(1)::-webkit-scrollbar-thumb {
            background-color: #4A5568;
            border-radius: 10px;
            border: 2px solid #1A202C;
            background-clip: content-box;
        }
        [data-testid="stSidebar"] > div:nth-child(1)::-webkit-scrollbar-thumb:hover {
            background-color: #718096;
        }

        [data-testid="stHeader"] {
            background-color: #1A202C !important;
            border-bottom: 1px solid #2D3748 !important;
        }
        html[data-theme="light"] [data-testid="stHeader"] {
            background-color: #F7FAFC !important;
            border-bottom-color: #E2E8F0 !important;
        }
        [data-testid="stHeader"] [data-testid="stToolbar"] {
            padding-right: 1rem;
        }
        [data-testid="stHeader"] [data-testid="stToolbar"] button svg {
            fill: #A0AEC0;
        }
        html[data-theme="light"] [data-testid="stHeader"] [data-testid="stToolbar"] button svg {
            fill: #718096;
        }
        [data-testid="stHeader"] [data-testid="stToolbar"] button:hover svg {
            fill: var(--primary-color);
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ───────────────────────── Streamlit UI ────────────────────────
st.set_page_config(page_title="OpenRouter Chat", layout="wide", initial_sidebar_state="expanded")
load_custom_css()

# Initial SID Management
needs_save_and_rerun_on_startup = False
if "sid" not in st.session_state:
    st.session_state.sid = _new_sid()
    needs_save_and_rerun_on_startup = True
elif st.session_state.sid not in sessions:
    logger.warning(f"Session ID {st.session_state.sid} from state not found in loaded sessions. Creating a new chat.")
    st.session_state.sid = _new_sid()
    needs_save_and_rerun_on_startup = True
else:
    if _delete_unused_blank_sessions(keep_sid=st.session_state.sid):
        needs_save_and_rerun_on_startup = True

if needs_save_and_rerun_on_startup:
    _save(SESS_FILE, sessions)
    st.rerun() # CORRECTED

if "credits" not in st.session_state:
    st.session_state.credits = dict(zip(("total", "used", "remaining"), get_credits()))
    st.session_state.credits_ts = time.time()

# ───────────────────────── Sidebar ─────────────────────────────
with st.sidebar:
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://avatars.githubusercontent.com/u/130328222?s=200&v=4", width=38)
    with col2:
        st.title("OpenRouter Chat")
    st.markdown("---")

    current_session_is_truly_blank = (st.session_state.sid in sessions and
                                      sessions[st.session_state.sid].get("title") == "New chat" and
                                      not sessions[st.session_state.sid].get("messages"))

    if st.button("➕ New Chat", key="new_chat_button_top", use_container_width=True, type="primary", disabled=current_session_is_truly_blank):
        st.session_state.sid = _new_sid()
        _save(SESS_FILE, sessions)
        st.rerun() # CORRECTED

    st.markdown("---")

    with st.expander("Model Usage (Daily)", expanded=True):
        active_model_keys = sorted(MODEL_MAP.keys())
        for m_key_idx, m_key in enumerate(active_model_keys): # Use enumerate for unique popover labels if needed
            left_d, _, _ = remaining(m_key)
            lim_d, _, _  = PLAN[m_key]

            is_unlimited = lim_d > 900_000
            progress_value = 1.0 if is_unlimited else (max(0.0, left_d / lim_d if lim_d > 0 else 0.0))
            progress_color = '#4caf50'
            if not is_unlimited:
                if progress_value <= 0.25: progress_color = '#f44336'
                elif progress_value <= 0.5: progress_color = '#ffc107'

            try:
                model_display_name = MODEL_DESCRIPTIONS[m_key].split('(')[1].split(')')[0].strip()
            except IndexError:
                model_display_name = MODEL_MAP.get(m_key, "Unknown Model").split('/')[-1]

            st.markdown(f"""
            <div class="model-usage-item">
                <div class="model-info">
                    <span class="model-emoji">{EMOJI.get(m_key, "❓")}</span>
                    <span class="model-key-name" title="{m_key}: {model_display_name}">{m_key}: {model_display_name}</span>
                </div>
                <div class="model-quota">
                    <span class="quota-text">{'∞' if is_unlimited else f'{left_d}/{lim_d}'}</span>
                </div>
            </div>
            <div class="progress-bar-container">
                <div class="progress-bar-fill" style="width: {progress_value*100}%; background-color: {progress_color};">
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.popover(f"Details: {m_key} ({m_key_idx})", use_container_width=True): # Added index to label for uniqueness
                st.markdown(f"**{MODEL_DESCRIPTIONS.get(m_key, 'No description available.')}**")
                st.markdown(f"**Model ID:** `{MODEL_MAP.get(m_key, 'N/A')}`")
                st.markdown(f"**Max Output Tokens:** {MAX_TOKENS.get(m_key, 'N/A'):,}")
                _, left_w, left_m = remaining(m_key)
                _, lim_w, lim_m = PLAN[m_key]
                st.caption(f"Daily: {left_d}/{lim_d} | Weekly: {left_w}/{lim_w} | Monthly: {left_m}/{lim_m}")

    st.markdown("---")

    if current_session_is_truly_blank and not st.session_state.get("new_chat_button_top_clicked_once", False):
         st.caption("Current chat is empty. Add a message or switch.")

    st.subheader("Chats")
    sorted_sids = sorted(sessions.keys(), key=lambda s: int(s), reverse=True)
    for sid_key_loop in sorted_sids:
        title = sessions[sid_key_loop].get("title", "Untitled")
        display_title_text = title[:28] + ("…" if len(title) > 28 else "")

        button_label = display_title_text
        if st.session_state.sid == sid_key_loop:
            button_label = f"🔹 {display_title_text}"

        if st.button(button_label, key=f"session_button_{sid_key_loop}", use_container_width=True):
            if st.session_state.sid != sid_key_loop:
                st.session_state.sid = sid_key_loop
                if _delete_unused_blank_sessions(keep_sid=sid_key_loop):
                    _save(SESS_FILE, sessions)
                st.rerun() # CORRECTED
    st.markdown("---")

    st.caption(f"Routing via: {ROUTER_MODEL_ID.split('/')[-1]}")

    tot, used, rem = (st.session_state.credits.get(k) for k in ("total","used","remaining"))
    with st.expander("Account Credits", expanded=False):
        if st.button("Refresh Credits", key="refresh_credits_button", use_container_width=True):
            st.session_state.credits = dict(zip(("total","used","remaining"), get_credits()))
            st.session_state.credits_ts = time.time()
            st.rerun() # CORRECTED
        if tot is None: st.warning("Could not fetch credits.")
        else:
            st.markdown(f"**Purchased:** ${tot:.2f} cr\n\n**Used:** ${used:.2f} cr\n\n**Remaining:** ${rem:.2f} cr")
            try: st.caption(f"Last updated: {datetime.fromtimestamp(st.session_state.credits_ts, TZ).strftime('%-d %b %Y, %H:%M:%S')}")
            except TypeError: st.caption("Last updated: N/A")

# ────────────────────────── Main Chat Panel ─────────────────────
current_sid = st.session_state.sid
if current_sid not in sessions:
    logger.error("Chat session error. Creating new.") # Use logger
    current_sid = _new_sid()
    st.session_state.sid = current_sid
    _save(SESS_FILE, sessions)
    st.rerun() # CORRECTED

chat_history = sessions[current_sid]["messages"]
is_new_empty_chat = not chat_history and sessions[current_sid]["title"] == "New chat"

if is_new_empty_chat:
    st.markdown(f"""<div class="empty-chat-container">
        <img src="https://avatars.githubusercontent.com/u/130328222?s=200&v=4" class="logo-main">
        <h2>How can I help you today, Asher?</h2>
        <p>I can help you choose the best model for your task based on its capabilities and your remaining quotas. Just type your query below!</p>
    </div>""", unsafe_allow_html=True)
else:
    for msg_idx, msg in enumerate(chat_history):
        role = msg["role"]; avatar = "👤"
        if role == "assistant":
            model_key_hist = msg.get("model")
            avatar = FALLBACK_MODEL_EMOJI if model_key_hist == FALLBACK_MODEL_KEY else EMOJI.get(model_key_hist, EMOJI.get("F", "🤖"))
        with st.chat_message(role, avatar=avatar): st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything…", key=f"chat_input_{current_sid}"):
    if current_session_is_truly_blank:
        st.session_state.new_chat_button_top_clicked_once = True

    chat_history.append({"role":"user","content":prompt})

    if not is_new_empty_chat:
        with st.chat_message("user", avatar="👤"): st.markdown(prompt)

    allowed_standard_models = [k for k in MODEL_MAP if remaining(k)[0] > 0]
    chosen_model_key_for_response = FALLBACK_MODEL_KEY
    model_id_to_use = FALLBACK_MODEL_ID
    max_tokens_api = FALLBACK_MODEL_MAX_TOKENS
    avatar_resp = FALLBACK_MODEL_EMOJI
    use_fallback = not allowed_standard_models

    if not allowed_standard_models:
        logger.info("No standard models have daily quota remaining. Using fallback model directly.")
        use_fallback = True
    else:
        chosen_model_key_for_response = route_choice(prompt, allowed_standard_models)
        logger.info(f"Router chose model key: '{chosen_model_key_for_response}' for current response.")

        if chosen_model_key_for_response in MODEL_MAP:
            model_id_to_use = MODEL_MAP[chosen_model_key_for_response]
            max_tokens_api = MAX_TOKENS[chosen_model_key_for_response]
            avatar_resp = EMOJI[chosen_model_key_for_response]
            use_fallback = False
        else:
            logger.warning(f"Router returned invalid key '{chosen_model_key_for_response}' or it's not in MODEL_MAP. Forcing fallback.")
            use_fallback = True
            chosen_model_key_for_response = FALLBACK_MODEL_KEY
            model_id_to_use = FALLBACK_MODEL_ID
            max_tokens_api = FALLBACK_MODEL_MAX_TOKENS
            avatar_resp = FALLBACK_MODEL_EMOJI

    if use_fallback:
        st.info(f"{FALLBACK_MODEL_EMOJI} Using fallback model: {FALLBACK_MODEL_ID.split('/')[-1]}")
        logger.info(f"Final decision: Using fallback model: {FALLBACK_MODEL_ID}")
        chosen_model_key_for_response = FALLBACK_MODEL_KEY
        model_id_to_use = FALLBACK_MODEL_ID
        max_tokens_api = FALLBACK_MODEL_MAX_TOKENS
        avatar_resp = FALLBACK_MODEL_EMOJI


    response_content, api_ok = "", True

    if not is_new_empty_chat:
        with st.chat_message("assistant", avatar=avatar_resp):
            placeholder = st.empty()
            for chunk, err_msg in streamed(model_id_to_use, chat_history, max_tokens_api):
                if err_msg: response_content = f"❗ **API Error**: {err_msg}"; placeholder.error(response_content); api_ok=False; break
                if chunk: response_content += chunk; placeholder.markdown(response_content + "▌")
            if api_ok: placeholder.markdown(response_content)
    else: # This block handles the case where it's a new, empty chat and a prompt is entered
        # We need to display the user's first message, then the assistant's response
        st.chat_message("user", avatar="👤").markdown(prompt) # Display the user's prompt
        with st.chat_message("assistant", avatar=avatar_resp): # Then stream the assistant's response
            placeholder = st.empty()
            for chunk, err_msg in streamed(model_id_to_use, chat_history, max_tokens_api): # Pass full chat history
                if err_msg: response_content = f"❗ **API Error**: {err_msg}"; placeholder.error(response_content); api_ok=False; break
                if chunk: response_content += chunk; placeholder.markdown(response_content + "▌")
            if api_ok: placeholder.markdown(response_content)


    chat_history.append({"role":"assistant","content":response_content,"model": chosen_model_key_for_response})
    if api_ok and not use_fallback and chosen_model_key_for_response != FALLBACK_MODEL_KEY:
        record_use(chosen_model_key_for_response)

    if sessions[current_sid]["title"] == "New chat" and len(chat_history) >=2 :
        sessions[current_sid]["title"] = _autoname(prompt)
        _delete_unused_blank_sessions(keep_sid=current_sid)

    _save(SESS_FILE, sessions)
    st.rerun() # CORRECTED

# ───────────────────────── Self-Relaunch (Local Development Only) ──────────────────────
# if __name__ == "__main__" and os.getenv("_IS_STRL") != "1":
#     logger.info("Attempting local self-relaunch for development...")
#     os.environ["_IS_STRL"] = "1"; port = os.getenv("PORT", "8501")
#     try:
#         import socket
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#             s.bind(("0.0.0.0", int(port)))
#         can_bind = True
#     except socket.error:
#         can_bind = False

#     if can_bind:
#         cmd = [sys.executable, "-m", "streamlit", "run", __file__, "--server.port", port, "--server.address", "0.0.0.0"]
#         logger.info(f"Relaunching with Streamlit: {' '.join(cmd)}"); subprocess.run(cmd, check=False)
#     else:
#         logger.info(f"Port {port} already in use. Assuming Streamlit is already running or will be managed externally.")
# else:
#    if __name__ == "__main__":
#        logger.info("Script running (likely on Streamlit Cloud or already relaunched).")
