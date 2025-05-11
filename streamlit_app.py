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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(module)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

# ────────────────────────── Configuration ───────────────────
OPENROUTER_API_KEY  = "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"  # Replace
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT     = 120

FALLBACK_MODEL_ID        = "deepseek/deepseek-chat-v3-0324:free"
FALLBACK_MODEL_KEY       = "_FALLBACK_"
FALLBACK_MODEL_EMOJI     = "🆓"
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
PLAN = {
    "A": (10,   70,  300),
    "B": (5,    35,  150),
    "C": (1,     7,   30),
    "D": (4,    28,  120),
    "F": (180, 500, 2000)
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

# ──────────────────────── Helper Functions ─────────────────────
def _load(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default

def _save(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2))

def _today():
    return date.today().isoformat()

def _yweek():
    return datetime.now(TZ).strftime("%G-%V")

def _ymonth():
    return datetime.now(TZ).strftime("%Y-%m")

# ───────────────────── Quota Management ───────────────────────
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
    _reset(q, "d", _today(), zeros)
    _reset(q, "w", _yweek(), zeros)
    _reset(q, "m", _ymonth(), zeros)
    _save(QUOTA_FILE, q)
    return q

quota = _load_quota()

def remaining(key: str):
    ud = quota.get("d_u", {}).get(key, 0)
    uw = quota.get("w_u", {}).get(key, 0)
    um = quota.get("m_u", {}).get(key, 0)
    if key not in PLAN:
        logger.error(f"Unknown key for remaining: {key}")
        return 0, 0, 0
    ld, lw, lm = PLAN[key]
    return ld - ud, lw - uw, lm - um

def record_use(key: str):
    if key not in MODEL_MAP:
        logger.warning(f"Unknown model key for record_use: {key}")
        return
    for blk_key in ("d_u", "w_u", "m_u"):
        if blk_key not in quota:
            quota[blk_key] = {k: 0 for k in MODEL_MAP}
        quota[blk_key][key] = quota[blk_key].get(key, 0) + 1
    _save(QUOTA_FILE, quota)

# ───────────────────── Session Management ───────────────────────
def _delete_unused_blank_sessions(keep_sid: str = None):
    sids_to_delete = [sid for sid, data in sessions.items()
                        if sid != keep_sid and data.get("title") == "New chat"
                        and not data.get("messages")]
    if sids_to_delete:
        for sid_del in sids_to_delete:
            logger.info(f"Auto-deleting blank session: {sid_del}")
            del sessions[sid_del]
        return True
    return False

sessions = _load(SESS_FILE, {})
def _new_sid():
    _delete_unused_blank_sessions(keep_sid=None)
    sid = str(int(time.time() * 1000))
    sessions[sid] = {"title": "New chat", "messages": []}
    return sid

def _autoname(seed: str) -> str:
    words = seed.strip().split()
    cand = " ".join(words[:3]) or "Chat"
    return (cand[:25] + "…") if len(cand) > 25 else cand

# ───────────────────────── API Calls ─────────────────────────
def api_post(payload: dict, *, stream: bool=False, timeout: int=DEFAULT_TIMEOUT):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type":  "application/json"}
    logger.info(f"POST /chat/completions → model={payload.get('model')}, stream={stream}, max_tokens={payload.get('max_tokens')}")
    return requests.post(f"{OPENROUTER_API_BASE}/chat/completions", headers=headers, json=payload, stream=stream, timeout=timeout)

def streamed(model: str, messages: list, max_tokens_out: int):
    payload = {"model": model, "messages": messages, "stream": True, "max_tokens": max_tokens_out}
    with api_post(payload, stream=True) as r:
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            text = r.text
            logger.error(f"Stream HTTPError {e.response.status_code}: {text}")
            yield None, f"HTTP {e.response.status_code}: {text}"
            return
        for line in r.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8")
            if line_str.startswith(": OPENROUTER PROCESSING"):
                logger.info(f"OpenRouter PING: {line_str.strip()}")
                continue
            if not line_str.startswith("data: "):
                logger.warning(f"Unexpected non-event-stream line (decoded): {line_str.strip()}")
                continue
            data = line_str[6:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                logger.error(f"Bad JSON chunk: {data}")
                yield None, "Error decoding response chunk"
                return

            if "error" in chunk:
                msg_obj = chunk["error"]
                msg = "Unknown API error in stream chunk"
                if isinstance(msg_obj, dict) and "message" in msg_obj:
                    msg = msg_obj["message"]
                logger.error(f"API stream chunk error: {msg}")
                yield None, msg
                return

            delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            if delta is not None:
                yield delta, None

# ───────────────────────── Model Routing ───────────────────────
def route_choice(user_msg: str, allowed: list[str]) -> str:
    if not allowed:
        logger.warning("Router: No models allowed, defaulting to F or first available.")
        return "F" if "F" in MODEL_MAP else (list(MODEL_MAP.keys())[0] if MODEL_MAP else "F")
    if len(allowed) == 1:
        logger.info(f"Router: Only one model allowed ({allowed[0]}), selecting it directly.")
        return allowed[0]

    system_lines = [
        "You are an intelligent model-routing assistant.",
        "Select ONLY one letter from the following available models:"
    ]
    for k in allowed:
        if k in MODEL_DESCRIPTIONS and '–' in MODEL_DESCRIPTIONS[k]:
            desc_for_router = MODEL_DESCRIPTIONS[k].split('–')[1].strip()
        else:
            desc_for_router = MODEL_DESCRIPTIONS.get(k, MODEL_MAP.get(k, ''))
        system_lines.append(f"- {k}: {MODEL_MAP[k].split('/')[-1]} ({desc_for_router})")
    system_lines.extend([
        "Based on the user's query, choose the letter that best balances quality, speed, and cost-sensitivity.",
        "Respond with ONLY the single capital letter. No extra text."
    ])

    router_messages = [
        {"role": "system",  "content": "\n".join(system_lines)},
        {"role": "user",    "content": user_msg}
    ]
    payload_r = {"model": ROUTER_MODEL_ID, "messages": router_messages, "max_tokens": 10, "temperature": 0.1}

    try:
        r = api_post(payload_r)
        r.raise_for_status()
        r_json = r.json()

        if "error" in r_json:
            logger.error(f"Router API returned an error object: {r_json['error']}")
        elif not r_json.get("choices") or not r_json["choices"][0].get("message") or not r_json["choices"][0]["message"].get("content"):
            logger.error(f"Router API response malformed: {r_json}")
        else:
            raw_text = r_json["choices"][0]["message"]["content"].strip().upper()
            logger.info(f"Router raw response: '{raw_text}'")

            # look for exact standalone letter matches first
            for letter_allowed in sorted(allowed):
                if re.search(rf"\b{re.escape(letter_allowed)}\b", raw_text):
                    logger.info(f"Router selected model: '{letter_allowed}' (standalone regex match).")
                    return letter_allowed
            # fallback: first character in the response that matches
            for char_code in raw_text:
                if char_code in allowed:
                    logger.info(f"Router selected model: '{char_code}' (first character match).")
                    return char_code
            logger.warning(f"Router response '{raw_text}' did not contain an identifiable allowed model from {allowed}. Falling back.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Router API call failed (RequestException): {e}")
    except json.JSONDecodeError as e:
        logger.error(f"Router API response not valid JSON: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during router call: {e}")

    # final fallback
    fallback_choice = "F" if "F" in allowed else allowed[0]
    logger.warning(f"Router falling back to model: {fallback_choice}")
    return fallback_choice

# ───────────────────── Credits Endpoint ───────────────────────
def get_credits():
    try:
        r = requests.get(
            f"{OPENROUTER_API_BASE}/credits",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            timeout=10
        )
        r.raise_for_status()
        d = r.json().get("data", {})
        return d.get("total_credits"), d.get("total_usage"), d.get("total_credits") - d.get("total_usage")
    except Exception as e:
        logger.warning(f"Could not fetch /credits: {e}")
        return None, None, None

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

        /* +++ NEW CHAT INPUT STYLES +++ */
        [data-testid="stChatInput"] {
            background-color: #2D3748 !important;
            border: 1px solid #4A5568 !important;
            border-radius: 10px !important;
            padding: 0.5rem 0.75rem !important;
            margin: 0.5rem 1rem 1rem 1rem !important;
            position: sticky !important;
            bottom: 1rem !important;
            left: 0 !important; right: 0 !important;
            width: calc(100% - 2rem) !important;
            max-width: calc(860px - 2rem) !important;
            margin-left: auto !important;
            margin-right: auto !important;
            z-index: 100 !important;
            display: flex !important;
            align-items: center !important;
            gap: 0.75rem !important;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.2) !important;
        }

        html[data-theme="light"] [data-testid="stChatInput"] {
            background-color: #F3F4F6 !important;
            border-color: #D1D5DB !important;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.05) !important;
        }

        [data-testid="stChatInput"] textarea {
            flex-grow: 1 !important;
            background-color: #1F2937 !important;
            border: 1px solid #374151 !important;
            color: #E5E7EB !important;
            padding: 10px 14px !important;
            line-height: 1.5 !important;
            box-shadow: none !important;
            margin: 0 !important;
            border-radius: 8px !important;
            outline: none !important;
            transition: border-color 0.2s, box-shadow 0.2s;
            min-height: 42px !important;
            max-height: 200px !important;
            resize: none !important;
        }

        html[data-theme="light"] [data-testid="stChatInput"] textarea {
            background-color: #FFFFFF !important;
            border-color: #D1D5DB !important;
            color: #111827 !important;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: #6B7280 !important;
        }
        html[data-theme="light"] [data-testid="stChatInput"] textarea::placeholder {
            color: #6B7280 !important;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ───────────────────── Main App Logic (Placeholder) ─────────────────────
# This is where your main Streamlit app logic would go.
# For demonstration, I'll add a basic structure.

def display_chat_messages(session_messages):
    for msg in session_messages:
        avatar = "🧑‍💻" if msg["role"] == "user" else EMOJI.get(msg.get("model_key", "_"), "🤖")
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

def main():
    st.set_page_config(page_title="OpenRouter Chat", page_icon="🤖", layout="wide")
    load_custom_css()

    # Initialize session state
    if "current_session_id" not in st.session_state:
        if sessions:
            st.session_state.current_session_id = next(iter(sessions)) # Load first existing
        else:
            st.session_state.current_session_id = _new_sid() # Or create new
            _save(SESS_FILE, sessions) # Save new session file

    if "sessions" not in st.session_state: # Load all sessions into session_state
        st.session_state.sessions = sessions.copy()


    with st.sidebar:
        col1, col2 = st.columns([1,3])
        with col1:
            st.image("https://raw.githubusercontent.com/simonw/llm-ui-grouped/main/static/icon.png") # Replace with your logo
        with col2:
            st.title("OpenRouter Chat")

        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            old_sid = st.session_state.current_session_id
            new_id = _new_sid()
            st.session_state.current_session_id = new_id
            st.session_state.sessions = sessions.copy() # update session_state
            if _delete_unused_blank_sessions(keep_sid=new_id):
                st.session_state.sessions = sessions.copy() # update if deletions occurred
            _save(SESS_FILE, sessions)
            st.rerun()

        st.subheader("Chats")
        # Display existing chats
        sorted_sids = sorted(st.session_state.sessions.keys(), reverse=True)
        for sid_key in sorted_sids:
            s_data = st.session_state.sessions[sid_key]
            label = s_data.get("title", "Chat")
            if st.session_state.current_session_id == sid_key:
                label = f"🔹 {label}"
            if st.button(label, key=f"session_btn_{sid_key}", use_container_width=True):
                st.session_state.current_session_id = sid_key
                st.rerun()
        st.divider()

        with st.expander("Model Usage (Daily)", expanded=True):
            for key, model_id_part in MODEL_MAP.items():
                rd, rw, rm = remaining(key)
                name = model_id_part.split('/')[-1].replace('-preview','').replace('-latest','')
                used_daily = PLAN[key][0] - rd
                max_daily = PLAN[key][0]

                # Progress bar color
                progress_percent = (used_daily / max_daily) * 100 if max_daily > 0 else 0
                if progress_percent < 50: color = "#48BB78" # Green
                elif progress_percent < 85: color = "#F6E05E" # Yellow
                else: color = "#F56565" # Red

                st.markdown(f"""
                <div class="model-usage-item">
                    <div class="model-info">
                        <span class="model-emoji">{EMOJI.get(key, '❓')}</span>
                        <span class="model-key-name" title="{MODEL_DESCRIPTIONS.get(key, '')}">{key}: {name}</span>
                    </div>
                    <span class="quota-text">{used_daily}/{max_daily}</span>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-bar-fill" style="width: {progress_percent}%; background-color: {color};"></div>
                </div>
                """, unsafe_allow_html=True)

            popover_details = st.popover("Details (D/W/M)", use_container_width=True)
            popover_details.markdown("Remaining uses (Daily / Weekly / Monthly)")
            for key in MODEL_MAP.keys():
                rd, rw, rm = remaining(key)
                popover_details.text(f"{EMOJI.get(key, '')} {key}: {rd} / {rw} / {rm}")

        st.divider()
        _, _, credits_rem = get_credits()
        if credits_rem is not None:
            st.caption(f"OpenRouter Credits: ${credits_rem:.2f}")
        else:
            st.caption("OpenRouter Credits: N/A")

        st.caption(f"© {datetime.now().year} • v0.2.1-polished")


    # Main chat area
    current_session_data = st.session_state.sessions.get(st.session_state.current_session_id, {"title": "New Chat", "messages": []})
    session_messages = current_session_data.get("messages", [])

    if not session_messages:
        st.markdown(f"""
        <div class="empty-chat-container">
            <img src="https://raw.githubusercontent.com/simonw/llm-ui-grouped/main/static/icon.png" class="logo-main" alt="App Logo">
            <h2>How can I help you today, Asher?</h2>
            <p>I can help you choose the best model for your task based on its capabilities and your remaining quotas. Just type your query below!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        display_chat_messages(session_messages)


    if prompt := st.chat_input("Ask anything..."):
        session_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        # Auto-name session if it's the first message
        if len(session_messages) == 1 and current_session_data["title"] == "New chat":
            current_session_data["title"] = _autoname(prompt)
            st.session_state.sessions[st.session_state.current_session_id]["title"] = current_session_data["title"]
            _save(SESS_FILE, sessions) # Save all sessions, including updated title
            # No st.rerun() here, let it flow to model choice

        # Determine available models based on quota
        available_models_keys = [k for k in MODEL_MAP if remaining(k)[0] > 0] # Daily quota
        if not available_models_keys: # If all daily quotas used, check weekly
            available_models_keys = [k for k in MODEL_MAP if remaining(k)[1] > 0]
            if not available_models_keys: # If all weekly quotas used, check monthly
                 available_models_keys = [k for k in MODEL_MAP if remaining(k)[2] > 0]

        chosen_model_key = FALLBACK_MODEL_KEY # Default to fallback if no models have quota

        if not available_models_keys:
            logger.warning("All models have exhausted their D/W/M quotas. Using fallback.")
            chosen_model_id = FALLBACK_MODEL_ID
            chosen_model_max_tokens = FALLBACK_MODEL_MAX_TOKENS
            chosen_model_emoji = FALLBACK_MODEL_EMOJI
            st.warning(f"All model quotas exhausted. Using fallback model: {FALLBACK_MODEL_ID.split('/')[-1]}. Responses may be limited.", icon="⚠️")
        else:
            logger.info(f"Models with quota: {available_models_keys}")
            chosen_model_key = route_choice(prompt, available_models_keys)
            chosen_model_id = MODEL_MAP[chosen_model_key]
            chosen_model_max_tokens = MAX_TOKENS.get(chosen_model_key, 8000)
            chosen_model_emoji = EMOJI.get(chosen_model_key, "🤖")


        with st.chat_message("assistant", avatar=chosen_model_emoji):
            message_placeholder = st.empty()
            full_response = ""
            error_message = None

            api_messages = [{"role": m["role"], "content": m["content"]} for m in session_messages]

            for chunk_text, err in streamed(chosen_model_id, api_messages, chosen_model_max_tokens):
                if err:
                    error_message = f"Error: {err}"
                    logger.error(error_message)
                    break
                if chunk_text:
                    full_response += chunk_text
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        if error_message:
            session_messages.append({"role": "assistant", "content": error_message, "model_key": chosen_model_key, "error": True})
        else:
            session_messages.append({"role": "assistant", "content": full_response, "model_key": chosen_model_key})
            if chosen_model_key != FALLBACK_MODEL_KEY: # Only record use if not fallback
                record_use(chosen_model_key)
                _load_quota() # Refresh quota in memory after recording

        # Update the specific session in the main 'sessions' dict and save
        st.session_state.sessions[st.session_state.current_session_id]["messages"] = session_messages
        _save(SESS_FILE, st.session_state.sessions)
        st.rerun() # Rerun to update UI with new message and possibly quota changes


if __name__ == "__main__":
    main()
