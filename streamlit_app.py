#!/usr/bin/env python3
"""
OpenRouter Streamlit Chat — Full Edition (Redesigned UI - Polished)
"""

# ───────────────────────── Imports ─────────────────────────
import json, logging, os, sys, subprocess, time, requests
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo
import streamlit as st

# ────────────────────────── Configuration ───────────────────
OPENROUTER_API_KEY  = "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" # Replace
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT     = 120

FALLBACK_MODEL_ID = "deepseek/deepseek-chat-v3-0324:free" # Example, ensure this model is actually free/available
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
ROUTER_MODEL_ID = "mistralai/mistral-7b-instruct:free" # Example, ensure this model is available

MAX_TOKENS = {"A": 16_000, "B": 8_000, "C": 16_000, "D": 8_000, "F": 8_000}
PLAN = { # (Daily, Weekly, Monthly)
    "A": (10, 70, 300), "B": (5, 35, 150), "C": (1, 7, 30),
    "D": (4, 28, 120), "F": (180, 500, 2000) # 'F' is often a free/high-limit tier
}
EMOJI = {"A": "🌟", "B": "🔷", "C": "🟥", "D": "🟢", "F": "🌀"}
MODEL_DESCRIPTIONS = { # For popover details
    "A": "🌟 (gemini-2.5-pro-preview) – Top-quality, creative, expensive.",
    "B": "🔷 (o4-mini) – Mid-stakes reasoning, cost-effective.",
    "C": "🟥 (chatgpt-4o-latest) – Polished/empathetic, pricier.",
    "D": "🟢 (deepseek-r1) – Cheap factual reasoning.",
    "F": "🌀 (gemini-2.5-flash-preview) – Quick, general purpose."
}

TZ = ZoneInfo("Australia/Sydney") # Replace with your timezone
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
                logging.info(f"Removed old model key '{k_rem}' from quota usage '{period_usage_key}'.")
    _reset(q, "d", _today(), zeros); _reset(q, "w", _yweek(), zeros); _reset(q, "m", _ymonth(), zeros)
    _save(QUOTA_FILE, q); return q
quota = _load_quota()

def remaining(key: str):
    ud = quota.get("d_u", {}).get(key, 0); uw = quota.get("w_u", {}).get(key, 0); um = quota.get("m_u", {}).get(key, 0)
    if key not in PLAN: logging.error(f"Unknown key for remaining: {key}"); return 0,0,0
    ld, lw, lm = PLAN[key]; return ld - ud, lw - uw, lm - um

def record_use(key: str):
    if key not in MODEL_MAP: logging.warning(f"Unknown model key for record_use: {key}"); return
    for blk_key in ("d_u", "w_u", "m_u"):
        if blk_key not in quota: quota[blk_key] = {k: 0 for k in MODEL_MAP}
        quota[blk_key][key] = quota[blk_key].get(key, 0) + 1
    _save(QUOTA_FILE, quota)

# ───────────────────── Session Management ───────────────────────
def _delete_unused_blank_sessions(keep_sid: str = None):
    sids_to_delete = [sid for sid, data in sessions.items() if sid != keep_sid and data.get("title") == "New chat" and not data.get("messages")]
    if sids_to_delete:
        for sid_del in sids_to_delete: logging.info(f"Auto-deleting blank session: {sid_del}"); del sessions[sid_del]
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

# ─────────────────────────── Logging ────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)

# ────────────────────────── API Calls ──────────────────────────
def api_post(payload: dict, *, stream: bool=False, timeout: int=DEFAULT_TIMEOUT):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type":  "application/json"}
    logging.info(f"POST /chat/completions → model={payload.get('model')}, stream={stream}, max_tokens={payload.get('max_tokens')}")
    return requests.post(f"{OPENROUTER_API_BASE}/chat/completions", headers=headers, json=payload, stream=stream, timeout=timeout)

def streamed(model: str, messages: list, max_tokens_out: int):
    payload = {"model": model, "messages": messages, "stream": True, "max_tokens": max_tokens_out}
    with api_post(payload, stream=True) as r:
        try: r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            text = r.text; logging.error(f"Stream HTTPError {e.response.status_code}: {text}"); yield None, f"HTTP {e.response.status_code}: {text}"; return
        for line in r.iter_lines():
            if not line: continue
            line_str = line.decode("utf-8")
            if line_str.startswith(": OPENROUTER PROCESSING"): logging.info(f"OpenRouter PING: {line_str.strip()}"); continue
            if not line_str.startswith("data: "): logging.warning(f"Unexpected non-event-stream line: {line}"); continue
            data = line_str[6:].strip()
            if data == "[DONE]": break
            try: chunk = json.loads(data)
            except json.JSONDecodeError: logging.error(f"Bad JSON chunk: {data}"); yield None, "Error decoding response chunk"; return
            if "error" in chunk: msg = chunk["error"].get("message", "Unknown API error"); logging.error(f"API chunk error: {msg}"); yield None, msg; return
            delta = chunk["choices"][0]["delta"].get("content")
            if delta is not None: yield delta, None

# ───────────────────────── Model Routing ───────────────────────
def route_choice(user_msg: str, allowed: list[str]) -> str:
    if not allowed: return "F" if "F" in MODEL_MAP else (list(MODEL_MAP.keys())[0] if MODEL_MAP else "F") # Fallback if no models allowed (should not happen)
    if len(allowed) == 1: return allowed[0]
    
    system_lines = ["You are an intelligent model-routing assistant.", "Select ONLY one letter from the following available models:"]
    for k in allowed:
        if k in MODEL_DESCRIPTIONS: 
            # Provide a concise description for routing decision
            desc_for_router = MODEL_DESCRIPTIONS[k].split('–')[1].strip() if '–' in MODEL_DESCRIPTIONS[k] else MODEL_DESCRIPTIONS[k]
            system_lines.append(f"- {k}: {MODEL_MAP[k].split('/')[-1]} ({desc_for_router})")
        else:
            system_lines.append(f"- {k}: {MODEL_MAP[k].split('/')[-1]}") # Fallback if no description
            
    system_lines.extend(["Based on the user's query, choose the letter that best balances quality, speed, and cost-sensitivity.", "Respond with ONLY the single capital letter. No extra text."])
    
    router_messages = [{"role": "system", "content": "\n".join(system_lines)}, {"role": "user", "content": user_msg}]
    payload_r = {"model": ROUTER_MODEL_ID, "messages": router_messages, "max_tokens": 10, "temperature": 0.1} # Low temp for deterministic choice
    
    try:
        r = api_post(payload_r); r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip().upper()
        logging.info(f"Router raw response: {text}")
        for ch in text: # Pick the first valid character
            if ch in allowed: return ch
    except Exception as e: logging.error(f"Router call error: {e}")
    
    fallback_choice = "F" if "F" in allowed else allowed[0] # Prefer 'F' if available and allowed
    logging.warning(f"Router fallback to model: {fallback_choice}"); return fallback_choice

# ───────────────────── Credits Endpoint ───────────────────────
def get_credits():
    try:
        r = requests.get(f"{OPENROUTER_API_BASE}/credits", headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}, timeout=10)
        r.raise_for_status(); d = r.json()["data"]; return d["total_credits"], d["total_usage"], d["total_credits"] - d["total_usage"]
    except Exception as e: logging.warning(f"Could not fetch /credits: {e}"); return None, None, None

# ───────────────────────── UI Styling ──────────────────────────
def load_custom_css():
    css = """
    <style>
        /* General Styles */
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
        }
        /* Ensure main content area uses specified background */
        [data-testid="stAppViewContainer"] > .main {
            background-color: #171923; /* Dark main area */
        }
        html[data-theme="light"] [data-testid="stAppViewContainer"] > .main {
            background-color: #FFFFFF; /* Light main area */
        }
        [data-testid="stAppViewContainer"] > .main > .block-container {
            padding-top: 2rem; 
            padding-bottom: 2rem;
            max-width: 860px; 
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #1A202C; /* Darker sidebar */
            padding: 1.25rem 1rem;
            border-right: 1px solid #2D3748; 
        }
        html[data-theme="light"] [data-testid="stSidebar"] {
            background-color: #F7FAFC; /* Light sidebar */
            border-right: 1px solid #E2E8F0;
        }

        /* Sidebar Header (Logo + Title) */
        /* Targeting might need adjustment if Streamlit's DOM changes */
        [data-testid="stSidebar"] > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) { 
            display: flex !important; align-items: center !important; gap: 10px; 
            margin-bottom: 0 !important; 
            padding-bottom: 0;
            border-bottom: none; 
        }
        [data-testid="stSidebar"] .stImage > img {
            border-radius: 6px; width: 38px !important; height: 38px !important;
        }
        [data-testid="stSidebar"] h1 { /* App Title */
            font-size: 1.3rem !important; color: #E2E8F0;
            font-weight: 600; margin-bottom: 0;
        }
        html[data-theme="light"] [data-testid="stSidebar"] h1 { color: #2D3748; }

        /* Primary New Chat Button (Streamlit handles most of type="primary") */
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
             font-weight: 500; /* Ensure font weight */
        }
        
        /* Sidebar Subheaders */
        [data-testid="stSidebar"] h3 {
            font-size: 0.7rem !important; text-transform: uppercase;
            font-weight: 600; color: #A0AEC0;
            margin-top: 1.5rem; margin-bottom: 0.75rem; letter-spacing: 0.05em;
        }
        html[data-theme="light"] [data-testid="stSidebar"] h3 { color: #718096; }

        /* --- Custom Model Usage Display --- */
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
            margin-bottom: 0.4rem; /* Space before popover button */
            overflow: hidden;
        }
        html[data-theme="light"] .progress-bar-container { background-color: #E2E8F0; }
        .progress-bar-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.3s ease-in-out, background-color 0.3s ease-in-out;
        }
        
        /* Popover button (for model details) */
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
        }
        [data-testid="stSidebar"] button[data-testid*="stPopover"]:hover {
            color: var(--primary-color) !important;
            border-color: var(--primary-color) !important;
            background-color: rgba(var(--primary-color-rgb), 0.1) !important;
        }


        /* Sidebar Chat List Buttons */
        [data-testid="stSidebar"] .stButton > button:not([kind="primary"]) { /* Exclude primary button */
            border-radius: 6px; border: none; 
            padding: 0.6rem 0.75rem; font-size: 0.875rem; font-weight: 400;
            background-color: transparent; color: #CBD5E0;
            transition: background-color 0.2s, color 0.2s, border-left-color 0.2s;
            width: 100%; margin-bottom: 0.2rem; text-align: left;
            display: flex; align-items: center; gap: 8px;
            border-left: 3px solid transparent;
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
        
        /* Active session button */
        [data-testid="stSidebar"] .stButton > button:not([kind="primary"]):has(span:contains("🔹")) { 
            color: var(--primary-color) !important; 
            background-color: color-mix(in srgb, var(--primary-color) 10%, transparent);
            border-left: 3px solid var(--primary-color);
            font-weight: 500;
        }

        /* Sidebar Caption ("Current chat is empty...") */
        [data-testid="stSidebar"] .stCaption {
            color: #718096; font-size: 0.8rem; text-align: left;
            padding: 0.2rem 0.1rem 1rem 0.1rem; line-height: 1.4;
        }
        html[data-theme="light"] [data-testid="stSidebar"] .stCaption { color: #6B7280; }

        /* General Separator (for st.markdown("---")) */
        [data-testid="stSidebar"] hr {
            margin: 1.25rem -1rem; /* Extend to edges */
            border: 0;
            border-top: 1px solid #2D3748;
        }
        html[data-theme="light"] [data-testid="stSidebar"] hr { border-top-color: #E2E8F0; }

        /* Main Chat Area - Empty State */
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

        /* Chat Input */
        [data-testid="stChatInput"] {
            background-color: #1A202C; /* Match sidebar for footer consistency */
            border-top: 1px solid #2D3748; padding: 0.75rem 1rem;
            position: sticky; bottom: 0; /* Make chat input sticky */
        }
        html[data-theme="light"] [data-testid="stChatInput"] { background-color: #F7FAFC; border-top-color: #E2E8F0; }

        [data-testid="stChatInput"] textarea { 
            border-radius: 8px !important; 
            border: 1px solid #4A5568 !important;
            background-color: #2D3748 !important; 
            color: #E2E8F0 !important;
            padding: 10px 14px !important;
            line-height: 1.5 !important;
        }
        html[data-theme="light"] [data-testid="stChatInput"] textarea {
            border-color: #CBD5E0 !important; background-color: #FFFFFF !important; color: #2D3748 !important;
        }
        [data-testid="stChatInput"] textarea:focus {
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 2px color-mix(in srgb, var(--primary-color) 30%, transparent) !important;
        }
        [data-testid="stChatInput"] button svg { fill: #A0AEC0; }
        [data-testid="stChatInput"] button:hover svg { fill: var(--primary-color); }

        /* Chat Message Bubbles */
        [data-testid="stChatMessage"] {
            border-radius: 10px; padding: 10px 16px; margin-bottom: 8px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05); border: none; max-width: 75%; line-height: 1.6;
        }
        [data-testid^="stChatMessageUser"] { 
            background-color: var(--primary-color); color: white; margin-left: auto; 
            border-bottom-right-radius: 4px; border-top-left-radius: 10px; border-top-right-radius: 10px; border-bottom-left-radius: 10px;
        }
        [data-testid^="stChatMessageUser"] .stMarkdown p, [data-testid^="stChatMessageUser"] .stMarkdown li { color: white !important; }
        
        [data-testid^="stChatMessageAssistant"] { 
            background-color: #2D3748; color: #E2E8F0; 
            border-bottom-left-radius: 4px;  border-top-left-radius: 10px; border-top-right-radius: 10px; border-bottom-right-radius: 10px;
            margin-right: auto;
        }
        html[data-theme="light"] [data-testid^="stChatMessageAssistant"] { background-color: #E9ECF2; color: #2D3748; }
        [data-testid^="stChatMessageAssistant"] .stMarkdown p, [data-testid^="stChatMessageAssistant"] .stMarkdown li { color: inherit !important; }
        
        /* Expander Styling */
        .stExpander {
            border: 1px solid #2D3748; border-radius: 8px; margin-bottom: 1rem;
            background-color: transparent; 
        }
        html[data-theme="light"] .stExpander { border-color: #CBD5E0; }
        .stExpander header {
            font-weight: 500; font-size: 0.8rem; padding: 0.5rem 0.8rem !important;
            background-color: rgba(45, 55, 72, 0.5); /* Semi-transparent header */
            border-bottom: 1px solid #2D3748;
            border-top-left-radius: 7px; border-top-right-radius: 7px;
            color: #A0AEC0;
        }
        html[data-theme="light"] .stExpander header { background-color: rgba(226, 232, 240, 0.5); border-color: #CBD5E0; color: #4A5568; }
        .stExpander header:hover { background-color: #2D3748; }
        html[data-theme="light"] .stExpander header:hover { background-color: #E2E8F0; }
        .stExpander div[data-testid="stExpanderDetails"] { padding: 0.75rem 1rem; background-color: transparent; }

        /* Main area scrollbar - optional subtle styling */
        /* For Webkit browsers */
        .main::-webkit-scrollbar { width: 8px; }
        .main::-webkit-scrollbar-track { background: transparent; }
        .main::-webkit-scrollbar-thumb { background-color: #4A5568; border-radius: 10px; border: 2px solid transparent; background-clip: content-box;}
        .main::-webkit-scrollbar-thumb:hover { background-color: #718096; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ───────────────────────── Streamlit UI ────────────────────────
st.set_page_config(page_title="OpenRouter Chat", layout="wide", initial_sidebar_state="expanded")
load_custom_css()

# Initial SID Management
needs_save_and_rerun_on_startup = False
if "sid" not in st.session_state: st.session_state.sid = _new_sid(); needs_save_and_rerun_on_startup = True
elif st.session_state.sid not in sessions:
    logging.warning(f"SID {st.session_state.sid} not found. New chat."); st.session_state.sid = _new_sid(); needs_save_and_rerun_on_startup = True
else:
    if _delete_unused_blank_sessions(keep_sid=st.session_state.sid): needs_save_and_rerun_on_startup = True
if needs_save_and_rerun_on_startup: _save(SESS_FILE, sessions); st.rerun()

if "credits" not in st.session_state:
    st.session_state.credits = dict(zip(("total", "used", "remaining"), get_credits()))
    st.session_state.credits_ts = time.time()

# ───────────────────────── Sidebar ─────────────────────────────
with st.sidebar:
    st.image("https://avatars.githubusercontent.com/u/130328222?s=200&v=4", width=38) 
    st.title("OpenRouter Chat")
    st.markdown("---") # Visual separator below title

    current_session_is_truly_blank = (st.session_state.sid in sessions and
                                      sessions[st.session_state.sid].get("title") == "New chat" and
                                      not sessions[st.session_state.sid].get("messages"))
    
    if st.button("➕ New Chat", key="new_chat_button_top", use_container_width=True, type="primary", disabled=current_session_is_truly_blank):
        st.session_state.sid = _new_sid(); _save(SESS_FILE, sessions); st.rerun()
    
    st.markdown("---") 

    st.subheader("Model Usage (Daily)")
    active_model_keys = sorted(MODEL_MAP.keys())
    for m_key in active_model_keys:
        left_d, _, _ = remaining(m_key) # Only showing daily for brevity in this new UI
        lim_d, _, _  = PLAN[m_key]
        
        is_unlimited = lim_d > 900_000 # Heuristic for "unlimited"
        progress_value = 1.0 if is_unlimited else (max(0.0, left_d / lim_d if lim_d > 0 else 0.0))
        
        try:
            # Attempt to extract a cleaner model name (e.g., "gemini-2.5-pro-preview")
            model_display_name = MODEL_DESCRIPTIONS[m_key].split('(')[1].split(')')[0].strip()
        except IndexError: # Fallback if description format is unexpected
            model_display_name = MODEL_MAP[m_key].split('/')[-1] # Basic name extraction

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
            <div class="progress-bar-fill" style="width: {progress_value*100}%; background-color: {'#4caf50' if progress_value > 0.5 else ('#ffc107' if progress_value > 0.25 else '#f44336')};">
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Popover for more details
        with st.popover(f"Details: {m_key}", use_container_width=True, key=f"popover_{m_key}"):
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
    for sid_key in sorted_sids:
        title = sessions[sid_key].get("title", "Untitled")
        display_title_text = title[:28] + ("…" if len(title) > 28 else "")
        
        button_label = display_title_text
        if st.session_state.sid == sid_key:
            # Using a simple text prefix for active state for broader CSS compatibility with :has
            button_label = f"🔹 {display_title_text}" 
            # The CSS targets this with: button:has(span:contains("🔹"))

        if st.button(button_label, key=f"session_button_{sid_key}", use_container_width=True):
            if st.session_state.sid != sid_key:
                st.session_state.sid = sid_key
                if _delete_unused_blank_sessions(keep_sid=sid_key): _save(SESS_FILE, sessions)
                st.rerun()
    st.markdown("---")

    st.caption(f"Routing via: {ROUTER_MODEL_ID.split('/')[-1]}")
    
    tot, used, rem = (st.session_state.credits.get(k) for k in ("total","used","remaining"))
    with st.expander("Account Credits", expanded=False):
        if st.button("Refresh Credits", key="refresh_credits_button", use_container_width=True):
            st.session_state.credits = dict(zip(("total","used","remaining"), get_credits()))
            st.session_state.credits_ts = time.time(); st.rerun()
        if tot is None: st.warning("Could not fetch credits.")
        else:
            st.markdown(f"**Purchased:** ${tot:.2f} cr\n\n**Used:** ${used:.2f} cr\n\n**Remaining:** ${rem:.2f} cr")
            try: st.caption(f"Last updated: {datetime.fromtimestamp(st.session_state.credits_ts, TZ).strftime('%-d %b %Y, %H:%M:%S')}")
            except TypeError: st.caption("Last updated: N/A")

# ────────────────────────── Main Chat Panel ─────────────────────
current_sid = st.session_state.sid
if current_sid not in sessions: 
    st.error("Chat session error. Creating new."); current_sid = _new_sid(); st.session_state.sid = current_sid
    _save(SESS_FILE, sessions); st.rerun()

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
        role = msg["role"]; avatar = "👤" # Default to user
        if role == "assistant":
            model_key = msg.get("model")
            avatar = FALLBACK_MODEL_EMOJI if model_key == FALLBACK_MODEL_KEY else EMOJI.get(model_key, EMOJI.get("F", "🤖"))
        with st.chat_message(role, avatar=avatar): st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything…", key=f"chat_input_{current_sid}"):
    if current_session_is_truly_blank: # If it's the first message in a "New chat"
        st.session_state.new_chat_button_top_clicked_once = True # Suppress "Current chat is empty" caption

    chat_history.append({"role":"user","content":prompt})
    
    # Display user message immediately if chat wasn't completely empty before this message
    # Or if it was empty, it will be shown on the rerun after assistant responds
    if not is_new_empty_chat: 
        with st.chat_message("user", avatar="👤"): st.markdown(prompt)

    # Determine model for response
    allowed_standard_models = [k for k in MODEL_MAP if remaining(k)[0] > 0] # Check daily quota
    chosen_model_key, model_id_to_use, max_tokens_api, avatar_resp = FALLBACK_MODEL_KEY, FALLBACK_MODEL_ID, FALLBACK_MODEL_MAX_TOKENS, FALLBACK_MODEL_EMOJI
    use_fallback = not allowed_standard_models

    if not use_fallback:
        chosen_model_key = route_choice(prompt, allowed_standard_models) # No need for len check, route_choice handles it
        logging.info(f"Chosen model key: '{chosen_model_key}' (Fallback={use_fallback})")
        if chosen_model_key in MODEL_MAP: # Ensure router returned a valid key
            model_id_to_use = MODEL_MAP[chosen_model_key]
            max_tokens_api = MAX_TOKENS[chosen_model_key]
            avatar_resp = EMOJI[chosen_model_key]
        else: # Router failed or returned invalid key, force fallback
            logging.warning(f"Router returned invalid key '{chosen_model_key}'. Forcing fallback.")
            use_fallback = True 
            # chosen_model_key, model_id_to_use, etc. remain as fallback values
    
    if use_fallback and chosen_model_key != FALLBACK_MODEL_KEY: # This means standard models were allowed, but router failed or picked one that became unavailable
        st.info(f"{FALLBACK_MODEL_EMOJI} Could not use routed choice. Using fallback: {FALLBACK_MODEL_ID.split('/')[-1]}")
        chosen_model_key = FALLBACK_MODEL_KEY # Ensure it's marked as fallback
        logging.info(f"Forcing fallback model: {FALLBACK_MODEL_ID}")
    elif use_fallback: # All standard quotas were used initially
        st.info(f"{FALLBACK_MODEL_EMOJI} Standard daily quotas used. Using fallback: {FALLBACK_MODEL_ID.split('/')[-1]}")
        logging.info(f"All standard daily quotas used. Using fallback model: {FALLBACK_MODEL_ID}")
        
    response_content, api_ok = "", True
    
    # Streaming logic: always stream, but for new empty chat, the user message and assistant response appear together after full generation.
    # For existing chats, assistant streams live.
    if not is_new_empty_chat:
        with st.chat_message("assistant", avatar=avatar_resp):
            placeholder = st.empty()
            for chunk, err_msg in streamed(model_id_to_use, chat_history, max_tokens_api):
                if err_msg: response_content = f"❗ **API Error**: {err_msg}"; placeholder.error(response_content); api_ok=False; break
                if chunk: response_content += chunk; placeholder.markdown(response_content + "▌")
            if api_ok: placeholder.markdown(response_content)
    else: # For new empty chats, generate response but don't stream live to UI, will show on full rerun
        for chunk, err_msg in streamed(model_id_to_use, chat_history, max_tokens_api):
            if err_msg: response_content = f"❗ **API Error**: {err_msg}"; api_ok=False; break
            if chunk: response_content += chunk
            
    chat_history.append({"role":"assistant","content":response_content,"model": chosen_model_key}) # Store chosen_model_key, not FALLBACK_MODEL_KEY directly if it was a forced fallback
    if api_ok and not use_fallback and chosen_model_key != FALLBACK_MODEL_KEY: # Only record use for non-fallback standard models
        record_use(chosen_model_key)
    
    if sessions[current_sid]["title"] == "New chat" and len(chat_history) >=2 : 
        sessions[current_sid]["title"] = _autoname(prompt)
        _delete_unused_blank_sessions(keep_sid=current_sid) 

    _save(SESS_FILE, sessions)
    st.rerun()

# ───────────────────────── Self-Relaunch ──────────────────────
if __name__ == "__main__" and os.getenv("_IS_STRL") != "1":
    os.environ["_IS_STRL"] = "1"; port = os.getenv("PORT", "8501")
    cmd = [sys.executable, "-m", "streamlit", "run", __file__, "--server.port", port, "--server.address", "0.0.0.0"]
    logging.info(f"Relaunching with Streamlit: {' '.join(cmd)}"); subprocess.run(cmd, check=False)
