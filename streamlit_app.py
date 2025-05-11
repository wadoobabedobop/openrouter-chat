#!/usr/bin/env python3
"""
OpenRouter Streamlit Chat — Full Edition (Redesigned UI)
• Persistent chat sessions
• Daily/weekly/monthly quotas
• Pretty ‘token-jar’ gauges (fixed at top)
• Detailed model-routing panel (Mistral router)
• Live credit/usage stats (GET /credits)
• Auto-titling of new chats
• Comprehensive logging
• Self-relaunch under python main.py
"""

# ───────────────────────── Imports ─────────────────────────
import json, logging, os, sys, subprocess, time, requests
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo
import streamlit as st

# ────────────────────────── Configuration ───────────────────
OPENROUTER_API_KEY  = "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" # Replace with your actual key or environment variable
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT     = 120

# Fallback Model Configuration (used when other quotas are exhausted)
FALLBACK_MODEL_ID = "deepseek/deepseek-chat-v3-0324:free"
FALLBACK_MODEL_KEY = "_FALLBACK_"  # Internal key, not for display in jars or regular selection
FALLBACK_MODEL_EMOJI = "🆓"        # Emoji for the fallback model
FALLBACK_MODEL_MAX_TOKENS = 8000   # Max output tokens for the fallback model

# Model definitions (standard, quota-tracked models)
MODEL_MAP = {
    "A": "google/gemini-2.5-pro-preview",
    "B": "openai/o4-mini",
    "C": "openai/chatgpt-4o-latest",
    "D": "deepseek/deepseek-r1",
    "F": "google/gemini-2.5-flash-preview"
}
ROUTER_MODEL_ID = "mistralai/mistral-7b-instruct:free"

MAX_TOKENS = {
    "A": 16_000, "B": 8_000, "C": 16_000,
    "D": 8_000, "F": 8_000
}

PLAN = {
    "A": (10, 10 * 7, 10 * 30),
    "B": (5, 5 * 7, 5 * 30),
    "C": (1, 1 * 7, 1 * 30),
    "D": (4, 4 * 7, 4 * 30),
    "F": (180, 500, 2000)
}

EMOJI = {
    "A": "🌟", "B": "🔷", "C": "🟥", "D": "🟢", "F": "🌀"
}

MODEL_DESCRIPTIONS = {
    "A": "🌟 (gemini-2.5-pro-preview) – top-quality, creative, expensive.",
    "B": "🔷 (o4-mini) – mid-stakes reasoning, cost-effective.",
    "C": "🟥 (chatgpt-4o-latest) – polished/empathetic, pricier.",
    "D": "🟢 (deepseek-r1) – cheap factual reasoning.",
    "F": "🌀 (gemini-2.5-flash-preview) – quick, free-tier, general purpose."
}

TZ = ZoneInfo("Australia/Sydney")
DATA_DIR   = Path(__file__).parent
SESS_FILE  = DATA_DIR / "chat_sessions.json"
QUOTA_FILE = DATA_DIR / "quotas.json"


# ──────────────────────── Helper Functions ───────────────────────

def _load(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default

def _save(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2))

def _today():    return date.today().isoformat()
def _yweek():    return datetime.now(TZ).strftime("%G-%V")
def _ymonth():   return datetime.now(TZ).strftime("%Y-%m")


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
        logging.error(f"Attempted to get remaining quota for unknown key: {key}")
        return 0, 0, 0
    ld, lw, lm = PLAN[key]
    return ld - ud, lw - uw, lm - um

def record_use(key: str):
    if key not in MODEL_MAP:
        logging.warning(f"Attempted to record usage for unknown or non-standard model key: {key}")
        return
    for blk_key in ("d_u", "w_u", "m_u"):
        if blk_key not in quota:
            quota[blk_key] = {k: 0 for k in MODEL_MAP}
        quota[blk_key][key] = quota[blk_key].get(key, 0) + 1
    _save(QUOTA_FILE, quota)


# ───────────────────── Session Management ───────────────────────

def _delete_unused_blank_sessions(keep_sid: str = None):
    sids_to_delete = []
    for sid, data in sessions.items():
        if sid == keep_sid:
            continue
        if data.get("title") == "New chat" and not data.get("messages"):
            sids_to_delete.append(sid)

    if sids_to_delete:
        for sid_del in sids_to_delete:
            logging.info(f"Auto-deleting blank session: {sid_del}")
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


# ─────────────────────────── Logging ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout
)

# ────────────────────────── API Calls ──────────────────────────
def api_post(payload: dict, *, stream: bool=False, timeout: int=DEFAULT_TIMEOUT):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json"
    }
    logging.info(f"POST /chat/completions → model={payload.get('model')}, stream={stream}, max_tokens={payload.get('max_tokens')}")
    return requests.post(
        f"{OPENROUTER_API_BASE}/chat/completions",
        headers=headers, json=payload, stream=stream, timeout=timeout
    )

def streamed(model: str, messages: list, max_tokens_out: int):
    payload = {
        "model":      model,
        "messages":   messages,
        "stream":     True,
        "max_tokens": max_tokens_out
    }
    with api_post(payload, stream=True) as r:
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            text = r.text
            logging.error(f"Stream HTTPError {e.response.status_code}: {text}")
            yield None, f"HTTP {e.response.status_code}: {text}"
            return

        for line in r.iter_lines():
            if not line: continue
            line_str = line.decode("utf-8")
            if line_str.startswith(": OPENROUTER PROCESSING"):
                logging.info(f"OpenRouter PING: {line_str.strip()}")
                continue
            if not line_str.startswith("data: "):
                logging.warning(f"Unexpected non-event-stream line: {line}")
                continue
            data = line_str[6:].strip()
            if data == "[DONE]": break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                logging.error(f"Bad JSON chunk: {data}")
                yield None, "Error decoding response chunk"
                return
            if "error" in chunk:
                msg = chunk["error"].get("message", "Unknown API error")
                logging.error(f"API chunk error: {msg}")
                yield None, msg
                return
            delta = chunk["choices"][0]["delta"].get("content")
            if delta is not None: yield delta, None

# ───────────────────────── Model Routing ───────────────────────
def route_choice(user_msg: str, allowed: list[str]) -> str:
    if not allowed:
        logging.warning("route_choice called with empty allowed list. Defaulting to 'F'.")
        return "F" if "F" in MODEL_MAP else (list(MODEL_MAP.keys())[0] if MODEL_MAP else "F")
    if len(allowed) == 1:
        logging.info(f"Router: Only one model allowed {allowed[0]}, selecting it directly.")
        return allowed[0]
    system_lines = [
        "You are an intelligent model-routing assistant.",
        "Select ONLY one letter from the following available models:",
    ]
    for k in allowed:
        if k in MODEL_DESCRIPTIONS:
            system_lines.append(f"- {k}: {MODEL_DESCRIPTIONS[k]}")
        else:
            logging.warning(f"Model key {k} found in 'allowed' but not in MODEL_DESCRIPTIONS.")
    system_lines.extend([
        "Based on the user's query, choose the letter that best balances quality, speed, and cost-sensitivity.",
        "Respond with ONLY the single capital letter. No extra text."
    ])
    router_messages = [
        {"role": "system", "content": "\n".join(system_lines)},
        {"role": "user",   "content": user_msg}
    ]
    payload_r = {"model": ROUTER_MODEL_ID, "messages": router_messages, "max_tokens": 10}
    try:
        r = api_post(payload_r)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip().upper()
        logging.info(f"Router raw response: {text}")
        for ch in text:
            if ch in allowed: return ch
    except Exception as e:
        logging.error(f"Router call error: {e}")
    fallback_choice = "F" if "F" in allowed else allowed[0]
    logging.warning(f"Router fallback to model: {fallback_choice}")
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
        d = r.json()["data"]
        return d["total_credits"], d["total_usage"], d["total_credits"] - d["total_usage"]
    except Exception as e:
        logging.warning(f"Could not fetch /credits: {e}")
        return None, None, None

# ───────────────────────── UI Styling ──────────────────────────
def load_custom_css():
    css = """
    <style>
        /* General Styles - Inspired by modern chat UIs */
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
            color: var(--text-color); /* Ensure text color respects theme */
        }

        /* Main app container styling for better background consistency */
        [data-testid="stAppViewContainer"] > .main > .block-container {
            padding-top: 2rem; 
            padding-bottom: 2rem;
            /* max-width: 900px; /* Optional: Constrain main content width */
        }
        /* Attempt to make the main area background consistent for dark themes */
        html[data-theme="dark"] [data-testid="stAppViewContainer"],
        html[data-theme="dark"] .main {
             /* background-color: #171717; /* A very dark grey, adjust if needed or remove if Streamlit's default is fine */
        }


        /* Sidebar Styling - More aligned with mockups */
        [data-testid="stSidebar"] {
            background-color: #1F2937; /* Darker sidebar for dark theme */
            padding: 1.5rem 1rem;
            border-right: 1px solid #374151; /* Darker border for dark theme */
        }
        html[data-theme="light"] [data-testid="stSidebar"] {
            background-color: #F9FAFB; /* Light grey for light theme sidebar */
            border-right: 1px solid var(--border-color);
        }


        /* Sidebar Header (Logo + Title) */
        [data-testid="stSidebar"] > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) { 
            display: flex !important;
            align-items: center !important;
            gap: 12px; 
            margin-bottom: 1.5rem !important;
            padding-bottom: 1rem;
            border-bottom: 1px solid color-mix(in srgb, var(--text-color) 20%, transparent);
        }
        [data-testid="stSidebar"] .stImage > img { /* Your logo */
            border-radius: 8px; 
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            width: 40px !important; 
            height: 40px !important;
        }
        [data-testid="stSidebar"] h1 { /* Targets st.title in sidebar - "OpenRouter Chat" */
            font-size: 1.4rem !important; 
            color: var(--text-color); 
            font-weight: 600;
            margin-bottom: 0;
        }

        /* Sidebar Subheaders ("Daily Jars", "Chats", etc.) */
        [data-testid="stSidebar"] h3 {
            font-size: 0.8rem !important; /* Slightly smaller */
            text-transform: uppercase;
            font-weight: 600;
            color: var(--text-color-secondary);
            margin-top: 1.8rem; /* Adjusted spacing */
            margin-bottom: 0.6rem;
        }

        /* Sidebar Buttons (Session list, etc.) */
        [data-testid="stSidebar"] .stButton > button {
            border-radius: 6px; 
            border: none; 
            padding: 0.6em 0.8em; 
            font-size: 0.9em;
            font-weight: 500;
            background-color: transparent; 
            color: var(--text-color-secondary); 
            transition: background-color 0.2s, color 0.2s;
            width: 100%;
            margin-bottom: 0.2rem;
            text-align: left;
            display: flex; 
            align-items: center;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background-color: color-mix(in srgb, var(--primary-color, #71717A) 15%, transparent);
            color: var(--primary-color, #FAFAFA);
        }
        html[data-theme="light"] [data-testid="stSidebar"] .stButton > button:hover {
            color: var(--primary-color, #1E40AF);
        }
        [data-testid="stSidebar"] .stButton > button:focus,
        [data-testid="stSidebar"] .stButton > button:focus-visible {
            outline: 1px auto var(--primary-color);
            outline-offset: 1px;
        }
        /* Active session button styling for the 🔹 indicator */
        [data-testid="stSidebar"] .stButton > button:has(span:contains("🔹")) { /* If you wrap the title in a span */
             background-color: color-mix(in srgb, var(--primary-color, #71717A) 20%, transparent);
             color: var(--primary-color, #FAFAFA);
             font-weight: 600;
        }
         html[data-theme="light"] [data-testid="stSidebar"] .stButton > button:has(span:contains("🔹")) {
             color: var(--primary-color, #1E40AF);
         }


        /* "New Chat" Button - Make it stand out */
        [data-testid="stSidebar"] [data-testid="stButton-new_chat_button_top"] > button {
             background-color: var(--primary-color, #0EA5E9); /* Default to a blue if not set */
             color: white; 
             font-weight: 600;
        }
        html[data-theme="dark"] [data-testid="stSidebar"] [data-testid="stButton-new_chat_button_top"] > button {
             /* color: var(--text-color-on-primary, white); */
        }
        [data-testid="stSidebar"] [data-testid="stButton-new_chat_button_top"] > button:hover {
             filter: brightness(110%); 
        }


        /* Custom Token Jar Styling */
        .token-jar-container {
            width: 100%;
            max-width: 50px; /* Adjusted for potentially narrower columns if sidebar width changes */
            margin: 0 auto 0.3rem auto;
            text-align: center;
            font-family: inherit;
        }
        .token-jar {
            height: 50px; /* Slightly smaller */
            border: 1px solid var(--border-color);
            border-radius: 6px; /* Softer */
            background: var(--secondary-background-color); 
            position: relative;
            overflow: hidden;
            box-shadow: inset 0 1px 2px var(--shadow-sm, rgba(0,0,0,0.05)); 
            margin-bottom: 3px;
        }
        .token-jar-fill {
            position: absolute;
            bottom: 0;
            width: 100%;
            transition: height 0.3s ease-in-out, background-color 0.3s ease-in-out;
        }
        .token-jar-emoji {
            position: absolute;
            top: 4px; /* Adjust position */
            width: 100%;
            font-size: 16px; /* Adjust size */
            line-height: 1;
        }
        .token-jar-key {
            position: absolute;
            bottom: 4px; /* Adjust position */
            width: 100%;
            font-size: 10px; /* Adjust size */
            font-weight: 600;
            color: var(--text-color-secondary); 
            opacity: 0.9;
            line-height: 1;
        }
        .token-jar-remaining {
            display: block;
            margin-top: 1px;
            font-size: 10px;
            font-weight: 600;
            color: var(--text-color); 
            opacity: 0.9;
            line-height: 1;
        }

        /* Expander Styling - make them blend more */
        .stExpander {
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin-bottom: 1rem;
            background-color: transparent; 
        }
        .stExpander header {
            font-weight: 500; 
            font-size: 0.9rem;
            padding: 0.6rem 1rem !important;
            background-color: color-mix(in srgb, var(--text-color) 5%, transparent); 
            border-bottom: 1px solid var(--border-color);
            border-top-left-radius: 7px;
            border-top-right-radius: 7px;
            color: var(--text-color-secondary);
        }
        .stExpander header:hover {
            background-color: color-mix(in srgb, var(--text-color) 10%, transparent);
        }
        .stExpander div[data-testid="stExpanderDetails"] {
             padding: 0.75rem 1rem;
             background-color: transparent; 
        }


        /* Chat Message Styling - More like modern chat apps */
        [data-testid="stChatMessage"] {
            border-radius: 18px; 
            padding: 12px 18px; 
            margin-bottom: 10px;
            box-shadow: 0 1px 2px var(--shadow-sm, rgba(0,0,0,0.03)); 
            border: none; 
            max-width: 80%; /* Slightly less wide */
            line-height: 1.55;
        }

        /* User message */
        [data-testid^="stChatMessageUser"] { 
            background-color: var(--primary-color, #0EA5E9);
            color: white; 
            margin-left: auto; 
            border-bottom-right-radius: 6px; 
        }
        html[data-theme="light"] [data-testid^="stChatMessageUser"] {
            /* Ensure text color is white or very light for light theme primary */
            color: white;
        }
        [data-testid^="stChatMessageUser"] .stMarkdown p,
        [data-testid^="stChatMessageUser"] .stMarkdown li { /* Target paragraphs and list items */
            color: white !important; /* Force white for user messages */
        }
         html[data-theme="light"] [data-testid^="stChatMessageUser"] .stMarkdown p,
         html[data-theme="light"] [data-testid^="stChatMessageUser"] .stMarkdown li {
             color: white !important;
        }


        /* Assistant message */
        [data-testid^="stChatMessageAssistant"] { 
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            border-bottom-left-radius: 6px; 
            margin-right: auto; /* Ensure it stays left */
        }
        [data-testid^="stChatMessageAssistant"] .stMarkdown p,
        [data-testid^="stChatMessageAssistant"] .stMarkdown li {
            color: var(--text-color) !important; /* Ensure it respects theme */
        }
        
        /* Chat Input */
        [data-testid="stChatInput"] {
            background-color: transparent; /* Blend with main background */
            border-top: 1px solid var(--border-color);
            padding-top: 0.75rem; /* Add some space above input */
        }
        [data-testid="stChatInput"] input, /* The actual input field */
        [data-testid="stChatInput"] textarea {
            border-radius: 10px;
            border: 1px solid var(--border-color);
            background-color: var(--secondary-background-color);
            padding: 10px 12px; /* Adjust padding */
        }
        [data-testid="stChatInput"] input:focus,
        [data-testid="stChatInput"] textarea:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 1px var(--primary-color);
        }


        /* Centered Welcome Message for Empty Chat */
        .empty-chat-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 60vh; /* Ensure it takes up good vertical space */
            text-align: center;
            padding: 2rem;
        }
        .empty-chat-container img.logo-main { /* Class for the main logo */
            width: 70px;
            height: 70px;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .empty-chat-container h2 {
            font-size: 1.8rem; /* Slightly smaller */
            font-weight: 600;
            margin-bottom: 0.75rem;
            color: var(--text-color);
        }
        .empty-chat-container p {
            font-size: 1rem; /* Slightly smaller */
            color: var(--text-color-secondary);
            max-width: 450px;
            line-height: 1.6;
        }
        /* Styling for suggestion chips/buttons (if you implement them) */
        .suggestion-chip-container {
            display: flex;
            gap: 10px;
            margin-top: 1.5rem;
            flex-wrap: wrap; 
            justify-content: center;
        }
        .suggestion-chip {
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9rem;
            border: 1px solid var(--border-color);
            cursor: pointer;
            transition: background-color 0.2s, border-color 0.2s;
        }
        .suggestion-chip:hover {
            background-color: color-mix(in srgb, var(--primary-color, #71717A) 15%, var(--secondary-background-color));
            border-color: var(--primary-color, #71717A);
        }

        hr {
          margin-top: 1.5rem;
          margin-bottom: 1.5rem;
          border: 0;
          border-top: 1px solid var(--border-color);
        }

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ───────────────────────── Streamlit UI ────────────────────────
st.set_page_config(
    page_title="OpenRouter Chat",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_custom_css() # Load custom CSS styles

# Initial SID Management and Cleanup
needs_save_and_rerun_on_startup = False
if "sid" not in st.session_state:
    st.session_state.sid = _new_sid()
    needs_save_and_rerun_on_startup = True
elif st.session_state.sid not in sessions:
    logging.warning(f"Session ID {st.session_state.sid} from state not found in loaded sessions. Creating a new chat.")
    st.session_state.sid = _new_sid()
    needs_save_and_rerun_on_startup = True
else:
    if _delete_unused_blank_sessions(keep_sid=st.session_state.sid):
        needs_save_and_rerun_on_startup = True

if needs_save_and_rerun_on_startup:
    _save(SESS_FILE, sessions)
    st.rerun()

if "credits" not in st.session_state:
    st.session_state.credits = dict(zip(
        ("total", "used", "remaining"),
        get_credits()
    ))
    st.session_state.credits_ts = time.time()


# ───────────────────────── Sidebar ─────────────────────────────
with st.sidebar:
    # Sidebar Header: Logo + Title (CSS will style based on structure)
    st.image("https://avatars.githubusercontent.com/u/130328222?s=200&v=4", width=50) 
    st.title("OpenRouter Chat")

    st.subheader("Daily Jars (Msgs Left)")
    active_model_keys = sorted(MODEL_MAP.keys())
    cols = st.columns(len(active_model_keys))
    for i, m_key in enumerate(active_model_keys):
        left, _, _ = remaining(m_key)
        lim, _, _  = PLAN[m_key]
        pct = 1.0 if lim > 900_000 else max(0.0, left / lim if lim > 0 else 0.0)
        fill = int(pct * 100)
        
        if pct > .5: color = "#4caf50" 
        elif pct > .25: color = "#ffc107" 
        else: color = "#f44336" 
        
        cols[i].markdown(f"""
            <div class="token-jar-container">
              <div class="token-jar">
                <div class="token-jar-fill" style="height:{fill}%; background-color:{color};"></div>
                <div class="token-jar-emoji">{EMOJI[m_key]}</div>
                <div class="token-jar-key">{m_key}</div>
              </div>
              <span class="token-jar-remaining">{'∞' if lim > 900_000 else left}</span>
            </div>""", unsafe_allow_html=True)
    st.divider() 

    current_session_is_truly_blank = False
    if st.session_state.sid in sessions:
        current_session_data = sessions.get(st.session_state.sid)
        if current_session_data and \
           current_session_data.get("title") == "New chat" and \
           not current_session_data.get("messages"):
            current_session_is_truly_blank = True
    
    if st.button("➕ New chat", key="new_chat_button_top", use_container_width=True, disabled=current_session_is_truly_blank):
        new_session_id = _new_sid()
        st.session_state.sid = new_session_id
        _save(SESS_FILE, sessions)
        st.rerun()
    elif current_session_is_truly_blank:
        st.caption("Current chat is empty. Add a message or switch.")

    st.subheader("Chats")
    sorted_sids = sorted(sessions.keys(), key=lambda s: int(s), reverse=True)
    for sid_key in sorted_sids:
        title = sessions[sid_key].get("title", "Untitled")
        display_title = title[:25] + ("…" if len(title) > 25 else "")
        
        if st.session_state.sid == sid_key:
            # Using a span for more specific CSS targeting of active item
            # The :has selector in CSS tries to target this structure.
            # display_title = f"<span>🔹 {display_title}</span>" # Wrapping in span
             display_title = f"🔹 {display_title}" # Simpler, rely on text content for now or more complex CSS

        # Pass display_title as Markdown to render the span if used
        # if st.button(display_title, key=f"session_button_{sid_key}", use_container_width=True, unsafe_allow_html=(st.session_state.sid == sid_key)):
        if st.button(display_title, key=f"session_button_{sid_key}", use_container_width=True):
            if st.session_state.sid != sid_key:
                st.session_state.sid = sid_key
                if _delete_unused_blank_sessions(keep_sid=sid_key):
                    _save(SESS_FILE, sessions)
                st.rerun()
    st.divider()

    st.subheader("Model-Routing Map")
    st.caption(f"Router engine: {ROUTER_MODEL_ID}")
    with st.expander("Letters → Models", expanded=False):
        for k_model in sorted(MODEL_MAP.keys()):
            st.markdown(f"**{k_model}**: {MODEL_DESCRIPTIONS[k_model]} (max_output={MAX_TOKENS[k_model]:,})")
    st.divider()

    tot, used, rem = (
        st.session_state.credits.get("total"),
        st.session_state.credits.get("used"),
        st.session_state.credits.get("remaining"),
    )
    with st.expander("Account stats (credits)", expanded=False):
        if st.button("Refresh Credits", key="refresh_credits_button"):
            st.session_state.credits = dict(zip(
                ("total","used","remaining"), get_credits()
            ))
            st.session_state.credits_ts = time.time()
            st.rerun()
        if tot is None: st.warning("Could not fetch credits.")
        else:
            st.markdown(f"**Purchased:** ${tot:.2f} cr")
            st.markdown(f"**Used:** ${used:.2f} cr")
            st.markdown(f"**Remaining:** ${rem:.2f} cr")
            try:
                last_updated_str = datetime.fromtimestamp(st.session_state.credits_ts, TZ).strftime('%-d %b %Y, %H:%M:%S')
                st.caption(f"Last updated: {last_updated_str}")
            except TypeError: st.caption("Last updated: N/A")


# ────────────────────────── Main Chat Panel ─────────────────────
current_sid = st.session_state.sid
if current_sid not in sessions:
    st.error("Selected chat session not found. Creating a new one.")
    current_sid = _new_sid()
    st.session_state.sid = current_sid
    _save(SESS_FILE, sessions)
    st.rerun()

chat_history = sessions[current_sid]["messages"]
is_new_empty_chat = not chat_history and sessions[current_sid]["title"] == "New chat"

if is_new_empty_chat:
    st.markdown(f"""
    <div class="empty-chat-container">
        <img src="https://avatars.githubusercontent.com/u/130328222?s=200&v=4" class="logo-main">
        <h2>How can I help you today, Asher?</h2>
        <p>I can help you choose the best model for your task based on its capabilities and your remaining quotas. Just type your query below!</p>
        <!-- Example suggestion chips (functionality would need to be added) -->
        <!--
        <div class="suggestion-chip-container">
            <div class="suggestion-chip">Brainstorm ideas</div>
            <div class="suggestion-chip">Write a short story</div>
            <div class="suggestion-chip">Explain a complex topic</div>
        </div>
        -->
    </div>
    """, unsafe_allow_html=True)
else:
    # Optionally, display chat title in main panel if not empty
    # current_title = sessions[current_sid].get('title', 'Untitled')
    # if current_title != "New chat":
    #    st.markdown(f"<h3 style='text-align:center; color: var(--text-color-secondary); font-weight: 500; margin-bottom:1.5rem;'>{current_title}</h3>", unsafe_allow_html=True)

    for msg_idx, msg in enumerate(chat_history):
        role = msg["role"]
        avatar_for_display = "👤" # User
        if role == "assistant":
            model_key_in_message = msg.get("model")
            if model_key_in_message == FALLBACK_MODEL_KEY: avatar_for_display = FALLBACK_MODEL_EMOJI
            elif model_key_in_message in EMOJI: avatar_for_display = EMOJI[model_key_in_message]
            else: avatar_for_display = EMOJI.get("F", "🤖") 
        
        with st.chat_message(role, avatar=avatar_for_display):
            st.markdown(msg["content"])


if prompt := st.chat_input("Ask anything…", key=f"chat_input_{current_sid}"):
    chat_history.append({"role":"user","content":prompt})
    
    # Display user message immediately only if the chat was NOT new/empty
    # If it was new/empty, the whole page will refresh and show it as part of history
    if not is_new_empty_chat:
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

    allowed_standard_models = [k for k in MODEL_MAP if remaining(k)[0] > 0]
    use_fallback_model = False
    chosen_model_key_for_api = None
    model_id_to_use_for_api = None
    max_tokens_for_api = None
    avatar_for_response = "🤖"

    if not allowed_standard_models:
        st.info(f"{FALLBACK_MODEL_EMOJI} All standard model daily quotas exhausted. Using free fallback model.")
        chosen_model_key_for_api = FALLBACK_MODEL_KEY
        model_id_to_use_for_api = FALLBACK_MODEL_ID
        max_tokens_for_api = FALLBACK_MODEL_MAX_TOKENS
        avatar_for_response = FALLBACK_MODEL_EMOJI
        use_fallback_model = True
        logging.info(f"All standard quotas used. Using fallback model: {FALLBACK_MODEL_ID}")
    else:
        if len(allowed_standard_models) == 1:
            chosen_model_key_for_api = allowed_standard_models[0]
            logging.info(f"Only one standard model ('{chosen_model_key_for_api}') has daily quota. Selecting it.")
        else:
            routed_key = route_choice(prompt, allowed_standard_models)
            logging.info(f"Router selected model: '{routed_key}'.")
            chosen_model_key_for_api = routed_key
        model_id_to_use_for_api = MODEL_MAP[chosen_model_key_for_api]
        max_tokens_for_api = MAX_TOKENS[chosen_model_key_for_api]
        avatar_for_response = EMOJI[chosen_model_key_for_api]

    # For new/empty chats, the assistant message will appear after rerun.
    # For existing chats, it streams live.
    if not is_new_empty_chat:
        with st.chat_message("assistant", avatar=avatar_for_response):
            response_placeholder, full_response_content = st.empty(), ""
            api_call_ok = True
            for chunk, error_message in streamed(model_id_to_use_for_api, chat_history, max_tokens_for_api):
                if error_message:
                    full_response_content = f"❗ **API Error**: {error_message}"
                    response_placeholder.error(full_response_content) 
                    api_call_ok = False; break
                if chunk:
                    full_response_content += chunk
                    response_placeholder.markdown(full_response_content + "▌")
            response_placeholder.markdown(full_response_content)
    else: # For new chats, generate response but don't display it live, it will show on rerun
        full_response_content = ""
        api_call_ok = True
        for chunk, error_message in streamed(model_id_to_use_for_api, chat_history, max_tokens_for_api):
            if error_message:
                full_response_content = f"❗ **API Error**: {error_message}"
                api_call_ok = False; break
            if chunk:
                full_response_content += chunk
        # full_response_content is now populated

    chat_history.append({"role":"assistant","content":full_response_content,"model": chosen_model_key_for_api})

    if api_call_ok:
        if not use_fallback_model: record_use(chosen_model_key_for_api)
        
        if sessions[current_sid]["title"] == "New chat" and sessions[current_sid]["messages"]: # Should be true if len(messages) >= 2 now
            sessions[current_sid]["title"] = _autoname(prompt)
            _delete_unused_blank_sessions(keep_sid=current_sid)

    _save(SESS_FILE, sessions)
    st.rerun()

# ───────────────────────── Self-Relaunch ──────────────────────
if __name__ == "__main__" and os.getenv("_IS_STRL") != "1":
    os.environ["_IS_STRL"] = "1"
    port = os.getenv("PORT", "8501")
    cmd = [sys.executable, "-m", "streamlit", "run", __file__, "--server.port", port, "--server.address", "0.0.0.0"]
    logging.info(f"Relaunching with Streamlit: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)
