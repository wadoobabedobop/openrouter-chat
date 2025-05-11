#!/usr/bin/env python3
"""
OpenRouter Streamlit Chat — Full Edition
• Persistent chat sessions
• Daily/weekly/monthly quotas
• Pretty ‘token-jar’ gauges (fixed at top)
• Detailed model-routing panel (Mistral router)
• Live credit/usage stats (GET /credits)
• Auto-titling of new chats
• Comprehensive logging
• Self-relaunch under python main.py
• In-app API Key configuration (via Settings panel)
"""

# ───────────────────────── Imports ─────────────────────────
import json, logging, os, sys, subprocess, time, requests
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo
import streamlit as st

# ────────────────────────── Configuration ───────────────────
# OPENROUTER_API_KEY  is now managed via app_config.json and st.session_state
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
CONFIG_FILE = DATA_DIR / "app_config.json" # For storing API key and other app settings


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

def _load_app_config():
    return _load(CONFIG_FILE, {})

def _save_app_config(api_key_value: str):
    config_data = _load_app_config() # Load existing to preserve other settings if any
    config_data["openrouter_api_key"] = api_key_value
    _save(CONFIG_FILE, config_data)


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

# Helper function to check API key validity
def is_api_key_valid(api_key_value):
    return api_key_value and isinstance(api_key_value, str) and api_key_value.startswith("sk-or-")

# ────────────────────────── API Calls ──────────────────────────
def api_post(payload: dict, *, stream: bool=False, timeout: int=DEFAULT_TIMEOUT):
    active_api_key = st.session_state.get("openrouter_api_key")
    if not is_api_key_valid(active_api_key):
        raise ValueError("OpenRouter API Key is not set or invalid in session state. Please configure it in Settings.")

    headers = {
        "Authorization": f"Bearer {active_api_key}",
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
    try:
        with api_post(payload, stream=True) as r:
            try:
                r.raise_for_status()
            except requests.exceptions.HTTPError as e:
                text = r.text
                status_code = e.response.status_code
                logging.error(f"Stream HTTPError {status_code}: {text}")
                if status_code == 401:
                     yield None, f"HTTP {status_code}: Unauthorized. Check your API Key. Details: {text}"
                else:
                    yield None, f"HTTP {status_code}: {text}"
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
    except ValueError as ve: # Catch API key not found/invalid from api_post
        logging.error(f"ValueError during streamed call setup: {ve}")
        yield None, str(ve)
    except Exception as e: # Catch broader exceptions like connection errors
        logging.error(f"Streamed API call failed before request: {e}")
        yield None, f"Failed to connect or make request: {e}"


# ───────────────────────── Model Routing ───────────────────────
def route_choice(user_msg: str, allowed: list[str]) -> str:
    active_api_key = st.session_state.get("openrouter_api_key")
    fallback_choice = "F" if "F" in allowed else (allowed[0] if allowed else "F") # Ensure fallback_choice is valid

    # Validate fallback_choice against MODEL_MAP if necessary for non-F cases
    if not allowed:
        logging.warning("route_choice called with empty allowed list. Defaulting to 'F' or first available model.")
        return "F" if "F" in MODEL_MAP else (list(MODEL_MAP.keys())[0] if MODEL_MAP else "F") # Final fallback

    if not is_api_key_valid(active_api_key):
        logging.warning(f"Router: API Key not set or invalid. Cannot make router call. Falling back to: {fallback_choice}")
        return fallback_choice

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
            logging.warning(f"Model key {k} found in 'allowed' but not in MODEL_DESCRIPTIONS for router prompt.")
            # Optionally, provide a generic description if k is in MODEL_MAP
            if k in MODEL_MAP:
                 system_lines.append(f"- {k}: (Model {MODEL_MAP[k]})")

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
        r = api_post(payload_r) # This will use session_state key and can raise ValueError
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip().upper()
        logging.info(f"Router raw response: {text}")
        for ch in text:
            if ch in allowed: return ch
    except ValueError as ve: # Catch API key not found/invalid from api_post
        logging.error(f"ValueError during router call setup: {ve}")
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        if status_code == 401:
            logging.error(f"Router call HTTPError {status_code}: Unauthorized. Check API Key. {e.response.text}")
        else:
            logging.error(f"Router call HTTPError {status_code}: {e.response.text}")
    except Exception as e:
        logging.error(f"Router call error: {e}")

    logging.warning(f"Router fallback to model: {fallback_choice}")
    return fallback_choice

# ───────────────────── Credits Endpoint ───────────────────────
def get_credits():
    active_api_key = st.session_state.get("openrouter_api_key")
    if not is_api_key_valid(active_api_key):
        logging.warning("Could not fetch /credits: API Key not set or invalid.")
        return None, None, None
    try:
        r = requests.get(
            f"{OPENROUTER_API_BASE}/credits",
            headers={"Authorization": f"Bearer {active_api_key}"},
            timeout=10
        )
        r.raise_for_status()
        d = r.json()["data"]
        return d["total_credits"], d["total_usage"], d["total_credits"] - d["total_usage"]
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        if status_code == 401:
            logging.warning(f"Could not fetch /credits: HTTP {status_code} Unauthorized. Check API Key. {e.response.text}")
        else:
            logging.warning(f"Could not fetch /credits: HTTP {status_code}. {e.response.text}")
        return None, None, None
    except Exception as e:
        logging.warning(f"Could not fetch /credits: {e}")
        return None, None, None

# ───────────────────────── UI Styling ──────────────────────────
def load_custom_css():
    css = """
    <style>
        /* General Styles */
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: var(--secondary-background-color); /* Adapts to theme */
            padding: 1.5rem 1rem;
        }

        /* Sidebar Image (Logo) */
        [data-testid="stSidebar"] .stImage {
            margin-right: 12px;
        }
        [data-testid="stSidebar"] .stImage > img {
            border-radius: 50%;
            box-shadow: 0 2px 6px var(--shadow); /* Theme-aware shadow */
            width: 50px !important;
            height: 50px !important;
        }
        /* Sidebar Title */
        [data-testid="stSidebar"] h1 { /* Targets st.title in sidebar */
            font-size: 1.6rem !important;
            color: var(--primary); /* Use Streamlit's primary color */
            font-weight: 600;
            margin-bottom: 0; /* Adjust if st.columns adds too much space */
            padding-top: 0.3rem; /* Align better with image in columns */
        }

        /* Sidebar Subheaders */
        [data-testid="stSidebar"] h3 { /* Targets st.subheader */
            font-size: 0.9rem !important;
            text-transform: uppercase;
            font-weight: 600;
            color: var(--text-color-secondary); /* Adapts to theme, more subtle */
            margin-top: 1.5rem;
            margin-bottom: 0.75rem;
        }


        /* Button Styling (General for Sidebar - for session list) */
        [data-testid="stSidebar"] .stButton > button {
            border-radius: 8px;
            border: 1px solid var(--border-color);
            padding: 0.5em 1em;
            font-size: 0.95em;
            font-weight: 500;
            font-family: inherit;
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            cursor: pointer;
            transition: border-color 0.2s, background-color 0.2s, box-shadow 0.2s;
            width: 100%;
            margin-bottom: 0.3rem;
            text-align: left;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            border-color: var(--primary);
            background-color: color-mix(in srgb, var(--primary) 10%, var(--secondary-background-color));
            box-shadow: 0 1px 3px var(--shadow);
        }
        [data-testid="stSidebar"] .stButton > button:focus,
        [data-testid="stSidebar"] .stButton > button:focus-visible {
            outline: 2px auto var(--primary);
            outline-offset: 2px;
        }
        [data-testid="stSidebar"] .stButton > button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        /* Specific "New Chat" button - targeted by its key */
        [data-testid="stSidebar"] [data-testid="stButton-new_chat_button_top"] > button {
             background-color: var(--primary);
             color: white;
             border-color: var(--primary);
        }
        [data-testid="stSidebar"] [data-testid="stButton-new_chat_button_top"] > button:hover {
             filter: brightness(90%);
             border-color: var(--primary);
        }


        /* Custom Token Jar Styling */
        .token-jar-container {
            width: 100%;
            max-width: 55px;
            margin: 0 auto 0.5rem auto;
            text-align: center;
            font-family: inherit;
        }
        .token-jar {
            height: 60px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            background: var(--secondary-background-color);
            position: relative;
            overflow: hidden;
            box-shadow: inset 0 1px 2px var(--shadow-sm, rgba(0,0,0,0.05));
            margin-bottom: 4px;
        }
        .token-jar-fill {
            position: absolute;
            bottom: 0;
            width: 100%;
            transition: height 0.3s ease-in-out, background-color 0.3s ease-in-out;
        }
        .token-jar-emoji {
            position: absolute;
            top: 6px;
            width: 100%;
            font-size: 18px;
            line-height: 1;
        }
        .token-jar-key {
            position: absolute;
            bottom: 6px;
            width: 100%;
            font-size: 11px;
            font-weight: 600;
            color: var(--text-color);
            opacity: 0.8;
            line-height: 1;
        }
        .token-jar-remaining {
            display: block;
            margin-top: 2px;
            font-size: 11px;
            font-weight: 600;
            color: var(--text-color);
            opacity: 0.9;
            line-height: 1;
        }

        /* Expander Styling */
        .stExpander {
            border: 1px solid var(--border-color);
            border-radius: 8px;
            margin-bottom: 1rem;
            background-color: var(--background-color-primary);
        }
        .stExpander header {
            font-weight: 600;
            font-size: 0.95rem;
            padding: 0.6rem 1rem !important;
            background-color: var(--secondary-background-color);
            border-bottom: 1px solid var(--border-color);
            border-top-left-radius: 7px;
            border-top-right-radius: 7px;
            color: var(--text-color);
        }
        .stExpander header:hover {
            background-color: color-mix(in srgb, var(--text-color) 5%, var(--secondary-background-color));
        }
        .stExpander div[data-testid="stExpanderDetails"] {
             padding: 0.75rem 1rem;
             background-color: var(--background-color-primary);
        }

        /* Chat Message Styling */
        [data-testid="stChatMessage"] {
            border-radius: 12px;
            padding: 14px 20px;
            margin-bottom: 12px;
            box-shadow: 0 2px 5px var(--shadow);
            border: 1px solid transparent;
        }

        html[data-theme="light"] [data-testid="stChatMessage"][data-testid^="stChatMessageUser"] {
            background-color: #E3F2FD;
            border-left: 3px solid #1E88E5;
            color: #0D47A1;
        }
        html[data-theme="dark"] [data-testid="stChatMessage"][data-testid^="stChatMessageUser"] {
            background-color: color-mix(in srgb, var(--primary) 15%, var(--secondary-background-color));
            border-left: 3px solid var(--primary);
            color: var(--text-color);
        }

        html[data-theme="light"] [data-testid="stChatMessage"][data-testid^="stChatMessageAssistant"] {
            background-color: #f9f9f9;
            border-left: 3px solid #757575;
            color: var(--color-gray-80, #333);
        }
        html[data-theme="dark"] [data-testid="stChatMessage"][data-testid^="stChatMessageAssistant"] {
            background-color: var(--secondary-background-color);
            border-left: 3px solid var(--color-gray-60, #888);
            color: var(--text-color);
        }

        [data-testid="stChatMessage"] .stMarkdown p {
            margin-bottom: 0.3rem;
            line-height: 1.5;
        }

        hr {
          margin-top: 1.5rem;
          margin-bottom: 1.5rem;
          border: 0;
          border-top: 1px solid var(--border-color);
        }
        /* Style for the settings panel container */
        .settings-panel {
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            margin-top: 0.5rem;
            margin-bottom: 1rem;
            background-color: var(--secondary-background-color); /* Or var(--background-color-primary) */
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

# Initialize API key in session state from config file
if "openrouter_api_key" not in st.session_state:
    app_conf = _load_app_config()
    st.session_state.openrouter_api_key = app_conf.get("openrouter_api_key", None)

# Initialize settings_panel_open state
if "settings_panel_open" not in st.session_state:
    st.session_state.settings_panel_open = not is_api_key_valid(st.session_state.get("openrouter_api_key"))

load_custom_css()

# Initial SID Management and Cleanup
needs_save_and_rerun_on_startup = False
if "sid" not in st.session_state:
    st.session_state.sid = _new_sid()
    needs_save_and_rerun_on_startup = True
elif st.session_state.sid not in sessions:
    logging.warning(f"Session ID {st.session_state.sid} from state not found. Creating a new chat.")
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
        get_credits() # This will use API key from session_state
    ))
    st.session_state.credits_ts = time.time()


# ───────────────────────── Sidebar ─────────────────────────────
with st.sidebar:
    # --- Settings Panel Logic ---
    if st.button("⚙️ Settings", key="toggle_settings_button", use_container_width=True):
        st.session_state.settings_panel_open = not st.session_state.get("settings_panel_open", False)
        # Implicit rerun on button click will show/hide the panel

    if st.session_state.get("settings_panel_open"):
        st.markdown("<div class='settings-panel'>", unsafe_allow_html=True)
        st.subheader("API Key Configuration")
        
        current_api_key_in_panel = st.session_state.get("openrouter_api_key")
        key_display = "Not set"
        
        if current_api_key_in_panel and isinstance(current_api_key_in_panel, str): # Check type
            key_display = f"••••••••{current_api_key_in_panel[-4:]}" if len(current_api_key_in_panel) > 8 else "••••"
        st.caption(f"Current key: {key_display}")

        if current_api_key_in_panel and not is_api_key_valid(current_api_key_in_panel):
             st.warning("The saved API key appears invalid. Please enter a valid key (starts with 'sk-or-').")

        new_key_input = st.text_input(
            "Enter new OpenRouter API Key", type="password", key="api_key_settings_input", placeholder="sk-or-..."
        )
        if st.button("Save API Key", key="save_api_key_settings_button", use_container_width=True):
            if is_api_key_valid(new_key_input):
                st.session_state.openrouter_api_key = new_key_input
                _save_app_config(new_key_input)
                st.success("API Key saved!")
                # Refresh credits immediately after saving a valid key
                st.session_state.credits = dict(zip(("total","used","remaining"), get_credits()))
                st.session_state.credits_ts = time.time()
                st.session_state.settings_panel_open = False # Close panel on success
                time.sleep(0.5) # Give user time to see success message
                st.rerun()
            elif not new_key_input:
                st.warning("Please enter an API key.")
            else: # new_key_input is present but invalid
                st.error("Invalid API key format. It should start with 'sk-or-'.")
        
        if st.button("Close Settings Panel", key="close_settings_panel_button", use_container_width=False):
            st.session_state.settings_panel_open = False
            st.rerun() 
        st.markdown("</div>", unsafe_allow_html=True) # End styled container
        st.divider() # Divider after settings panel if it was open
    # --- End Settings Panel ---

    # Logo and Title
    logo_title_cols = st.columns([1, 4], gap="small") # Adjust ratio and gap as needed
    with logo_title_cols[0]:
        st.image("https://avatars.githubusercontent.com/u/130328222?s=200&v=4", width=50)
    with logo_title_cols[1]:
        st.title("OpenRouter Chat") # CSS for h1 in sidebar will style this
    st.divider()


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
            display_title = f"🔹 {display_title}"

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
                ("total","used","remaining"), get_credits() # Uses session_state key
            ))
            st.session_state.credits_ts = time.time()
            st.rerun()
        
        current_api_key_for_credits = st.session_state.get("openrouter_api_key")
        if not is_api_key_valid(current_api_key_for_credits):
            st.warning("Set a valid API Key to fetch credits (via ⚙️ Settings).")
        elif tot is None: # Key is set, but credits couldn't be fetched
            st.warning("Could not fetch credits. Check API key validity in OpenRouter or network connection.")
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

for msg_idx, msg in enumerate(chat_history):
    role = msg["role"]
    avatar_for_display = "👤"
    if role == "assistant":
        model_key_in_message = msg.get("model")
        if model_key_in_message == FALLBACK_MODEL_KEY: avatar_for_display = FALLBACK_MODEL_EMOJI
        elif model_key_in_message in EMOJI: avatar_for_display = EMOJI[model_key_in_message]
        else: avatar_for_display = EMOJI.get("F", "🤖") 

    with st.chat_message(role, avatar=avatar_for_display):
        st.markdown(msg["content"])

# Check for API key before showing chat input
api_key_for_chat_input = st.session_state.get("openrouter_api_key")
chat_input_enabled = is_api_key_valid(api_key_for_chat_input)

if not chat_input_enabled:
    st.warning("👋 Please set your OpenRouter API Key via '⚙️ Settings' in the sidebar to start chatting.")

if prompt := st.chat_input("Ask anything…", key=f"chat_input_{current_sid}", disabled=not chat_input_enabled):
    chat_history.append({"role":"user","content":prompt})
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
            logging.info(f"Only one standard model ('{chosen_model_key_for_api}') has daily quota. Selecting it directly.")
        else:
            # route_choice handles invalid API key by returning fallback_choice
            routed_key = route_choice(prompt, allowed_standard_models) 
            logging.info(f"Router selected model: '{routed_key}'.")
            chosen_model_key_for_api = routed_key

        # Check if router fell back or returned something not in MODEL_MAP
        if chosen_model_key_for_api not in MODEL_MAP:
            logging.warning(f"Chosen model key '{chosen_model_key_for_api}' is not in MODEL_MAP. Will use fallback.")
            st.info(f"{FALLBACK_MODEL_EMOJI} Model selection issue or routing fallback. Using free fallback model.")
            chosen_model_key_for_api = FALLBACK_MODEL_KEY
            model_id_to_use_for_api = FALLBACK_MODEL_ID
            max_tokens_for_api = FALLBACK_MODEL_MAX_TOKENS
            avatar_for_response = FALLBACK_MODEL_EMOJI
            use_fallback_model = True
        else: # chosen_model_key_for_api is valid and in MODEL_MAP
            model_id_to_use_for_api = MODEL_MAP[chosen_model_key_for_api]
            max_tokens_for_api = MAX_TOKENS[chosen_model_key_for_api]
            avatar_for_response = EMOJI[chosen_model_key_for_api]


    with st.chat_message("assistant", avatar=avatar_for_response):
        response_placeholder, full_response_content = st.empty(), ""
        api_call_ok = True
        # `streamed` will use the API key from session_state and handle related ValueErrors
        for chunk, error_message in streamed(model_id_to_use_for_api, chat_history, max_tokens_for_api):
            if error_message:
                full_response_content = f"❗ **API Error**: {error_message}"
                response_placeholder.error(full_response_content)
                api_call_ok = False; break
            if chunk:
                full_response_content += chunk
                response_placeholder.markdown(full_response_content + "▌")
        response_placeholder.markdown(full_response_content)

    chat_history.append({"role":"assistant","content":full_response_content,"model": chosen_model_key_for_api})

    if api_call_ok:
        if not use_fallback_model and chosen_model_key_for_api in MODEL_MAP:
             record_use(chosen_model_key_for_api)

        if sessions[current_sid]["title"] == "New chat" and sessions[current_sid]["messages"]:
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
