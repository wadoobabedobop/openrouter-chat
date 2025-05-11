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
• In-app API Key configuration (via Settings panel or initial setup)
"""

# ------------------------- Imports -------------------------
import json, logging, os, sys, subprocess, time, requests
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo
import streamlit as st

# -------------------------- Configuration -------------------
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
    "A": "🌟 (gemini-2.5-pro-preview) - top-quality, creative, expensive.",
    "B": "🔷 (o4-mini) - mid-stakes reasoning, cost-effective.",
    "C": "🟥 (chatgpt-4o-latest) - polished/empathetic, pricier.",
    "D": "🟢 (deepseek-r1) - cheap factual reasoning.",
    "F": "🌀 (gemini-2.5-flash-preview) - quick, free-tier, general purpose."
}

TZ = ZoneInfo("Australia/Sydney")
DATA_DIR   = Path(__file__).parent
SESS_FILE  = DATA_DIR / "chat_sessions.json"
QUOTA_FILE = DATA_DIR / "quotas.json"
CONFIG_FILE = DATA_DIR / "app_config.json" # For storing API key and other app settings


# ------------------------ Helper Functions -----------------------

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


# --------------------- Quota Management ------------------------

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

quota = _load_quota() # This is loaded once at startup. API key not needed for this part.

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


# --------------------- Session Management ------------------------

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

sessions = _load(SESS_FILE, {}) # Loaded once at startup. API key not needed.

def _new_sid():
    _delete_unused_blank_sessions(keep_sid=None)
    sid = str(int(time.time() * 1000))
    sessions[sid] = {"title": "New chat", "messages": []}
    return sid

def _autoname(seed: str) -> str:
    words = seed.strip().split()
    cand = " ".join(words[:3]) or "Chat"
    return (cand[:25] + "…" if len(cand) > 25 else cand


# -------------------------- Logging ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout
)

# Helper function to check API key syntactic validity
def is_api_key_valid(api_key_value):
    return api_key_value and isinstance(api_key_value, str) and api_key_value.startswith("sk-or-")

# -------------------------- API Calls --------------------------
def api_post(payload: dict, *, stream: bool=False, timeout: int=DEFAULT_TIMEOUT):
    active_api_key = st.session_state.get("openrouter_api_key")
    if not is_api_key_valid(active_api_key): # Should be caught by main app logic, but defensive check
        st.session_state.api_key_auth_failed = True # Ensure state reflects this
        raise ValueError("OpenRouter API Key is not set or syntactically invalid. Configure in Settings.")

    headers = {
        "Authorization": f"Bearer {active_api_key}",
        "Content-Type":  "application/json"
    }
    logging.info(f"POST /chat/completions - model={payload.get('model')}, stream={stream}, max_tokens={payload.get('max_tokens')}")
    try:
        response = requests.post(
            f"{OPENROUTER_API_BASE}/chat/completions",
            headers=headers, json=payload, stream=stream, timeout=timeout
        )
        response.raise_for_status()
        return response
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            st.session_state.api_key_auth_failed = True
            logging.error(f"API POST failed with 401: {e.response.text}")
        raise # Re-raise the original error to be handled by caller


def streamed(model: str, messages: list, max_tokens_out: int):
    payload = {
        "model":      model,
        "messages":   messages,
        "stream":     True,
        "max_tokens": max_tokens_out
    }
    try:
        with api_post(payload, stream=True) as r: # api_post can raise ValueError or HTTPError (401 sets flag)
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
        yield None, str(ve) # api_key_auth_failed should be set by api_post
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        text = e.response.text
        logging.error(f"Stream HTTPError {status_code}: {text}")
        # api_key_auth_failed flag is set by api_post if it's 401
        yield None, f"HTTP {status_code}: {text}"
    except Exception as e:
        logging.error(f"Streamed API call failed: {e}")
        yield None, f"Failed to connect or make request: {e}"


# ------------------------- Model Routing ------------------------
def route_choice(user_msg: str, allowed: list[str]) -> str:
    # API key syntactic validity is checked by api_post.
    # api_key_auth_failed will be set by api_post if 401 occurs.
    fallback_choice = "F" if "F" in allowed else (allowed[0] if allowed else "F")
    if not allowed:
        logging.warning("route_choice called with empty allowed list. Defaulting to 'F' or first available model.")
        return "F" if "F" in MODEL_MAP else (list(MODEL_MAP.keys())[0] if MODEL_MAP else "F")

    if len(allowed) == 1:
        logging.info(f"Router: Only one model allowed {allowed[0]}, selecting it directly.")
        return allowed[0]

    system_lines = [
        "You are an intelligent model-routing assistant.",
        "Select ONLY one letter from the following available models:",
    ]
    for k in allowed:
        if k in MODEL_DESCRIPTIONS: system_lines.append(f"- {k}: {MODEL_DESCRIPTIONS[k]}")
        elif k in MODEL_MAP: system_lines.append(f"- {k}: (Model {MODEL_MAP[k]})")

    system_lines.extend([
        "Based on the user's query, choose the letter that best balances quality, speed, and cost-sensitivity.",
        "Respond with ONLY the single capital letter. No extra text."
    ])
    router_messages = [{"role": "system", "content": "\n".join(system_lines)}, {"role": "user",   "content": user_msg}]
    payload_r = {"model": ROUTER_MODEL_ID, "messages": router_messages, "max_tokens": 10}

    try:
        r = api_post(payload_r) # api_post handles 401 by setting flag and re-raising
        text = r.json()["choices"][0]["message"]["content"].strip().upper()
        logging.info(f"Router raw response: {text}")
        for ch in text:
            if ch in allowed: return ch
    except ValueError as ve:
        logging.error(f"ValueError during router call (likely API key issue): {ve}")
    except requests.exceptions.HTTPError as e:
        # Flag st.session_state.api_key_auth_failed is set in api_post if 401
        logging.error(f"Router call HTTPError {e.response.status_code}: {e.response.text}")
    except Exception as e:
        logging.error(f"Router call error: {e}")

    logging.warning(f"Router fallback to model: {fallback_choice}")
    return fallback_choice

# --------------------- Credits Endpoint ------------------------
def get_credits():
    active_api_key = st.session_state.get("openrouter_api_key")
    if not is_api_key_valid(active_api_key):
        # This case should ideally be caught by the main app gating logic.
        # If reached, it means the key became syntactically invalid after passing initial checks.
        st.session_state.api_key_auth_failed = True # Treat as auth failure to go to setup
        logging.warning("get_credits: API Key is not syntactically valid.")
        return None, None, None
    try:
        r = requests.get(
            f"{OPENROUTER_API_BASE}/credits",
            headers={"Authorization": f"Bearer {active_api_key}"},
            timeout=10
        )
        r.raise_for_status()
        d = r.json()["data"]
        # If successful, ensure any prior auth failure flag related to this key is cleared
        # This is mainly for the setup page validation. In normal app flow, it's less critical here.
        # st.session_state.api_key_auth_failed = False # Controversial: should only be reset on explicit new key entry.
        return d["total_credits"], d["total_usage"], d["total_credits"] - d["total_usage"]
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        if status_code == 401:
            st.session_state.api_key_auth_failed = True
            logging.warning(f"Could not fetch /credits: HTTP {status_code} Unauthorized. {e.response.text}")
        else:
            logging.warning(f"Could not fetch /credits: HTTP {status_code}. {e.response.text}")
        return None, None, None
    except Exception as e:
        logging.warning(f"Could not fetch /credits: {e}")
        return None, None, None

# ------------------------- UI Styling --------------------------
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


# ---------------- API Key State Initialization ------------------
if "openrouter_api_key" not in st.session_state:
    app_conf = _load_app_config()
    st.session_state.openrouter_api_key = app_conf.get("openrouter_api_key", None)
if "api_key_auth_failed" not in st.session_state:
    st.session_state.api_key_auth_failed = False

# Determine if app should be in API key setup mode
api_key_is_syntactically_valid = is_api_key_valid(st.session_state.get("openrouter_api_key"))
app_requires_api_key_setup = not api_key_is_syntactically_valid or st.session_state.api_key_auth_failed


# ------------------- Main Application Rendering ------------------
if app_requires_api_key_setup:
    # --- RENDER API KEY SETUP PAGE ---
    st.set_page_config(page_title="OpenRouter API Key Setup", layout="centered")
    load_custom_css()

    st.title("🔒 OpenRouter API Key Required")
    st.markdown("---")

    if not api_key_is_syntactically_valid and st.session_state.get("openrouter_api_key") is not None:
        st.error("The configured API Key is invalid. It must start with 'sk-or-'.")
    elif st.session_state.api_key_auth_failed:
        st.error("API Key authentication failed. The key might be incorrect, revoked, or lack permissions/credits. Please verify and re-enter.")
    else: # Key is None
        st.info("Please configure your OpenRouter API Key to use the application.")

    st.markdown(
        "You can obtain an API Key from [OpenRouter.ai Settings](https://openrouter.ai/keys). "
        "Enter it below to unlock application features."
    )

    key_for_input_field = st.session_state.get("openrouter_api_key", "")
    new_key_input_val = st.text_input(
        "Enter OpenRouter API Key", type="password", key="api_key_setup_input",
        value=key_for_input_field if st.session_state.api_key_auth_failed else "", # Clear if auth failed for previous entry
        placeholder="sk-or-..."
    )

    if st.button("Save and Validate API Key", key="save_api_key_setup_button", use_container_width=True, type="primary"):
        if is_api_key_valid(new_key_input_val):
            st.session_state.openrouter_api_key = new_key_input_val # Update session with the key to be tested
            _save_app_config(new_key_input_val)                   # Save to config file

            st.session_state.api_key_auth_failed = False # Reset flag before attempting validation

            # Attempt to validate the NEW key by fetching credits
            # get_credits() uses st.session_state.openrouter_api_key, which we just updated.
            # It will set st.session_state.api_key_auth_failed to True if it encounters a 401.
            with st.spinner("Validating API Key..."):
                fetched_credits_data = get_credits()

            if st.session_state.api_key_auth_failed:
                 st.error("Authentication failed with the provided API Key. Please ensure it is correct, active, and has necessary permissions/credits.")
            else:
                st.success("API Key saved and validated! Initializing application...")
                if "credits" not in st.session_state: st.session_state.credits = {} # Ensure credits dict exists
                st.session_state.credits = dict(zip(("total", "used", "remaining"), fetched_credits_data or (0,0,0))) # Default on failure
                st.session_state.credits_ts = time.time()
                time.sleep(1.5)
                st.rerun() # Proceed to full app
        elif not new_key_input_val:
            st.warning("API Key field cannot be empty.")
        else:
            st.error("Invalid API key format. It must start with 'sk-or-'.")
    st.markdown("---")
    st.caption("Your API key is stored locally in `app_config.json` and not transmitted elsewhere except to OpenRouter.")

else:
    # --- RENDER FULL APPLICATION ---
    st.set_page_config(
        page_title="OpenRouter Chat",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    load_custom_css()

    # Initialize settings_panel_open state (only if API key is valid and we are in full app mode)
    if "settings_panel_open" not in st.session_state:
        st.session_state.settings_panel_open = False

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


    # Initialize credits if not already set (e.g., first run after successful key setup)
    # or if API key might have changed and needs re-fetch.
    # The get_credits call itself handles setting api_key_auth_failed if 401 occurs.
    if "credits" not in st.session_state or time.time() - st.session_state.get("
