#!/usr/bin/env python3
# -*- coding: utf-8 -*- # THIS SHOULD BE LINE 2
"""
OpenRouter Streamlit Chat — Full Edition
• Persistent chat sessions
• Daily/weekly/monthly quotas
• Pretty ‘token-jar’ gauges (fixed at top)
• Detailed model-routing panel (Mistral router)
• Live credit/usage stats (GET /credits)
• Auto-titling of new chats
• Comprehensive logging
• In-app API Key configuration (via Settings panel or initial setup)
"""

# ------------------------- Imports ------------------------- # USE STANDARD HYPHENS AND '#' FOR COMMENTS
import json, logging, os, sys, subprocess, time, requests
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo # Python 3.9+
import streamlit as st

# -------------------------- Configuration --------------------------- # USE STANDARD HYPHENS AND '#'
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


# ------------------------ Helper Functions -----------------------

def _load(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default

def _save(path: Path, obj):
     try:
       path.write_text(json.dumps(obj, indent=2))
     except Exception as e:
       logging.error(f"Failed to save file {path}: {e}")

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
            # Clean out keys for models that are no longer defined in MODEL_MAP
            keys_to_remove = [k for k in current_usage_dict if k not in MODEL_MAP]
            for k_rem in keys_to_remove:
                try:
                  del current_usage_dict[k_rem]
                  logging.info(f"Removed old model key '{k_rem}' from quota usage '{period_usage_key}'.")
                except KeyError:
                   pass # Ignore if already gone
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


# --------------------- Session Management -----------------------

def _delete_unused_blank_sessions(keep_sid: str = None):
    sids_to_delete = []
    # Make a copy of items to iterate over, allowing deletion from original
    for sid, data in list(sessions.items()):
        if sid == keep_sid:
            continue
        if data.get("title") == "New chat" and not data.get("messages"):
            sids_to_delete.append(sid)

    if sids_to_delete:
        for sid_del in sids_to_delete:
            logging.info(f"Auto-deleting blank session: {sid_del}")
            try:
               del sessions[sid_del]
            except KeyError:
               logging.warning(f"Session {sid_del} already deleted, skipping.")
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
    return (cand[:25] + "…") if len(cand) > 25 else cand


# --------------------------- Logging ----------------------------
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
    logging.info(f"POST /chat/completions -> model={payload.get('model')}, stream={stream}, max_tokens={payload.get('max_tokens')}")
    try:
        response = requests.post(
            f"{OPENROUTER_API_BASE}/chat/completions",
            headers=headers, json=payload, stream=stream, timeout=timeout
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            st.session_state.api_key_auth_failed = True # <<< SET FLAG ON 401
            logging.error(f"API POST failed with 401 (Unauthorized): {e.response.text}")
        # Log other HTTP errors as well for more context
        else:
            logging.error(f"API POST failed with {e.response.status_code}: {e.response.text}")
        raise # Re-raise the original error to be handled by caller


def streamed(model: str, messages: list, max_tokens_out: int):
    payload = {
        "model":      model,
        "messages":   messages,
        "stream":     True,
        "max_tokens": max_tokens_out
    }
    try:
        # api_post can raise ValueError or HTTPError (401 sets flag)
        # Use a context manager for the response to ensure it's closed
        with api_post(payload, stream=True) as r:
            for line in r.iter_lines():
                if not line: continue
                line_str = line.decode("utf-8")
                if line_str.startswith(": OPENROUTER PROCESSING"):
                   # logging.info(f"OpenRouter PING: {line_str.strip()}")
                    continue
                if not line_str.startswith("data: "):
                    logging.warning(f"Unexpected non-event-stream line: {line_str}") # Log decoded string
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
        yield None, str(ve) # api_key_auth_failed should have been set by api_post
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        text = e.response.text
        logging.error(f"Stream HTTPError {status_code}: {text}")
        # api_key_auth_failed flag is set by api_post if it's 401
        yield None, f"HTTP {status_code}: An error occurred with the API provider. Details: {text}"
    except Exception as e: # Catch broader exceptions like connection errors
        logging.error(f"Streamed API call failed: {e}")
        yield None, f"Failed to connect or make request: {e}"


# ------------------------- Model Routing -----------------------
def route_choice(user_msg: str, allowed: list[str]) -> str:
    # API key syntactic validity is checked by api_post.
    # api_key_auth_failed will be set by api_post if 401 occurs.
    
    # Determine a sensible fallback choice early. If F is allowed, use it. Otherwise, first in allowed.
    # If allowed is empty (shouldn't happen if quotas exist), use F or any model as a last resort.
    if "F" in allowed:
        fallback_choice = "F"
    elif allowed:
        fallback_choice = allowed[0]
    elif "F" in MODEL_MAP: # F might not be in MODEL_MAP if config changes
        fallback_choice = "F"
    elif MODEL_MAP:
        fallback_choice = list(MODEL_MAP.keys())[0]
    else: # No models defined at all
        logging.error("Router: No models available in MODEL_MAP for fallback.")
        return FALLBACK_MODEL_KEY # Or handle this critical state appropriately

    if not allowed:
        logging.warning(f"route_choice called with empty allowed list. Defaulting to '{fallback_choice}'.")
        return fallback_choice

    if len(allowed) == 1:
        logging.info(f"Router: Only one model allowed ('{allowed[0]}'), selecting it directly.")
        return allowed[0]

    system_lines = [
        "You are an intelligent model-routing assistant.",
        "Select ONLY one letter from the following available models:",
    ]
    for k in allowed:
        if k in MODEL_DESCRIPTIONS:
            system_lines.append(f"- {k}: {MODEL_DESCRIPTIONS[k]}")
        elif k in MODEL_MAP: # Should always be true if k is in allowed from MODEL_MAP keys
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
        r = api_post(payload_r) # api_post handles 401 by setting flag and re-raising
        choice_data = r.json()
        text = choice_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()
        logging.info(f"Router raw response: '{text}'")
        for ch in text: # Iterate through characters in response to find first valid choice
            if ch in allowed:
               logging.info(f"Router selected model: '{ch}'")
               return ch
    except ValueError as ve: # Catch API key not found/invalid from api_post
        logging.error(f"ValueError during router call (likely API key issue): {ve}")
    except requests.exceptions.HTTPError as e:
         # Flag st.session_state.api_key_auth_failed is set in api_post if 401
        logging.error(f"Router call HTTPError {e.response.status_code}: {e.response.text}")
    except (KeyError, IndexError, AttributeError) as je: # Catch issues with JSON structure
        logging.error(f"Router call JSON parsing/structure error: {je}. Response: {choice_data if 'choice_data' in locals() else 'N/A'}")
    except Exception as e:
        logging.error(f"Router call unexpected error: {e}")

    logging.warning(f"Router failed or returned invalid. Falling back to model: {fallback_choice}")
    return fallback_choice

# --------------------- Credits Endpoint -----------------------
def get_credits():
    """Returns (total, used, remaining) or (None, None, None) on failure."""
    active_api_key = st.session_state.get("openrouter_api_key")
    if not is_api_key_valid(active_api_key):
        # Do not set api_key_auth_failed here if only syntactically invalid,
        # as it might be None intentionally before setup.
        # api_post and the main setup logic handle this.
        logging.warning("get_credits: API Key is not syntactically valid or not set.")
        return None, None, None
    try:
        r = requests.get(
            f"{OPENROUTER_API_BASE}/credits",
            headers={"Authorization": f"Bearer {active_api_key}"},
            timeout=10 # Reasonable timeout for credits check
        )
        r.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        d = r.json()["data"]
        # SUCCESS!
        st.session_state.api_key_auth_failed = False # Explicitly clear on success
        return d["total_credits"], d["total_usage"], d["total_credits"] - d["total_usage"]
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        err_text = e.response.text
        if status_code == 401:
            st.session_state.api_key_auth_failed = True # <<< SET FLAG ON 401
            logging.warning(f"Could not fetch /credits: HTTP {status_code} Unauthorized. {err_text}")
        else:
             logging.warning(f"Could not fetch /credits: HTTP {status_code}. {err_text}")
        return None, None, None
    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e: # Broader request/parsing errors
        logging.warning(f"Could not fetch /credits due to network/parsing error: {e}")
        # Potentially a network issue, not necessarily auth_failed unless it was a 401 captured above.
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
        [data-testid="stSidebar"] .stImage { margin-right: 12px; }
        [data-testid="stSidebar"] .stImage > img {
            border-radius: 50%;
            box-shadow: 0 2px 6px var(--shadow); /* Theme-aware shadow */
            width: 50px !important; height: 50px !important;
        }
        /* Sidebar Title */
        [data-testid="stSidebar"] h1 { /* Targets st.title in sidebar */
            font-size: 1.6rem !important; color: var(--primary); /* Use Streamlit's primary color */
            font-weight: 600; margin-bottom: 0; /* Adjust if st.columns adds too much space */
            padding-top: 0.3rem; /* Align better with image in columns */
        }
        /* Sidebar Subheaders */
        [data-testid="stSidebar"] h3 { /* Targets st.subheader */
            font-size: 0.9rem !important; text-transform: uppercase; font-weight: 600;
            color: var(--text-color-secondary); /* Adapts to theme, more subtle */
            margin-top: 1.5rem; margin-bottom: 0.75rem;
        }
        /* Button Styling (General for Sidebar - for session list) */
        [data-testid="stSidebar"] .stButton > button {
            border-radius: 8px; border: 1px solid var(--border-color);
            padding: 0.5em 1em; font-size: 0.95em; font-weight: 500;
            font-family: inherit; background-color: var(--secondary-background-color);
            color: var(--text-color); cursor: pointer;
            transition: border-color 0.2s, background-color 0.2s, box-shadow 0.2s;
            width: 100%; margin-bottom: 0.3rem; text-align: left;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            border-color: var(--primary);
            background-color: color-mix(in srgb, var(--primary) 10%, var(--secondary-background-color));
            box-shadow: 0 1px 3px var(--shadow);
        }
         [data-testid="stSidebar"] .stButton > button:disabled { opacity: 0.6; cursor: not-allowed; }
       /* Specific "New Chat" button - targeted by its key */
        [data-testid="stSidebar"] [data-testid="stButton-new_chat_button_top"] > button {
             background-color: var(--primary); color: white; border-color: var(--primary);
        }
        [data-testid="stSidebar"] [data-testid="stButton-new_chat_button_top"] > button:hover {
             filter: brightness(90%); border-color: var(--primary);
        }
        /* Custom Token Jar Styling */
        .token-jar-container {
            width: 100%; max-width: 55px; margin: 0 auto 0.5rem auto;
            text-align: center; font-family: inherit;
        }
        .token-jar {
            height: 60px; border: 1px solid var(--border-color); border-radius: 8px;
            background: var(--secondary-background-color); position: relative;
            overflow: hidden;
            box-shadow: inset 0 1px 2px var(--shadow-sm, rgba(0,0,0,0.05));
            margin-bottom: 4px;
        }
        .token-jar-fill {
            position: absolute; bottom: 0; width: 100%;
            transition: height 0.3s ease-in-out, background-color 0.3s ease-in-out;
        }
        .token-jar-emoji { position: absolute; top: 6px; width: 100%; font-size: 18px; line-height: 1;}
        .token-jar-key {
             position: absolute; bottom: 6px; width: 100%; font-size: 11px;
             font-weight: 600; color: var(--text-color); opacity: 0.8; line-height: 1;
        }
        .token-jar-remaining {
            display: block; margin-top: 2px; font-size: 11px; font-weight: 600;
             color: var(--text-color); opacity: 0.9; line-height: 1;
        }
        /* Expander Styling */
       .stExpander {
            border: 1px solid var(--border-color); border-radius: 8px;
            margin-bottom: 1rem; background-color: var(--background-color-primary); /* Changed for better contrast */
        }
       .stExpander header {
            font-weight: 600; font-size: 0.95rem; padding: 0.6rem 1rem !important;
            background-color: var(--secondary-background-color);
            border-bottom: 1px solid var(--border-color);
            border-top-left-radius: 7px; border-top-right-radius: 7px; color: var(--text-color);
        }
       .stExpander div[data-testid="stExpanderDetails"] {
             padding: 0.75rem 1rem; background-color: var(--background-color-primary); /* Changed to match expander base */
        }
        /* Chat Message Styling */
        [data-testid="stChatMessage"] {
            border-radius: 12px; padding: 14px 20px; margin-bottom: 12px;
            box-shadow: 0 2px 5px var(--shadow); border: 1px solid transparent; /* Adjusted from var(--border-color) */
        }
        html[data-theme="light"] [data-testid="stChatMessage"][data-testid^="stChatMessageUser"] {
            background-color: #E3F2FD; border-left: 3px solid #1E88E5; color: #0D47A1;
        }
        html[data-theme="dark"] [data-testid="stChatMessage"][data-testid^="stChatMessageUser"] {
            background-color: color-mix(in srgb, var(--primary) 15%, var(--secondary-background-color));
            border-left: 3px solid var(--primary); color: var(--text-color);
        }
         html[data-theme="light"] [data-testid="stChatMessage"][data-testid^="stChatMessageAssistant"] {
             background-color: #f9f9f9; border-left: 3px solid #757575; color: var(--color-gray-80, #333);
         }
        html[data-theme="dark"] [data-testid="stChatMessage"][data-testid^="stChatMessageAssistant"] {
            background-color: var(--secondary-background-color);
             border-left: 3px solid var(--color-gray-60, #888); color: var(--text-color);
        }
        hr {
          margin-top: 1.5rem; margin-bottom: 1.5rem; border: 0;
          border-top: 1px solid var(--border-color);
        }
       .settings-panel {
            border: 1px solid var(--border-color); border-radius: 8px; padding: 1rem;
            margin-top: 0.5rem; margin-bottom: 1rem;
            background-color: var(--secondary-background-color);
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ----------------- API Key State Initialization -------------------
# Load key from config into state if it's not already in state
if "openrouter_api_key" not in st.session_state:
    app_conf = _load_app_config()
    st.session_state.openrouter_api_key = app_conf.get("openrouter_api_key", None)

# Ensure auth failed flag exists
if "api_key_auth_failed" not in st.session_state:
    st.session_state.api_key_auth_failed = False # Default to False

# Determine if app should be in API key setup mode
# This check runs ONCE per rerun.
api_key_is_syntactically_valid = is_api_key_valid(st.session_state.get("openrouter_api_key"))
# App requires setup if key is syntactically invalid OR auth has previously failed with current key
app_requires_api_key_setup = not api_key_is_syntactically_valid or st.session_state.api_key_auth_failed


# -------------------- Main Application Rendering -------------------

if app_requires_api_key_setup:
    # --- RENDER API KEY SETUP PAGE ---
    st.set_page_config(page_title="OpenRouter API Key Setup", layout="centered")
    load_custom_css() # Load CSS for the setup page as well

    st.title("🔒 OpenRouter API Key Required")
    st.markdown("---")

    # Specific error for auth failure
    if st.session_state.api_key_auth_failed:
         st.error("API Key Authentication Failed. The key may be incorrect, revoked, disabled, or lack credits. Please verify your key on OpenRouter.ai and re-enter.")
    # Specific error for syntactically invalid key (but not None, which means it was entered before)
    elif not api_key_is_syntactically_valid and st.session_state.get("openrouter_api_key") is not None:
        st.error("The previously configured API Key has an invalid format. It must start with `sk-or-`.")
    # General info if key is simply None (first run or cleared)
    else:
        st.info("Please configure your OpenRouter API Key to use the application.")

    st.markdown(
        "You can get a key from [OpenRouter.ai Keys](https://openrouter.ai/keys). "
         "Enter it below to continue."
     )
    
    new_key_input_val = st.text_input(
        "Enter OpenRouter API Key", type="password", key="api_key_setup_input",
        value="", # Always start blank on setup page for security and clarity
        placeholder="sk-or-..."
    )

    if st.button("Save and Validate API Key", key="save_api_key_setup_button", use_container_width=True, type="primary"):
        if is_api_key_valid(new_key_input_val):
            st.session_state.openrouter_api_key = new_key_input_val # Update state with the key to be tested
            _save_app_config(new_key_input_val)                   # Save to config file

            st.session_state.api_key_auth_failed = False # Reset flag before attempting validation with new key

            with st.spinner("Validating API Key..."):
                fetched_credits_data = get_credits() # This will set api_key_auth_failed on 401

            if st.session_state.api_key_auth_failed: # Check flag set by get_credits
                 # Error will be shown on the *next* rerun due to app_requires_api_key_setup logic
                 st.error("Authentication failed with the provided API Key. Please check the key and try again.")
                 time.sleep(0.5) # Brief pause to allow user to see message before rerun
                 st.rerun()
            elif fetched_credits_data == (None, None, None): # Other error during get_credits
                st.error("Could not validate API Key. There might be a network issue or an unexpected problem with the API provider. Please try again.")
                # Do not necessarily rerun here; allow user to try again.
            else: # Success
                st.success("API Key saved and validated! Initializing application...")
                if "credits" not in st.session_state: st.session_state.credits = {}
                st.session_state.credits["total"], st.session_state.credits["used"], st.session_state.credits["remaining"] = fetched_credits_data
                st.session_state.credits_ts = time.time()
                time.sleep(1.0) # Pause for user to see success message
                st.rerun() # Proceed to full app
        elif not new_key_input_val:
            st.warning("API Key field cannot be empty.")
        else:
            st.error("Invalid API key format. It must start with 'sk-or-'.")
            
    st.markdown("---")
    st.caption("Your API key is stored locally in `app_config.json` and used only to communicate with the OpenRouter API.")

else:
    # --- RENDER FULL APPLICATION ---
    st.set_page_config(
        page_title="OpenRouter Chat",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    load_custom_css()

    if "settings_panel_open" not in st.session_state:
        st.session_state.settings_panel_open = False

    # --- SID Management ---
    needs_save_session = False # Renamed for clarity
    if "sid" not in st.session_state:
        st.session_state.sid = _new_sid()
        needs_save_session = True
    elif st.session_state.sid not in sessions:
        logging.warning(f"Session ID {st.session_state.sid} from state not found in loaded sessions. Creating a new chat.")
        st.session_state.sid = _new_sid()
        needs_save_session = True # Create new session, so save
        
    if _delete_unused_blank_sessions(keep_sid=st.session_state.sid):
       needs_save_session = True
       
    if needs_save_session:
       _save(SESS_FILE, sessions)
       # Rerun might be too disruptive here if only blank sessions were deleted without changing current SID
       # Only rerun if SID actually changed or was newly created.
       # The _new_sid call already implies a change, so this is likely fine.
       st.rerun()
       
    # --- Credits Initialization / Refresh ---
    if "credits" not in st.session_state: # Initialise if completely missing
         st.session_state.credits = {"total": 0.0, "used": 0.0, "remaining": 0.0} # Default structure
         st.session_state.credits_ts = 0

    # Fetch credits if they are stale (e.g > 1 hr old) OR if they are default zeros (might indicate previous failure)
    credits_are_stale = time.time() - st.session_state.get("credits_ts", 0) > 3600
    credits_are_default = st.session_state.credits.get("total") == 0.0 and \
                          st.session_state.credits.get("used") == 0.0 and \
                          st.session_state.credits.get("remaining") == 0.0 and \
                          st.session_state.credits_ts != 0 # Avoid if truly first run with 0 ts

    if credits_are_stale or credits_are_default:
        logging.info("Refreshing credits (stale or default values).")
        credits_data = get_credits() # This will set api_key_auth_failed on 401
        
        if st.session_state.get("api_key_auth_failed"):
            st.error("API Key authentication failed. Please update your API Key in ⚙️ Settings.")
            # No st.stop() here, allow UI to render so settings can be accessed.
            # The app_requires_api_key_setup will be True on next rerun and force setup page.
            # For now, we might show old credit data or zeros.
            st.rerun() # Force re-evaluation, which will lead to setup page
            st.stop() # Stop current run after triggering rerun
        
        if credits_data != (None, None, None):
            st.session_state.credits["total"], st.session_state.credits["used"], st.session_state.credits["remaining"] = credits_data
            st.session_state.credits_ts = time.time()
        else:
            # If fetching fails but not due to 401, keep old data or log warning.
            # It's already logged in get_credits. Here, we ensure credits_ts is updated to avoid rapid refetching.
            st.session_state.credits_ts = time.time() 
            # Don't overwrite existing valid credit data with zeros if fetch fails for non-auth reason
            if not all(isinstance(st.session_state.credits.get(k), (int,float)) for k in ["total", "used", "remaining"]):
                 st.session_state.credits = {"total": 0.0, "used": 0.0, "remaining": 0.0} # Fallback to zeros if current data is invalid


    # ------------------------- Sidebar -----------------------------
    with st.sidebar:
        # --- Settings Panel (for changing key) ---
        if st.button("⚙️ Settings", key="toggle_settings_button", use_container_width=True):
            st.session_state.settings_panel_open = not st.session_state.get("settings_panel_open", False)

        if st.session_state.get("settings_panel_open"):
            st.markdown("<div class='settings-panel'>", unsafe_allow_html=True)
            st.subheader("API Key Configuration")
            
            current_api_key_in_panel = st.session_state.get("openrouter_api_key")
            if current_api_key_in_panel and len(current_api_key_in_panel) > 8:
                 key_display = f"Current key: `sk-or-...{current_api_key_in_panel[-4:]}`"
            elif current_api_key_in_panel:
                 key_display = "Current key: `sk-or-...` (short key)"
            else:
                 key_display = "Current key: Not set"
            st.caption(key_display)


            new_key_input_sidebar = st.text_input(
                "Enter new OpenRouter API Key (optional)", type="password", key="api_key_sidebar_input", placeholder="sk-or-..."
            )
            if st.button("Save New API Key", key="save_api_key_sidebar_button", use_container_width=True):
                if is_api_key_valid(new_key_input_sidebar):
                    st.session_state.openrouter_api_key = new_key_input_sidebar
                    _save_app_config(new_key_input_sidebar)
                    st.session_state.api_key_auth_failed = False # Reset flag for the new key

                    with st.spinner("Validating new API key..."):
                        credits_data = get_credits() # Uses new key, sets auth_failed on 401
                    
                    if st.session_state.api_key_auth_failed:
                        st.error("New API Key failed authentication. Further actions may require re-setup.")
                        # App will go to setup page on next rerun if auth_failed is true
                    elif credits_data == (None,None,None):
                        st.warning("Could not validate the new API key (network or other API issue). Key is saved, but functionality may be affected.")
                    else: # Success
                        st.success("New API Key saved and validated!")
                        st.session_state.credits["total"],st.session_state.credits["used"],st.session_state.credits["remaining"] = credits_data
                        st.session_state.credits_ts = time.time()
                    
                    st.session_state.settings_panel_open = False # Close panel
                    time.sleep(0.8) # UI Pause
                    st.rerun() # Rerun to update display, or to go to setup page if key failed
                elif not new_key_input_sidebar: # Empty input
                    st.warning("API Key field is empty. No changes made.")
                else: # Syntactically invalid
                    st.error("Invalid API key format. It must start with 'sk-or-'.")
            
            if st.button("Close Settings", key="close_settings_panel_button_sidebar"):
                st.session_state.settings_panel_open = False
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
            st.divider()
        # --- End Settings Panel ---

        logo_title_cols = st.columns([1, 4], gap="small")
        with logo_title_cols[0]: st.image("https://avatars.githubusercontent.com/u/130328222?s=200&v=4", width=50)
        with logo_title_cols[1]: st.title("OpenRouter Chat")
        st.divider()

        st.subheader("Daily Jars (Msgs Left)")
        active_model_keys = sorted(MODEL_MAP.keys())
        cols = st.columns(len(active_model_keys))
        for i, m_key in enumerate(active_model_keys):
            left, _, _ = remaining(m_key); lim, _, _  = PLAN[m_key]
            pct = 1.0 if lim > 900_000 else max(0.0, left / lim if lim > 0 else 0.0) # Avoid div by zero
            fill = int(pct * 100)
            if pct > .5: color = "#4caf50"; elif pct > .25: color = "#ffc107"; else: color = "#f44336"
            # Ensure EMOJI[m_key] exists or provide a fallback
            emoji_char = EMOJI.get(m_key, "❔")
            cols[i].markdown(f"""
                <div class="token-jar-container">
                  <div class="token-jar"><div class="token-jar-fill" style="height:{fill}%; background-color:{color};"></div>
                    <div class="token-jar-emoji">{emoji_char}</div><div class="token-jar-key">{m_key}</div>
                  </div><span class="token-jar-remaining">{'∞' if lim > 900_000 else left}</span></div>""", unsafe_allow_html=True)
        st.divider()

        current_session_is_truly_blank = (st.session_state.sid in sessions and
                                          sessions[st.session_state.sid].get("title") == "New chat" and
                                          not sessions[st.session_state.sid].get("messages"))
                                          
        if st.button("➕ New chat", key="new_chat_button_top", use_container_width=True, disabled=current_session_is_truly_blank):
            old_sid = st.session_state.sid
            st.session_state.sid = _new_sid() # This creates the new session entry
            _delete_unused_blank_sessions(keep_sid=st.session_state.sid) # Clean up old one if it was blank
            _save(SESS_FILE, sessions) # Save changes
            st.rerun()
        
        st.subheader("Chats")
        # Ensure sessions is not empty and sids are valid before sorting
        valid_sids = [s for s in sessions.keys() if isinstance(s, str) and s.isdigit()]
        sorted_sids = sorted(valid_sids, key=lambda s: int(s), reverse=True)
        
        for sid_key in sorted_sids:
            if sid_key not in sessions: continue # Should not happen if valid_sids is correct
            title = sessions[sid_key].get("title", "Untitled")
            display_title = title[:25] + ("…" if len(title) > 25 else "")
            if st.session_state.sid == sid_key: display_title = f"🔹 {display_title}"
            if st.button(display_title, key=f"session_button_{sid_key}", use_container_width=True):
                if st.session_state.sid != sid_key:
                    # Before switching, ensure current blank chat (if any) is deleted if it's not the one we are switching to
                    _delete_unused_blank_sessions(keep_sid=sid_key)
                    st.session_state.sid = sid_key
                    _save(SESS_FILE, sessions) # Save after potential deletion
                    st.rerun()
        st.divider()

        st.subheader("Model-Routing Map")
        st.caption(f"Router: {ROUTER_MODEL_ID}")
        with st.expander("Letters → Models", expanded=False):
            for k_model in sorted(MODEL_MAP.keys()):
                # Ensure all keys exist before formatting
                desc = MODEL_DESCRIPTIONS.get(k_model, MODEL_MAP.get(k_model, "N/A"))
                max_tok = MAX_TOKENS.get(k_model, 0)
                st.markdown(f"**{k_model}**: {desc} (max_out={max_tok:,})")
        st.divider()

        with st.expander("Account stats (credits)", expanded=False):
            if st.button("Refresh Credits", key="refresh_credits_button"):
                 with st.spinner("Refreshing credits..."): # Matched spinner message
                    credits_data = get_credits() # Sets auth_failed on 401
                 if not st.session_state.get("api_key_auth_failed"): # Check flag after get_credits
                    if credits_data != (None,None,None):
                        st.session_state.credits["total"],st.session_state.credits["used"],st.session_state.credits["remaining"] = credits_data
                        st.session_state.credits_ts = time.time()
                        st.success("Credits refreshed!")
                    else:
                        st.warning("Could not refresh credits (network or API issue).")
                 # Rerun will happen, if auth failed, it will go to setup. Otherwise, shows new credits.
                 st.rerun() 
            
            tot = st.session_state.credits.get("total")
            used = st.session_state.credits.get("used")
            rem = st.session_state.credits.get("remaining")

            if tot is None or used is None or rem is None : # Check if credits data is valid
                 st.warning("Could not fetch/display credits. Check network or API key (in Settings).")
            else:
                st.markdown(f"**Remaining:** ${float(rem):.2f} cr")
                st.markdown(f"**Used:** ${float(used):.2f} cr")
            
            ts = st.session_state.get("credits_ts", 0)
            last_updated_str = datetime.fromtimestamp(ts, TZ).strftime('%-d %b, %H:%M:%S') if ts else "N/A"
            st.caption(f"Last updated: {last_updated_str}")


    # ------------------------- Main Chat Panel ---------------------
    # Ensure current_sid exists in sessions, otherwise reset (edge case)
    if st.session_state.sid not in sessions:
        logging.error(f"Current session ID {st.session_state.sid} missing from sessions. Resetting to new chat.")
        st.session_state.sid = _new_sid()
        _save(SESS_FILE, sessions)
        st.rerun()
        st.stop()

    current_sid = st.session_state.sid
    chat_history = sessions[current_sid]["messages"]
    
    for msg in chat_history:
        role = msg.get("role", "assistant") # Default role if missing
        avatar = "👤" if role == "user" else None # Default Streamlit avatar for assistant if not specified

        if role == "assistant":
            m_key = msg.get("model")
            if m_key == FALLBACK_MODEL_KEY:
                avatar = FALLBACK_MODEL_EMOJI
            elif m_key in EMOJI:
                avatar = EMOJI[m_key]
            else: # Default assistant avatar if model key unknown or not in EMOJI map
                avatar = "🤖" 
        
        with st.chat_message(role, avatar=avatar): 
             st.markdown(msg.get("content", "*empty message*")) # Handle empty content

    # Chat Input
    if prompt := st.chat_input("Ask anything…", key=f"chat_input_{current_sid}"):
        chat_history.append({"role":"user","content":prompt})
        with st.chat_message("user", avatar="👤"): st.markdown(prompt)

        # Re-check API key status before making calls
        if not is_api_key_valid(st.session_state.get("openrouter_api_key")) or st.session_state.get("api_key_auth_failed"):
            st.error("API Key is not configured or has failed. Please set it up in ⚙️ Settings.")
            st.rerun()
            st.stop()

        allowed_standard_models = [k for k in MODEL_MAP if remaining(k)[0] > 0] # Daily quota check
        use_fallback, chosen_model_key, model_id_to_use, max_tokens_api, avatar_resp = (
            False, None, None, None, "🤖" # Default avatar
        )
        
        # --- Decide model (Quota -> Router -> Fallback) ---
        if not allowed_standard_models:
            logging.info(f"Using fallback (all quotas used): {FALLBACK_MODEL_ID}")
            st.info(f"{FALLBACK_MODEL_EMOJI} Daily quotas for standard models exhausted. Using free fallback.")
            use_fallback, chosen_model_key, model_id_to_use, max_tokens_api, avatar_resp = (True, FALLBACK_MODEL_KEY, FALLBACK_MODEL_ID, FALLBACK_MODEL_MAX_TOKENS, FALLBACK_MODEL_EMOJI)
        else:
            routed_key = route_choice(prompt, allowed_standard_models)
            # Check if routing itself caused an Auth failure (e.g., router model key invalid)
            if st.session_state.get("api_key_auth_failed"):
                 st.error("API Authentication failed during model routing. Please check your API Key in Settings.")
                 st.rerun() 
                 st.stop()

            if routed_key not in MODEL_MAP or routed_key not in allowed_standard_models: 
                logging.warning(f"Router chose '{routed_key}' (invalid or no quota). Using fallback {FALLBACK_MODEL_ID}.")
                st.warning(f"{FALLBACK_MODEL_EMOJI} Model routing issue or chosen model '{routed_key}' has no quota. Using free fallback.")
                use_fallback, chosen_model_key, model_id_to_use, max_tokens_api, avatar_resp = (True, FALLBACK_MODEL_KEY, FALLBACK_MODEL_ID, FALLBACK_MODEL_MAX_TOKENS, FALLBACK_MODEL_EMOJI)
            else: # Route OK
                chosen_model_key = routed_key
                model_id_to_use = MODEL_MAP[chosen_model_key]
                max_tokens_api = MAX_TOKENS[chosen_model_key]
                avatar_resp = EMOJI.get(chosen_model_key, "🤖")
        
        # --- Stream response ---
        with st.chat_message("assistant", avatar=avatar_resp):
            response_placeholder, full_response = st.empty(), ""
            api_call_ok = True
            
            for chunk, err_msg in streamed(model_id_to_use, chat_history, max_tokens_api):
                if st.session_state.get("api_key_auth_failed"): # streamed sets this on 401
                    full_response = "❗ **API Authentication Error**: Your API Key failed. Please update it in ⚙️ Settings."
                    api_call_ok = False; break
                if err_msg:
                    full_response = f"❗ **API Error**: {err_msg}"
                    api_call_ok = False; break
                if chunk: 
                   full_response += chunk
                   response_placeholder.markdown(full_response + "▌")
                   
            response_placeholder.markdown(full_response) # Final content

        # --- Save state ---
        # Add assistant message to history even if API call failed, to show error
        chat_history.append({"role":"assistant","content":full_response,"model": chosen_model_key if api_call_ok else FALLBACK_MODEL_KEY}) # Tag with actual model used or fallback on error
        
        if api_call_ok:
            if not use_fallback: # Only record use for non-fallback standard models
               record_use(chosen_model_key) # This saves quota file
            if sessions[current_sid]["title"] == "New chat" and prompt: # Auto-title if new and prompt exists
               sessions[current_sid]["title"] = _autoname(prompt)
               _delete_unused_blank_sessions(keep_sid=current_sid) # Clean up other new chats if any
        
        _save(SESS_FILE, sessions) # Save session file
        
        # Rerun to update UI (e.g. quotas, title, or go to setup if api_key_auth_failed)
        st.rerun()
        
# NOTE: Removed the __main__ self-relaunch block 
# as it is not needed/desired for Streamlit Cloud deployment.
