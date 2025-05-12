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

# ------------------------- Imports ------------------------- #
import json, logging, os, sys, subprocess, time, requests
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo # Python 3.9+
import streamlit as st

# -------------------------- Configuration --------------------------- #
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
    "A": "google/gemini-2.5-pro-preview",    # Gemini 2.5 Pro
    "B": "openai/o4-mini",                   # GPT-4o mini
    "C": "openai/chatgpt-4o-latest",         # GPT-4o
    "D": "deepseek/deepseek-r1",             # DeepSeek R1
    "E": "anthropic/claude-3.7-sonnet",      # Claude 3.7 Sonnet
    "F": "google/gemini-2.5-flash-preview"   # Gemini 2.5 Flash
}
# Consider trying a slightly more capable free model if Flash consistently fails complex routing
# ROUTER_MODEL_ID = "mistralai/mistral-7b-instruct:free"
ROUTER_MODEL_ID = "google/gemini-2.0-flash-exp:free" # Keep Flash for now, try prompt fix first
MAX_HISTORY_CHARS_FOR_ROUTER = 3000  # Approx. 750 tokens for history context

MAX_TOKENS = { # Per-call max_tokens for API request (max output generation length)
    "A": 16_000, "B": 8_000, "C": 16_000,
    "D": 8_000, "E": 8_000, "F": 8_000
}

# QUOTA CONFIGURATION (Unchanged)
NEW_PLAN_CONFIG = {
    "A": (10, 200, 5000, 100000, 5000, 100000, 3, 3 * 3600),
    "B": (10, 200, 5000, 100000, 5000, 100000, 0, 0),
    "C": (10, 200, 5000, 100000, 5000, 100000, 0, 0),
    "D": (10, 200, 5000, 100000, 5000, 100000, 0, 0),
    "E": (10, 200, 5000, 100000, 5000, 100000, 0, 0),
    "F": (150, 3000, 75000, 1500000, 75000, 1500000, 0, 0)
}

EMOJI = {
    "A": "🌟", "B": "🔷", "C": "🟥", "D": "🟢", "E": "🖋️", "F": "🌀"
}

# MODEL_DESCRIPTIONS (Reflects cost F < D < B < A < E < C)
MODEL_DESCRIPTIONS = {
    "A": "🌟 (gemini-2.5-pro-preview) – High capability, moderate cost.",
    "B": "🔷 (o4-mini) – Mid-stakes reasoning, cost-effective.",
    "C": "🟥 (chatgpt-4o-latest) – Polished/empathetic, HIGHEST cost.",
    "D": "🟢 (deepseek-r1) – Very cheap factual/technical reasoning.",
    "E": "🖋️ (claude-3.7-sonnet) – Novel, creative, high cost.",
    "F": "🌀 (gemini-2.5-flash-preview) – Quick, CHEAPEST, simple tasks."
}

# ROUTER_MODEL_GUIDANCE (Reflects cost F < D < B < A < E < C, focuses on adequacy)
ROUTER_MODEL_GUIDANCE = {
    "A": "(Model A: High Capability, Moderate Cost [Cost Rank 4/6]) Use for complex reasoning, demanding creative tasks (sophisticated analysis, long-form generation) ONLY IF 'B', 'D', or 'F' definitively *lack the power*. Cheaper than E, C.",
    "B": "(Model B: Solid Mid-Tier [Cost Rank 3/6]) Use for general chat, moderate reasoning, summarization, standard tasks IF 'F'/'D' are *clearly insufficient*. Good balance. Cheaper than A, E, C.",
    "C": "(Model C: Polished & Empathetic, HIGHEST COST [Cost Rank 6/6]) ***AVOID UNLESS ABSOLUTELY NECESSARY***. Use ONLY for tasks demanding *extreme* polish/empathy where its specific style is *indispensable*, AND *ALL* cheaper options ('E', 'A', 'B', 'D', 'F') are *demonstrably inadequate*. Requires strong justification.",
    "D": "(Model D: Factual & Technical [Cost Rank 2/6]) Use for factual Q&A, code tasks, data extraction, straightforward logic IF 'F' is *too basic*. Prefer over 'B'/'A'/'E'/'C' for these tasks if cost allows. Slow.",
    "E": "(Model E: Novel & Creative, High Cost [Cost Rank 5/6]) Use for highly *original/unique* creative content, brainstorming fresh angles, or a distinct non-corporate style, ONLY IF 'A'/'B' *lack the required creative spark* AND the high cost is justified. Cheaper than C.",
    "F": "(Model F: Fast & Economical, CHEAPEST [Cost Rank 1/6]) Use for *simple, low-stakes* tasks: quick Q&A, short summaries, basic classification. ***DO NOT USE 'F' IF*** the query involves: complexity, multi-step reasoning, sensitive topics (mental health, grief, safety), math, deep analysis, nuanced creativity, or requires high accuracy/reliability." # Added strong negative constraints
}

TZ = ZoneInfo("Australia/Sydney")
DATA_DIR   = Path(__file__).parent
SESS_FILE  = DATA_DIR / "chat_sessions.json"
QUOTA_FILE = DATA_DIR / "quotas.json"
CONFIG_FILE = DATA_DIR / "app_config.json"


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

def _today():    return datetime.now(TZ).date().isoformat()
def _ymonth():   return datetime.now(TZ).strftime("%Y-%m")

def _load_app_config():
    return _load(CONFIG_FILE, {})

def _save_app_config(api_key_value: str):
    config_data = _load_app_config()
    config_data["openrouter_api_key"] = api_key_value
    _save(CONFIG_FILE, config_data)

def format_token_count(num):
    if num is None: return "N/A"
    num = float(num)
    if num < 1000:
        return str(int(num))
    elif num < 1_000_000:
        formatted_num = f"{num/1000:.1f}"
        return formatted_num.replace(".0", "") + "k"
    else:
        formatted_num = f"{num/1_000_000:.1f}"
        return formatted_num.replace(".0", "") + "M"

# --------------------- Quota Management (Revised) ------------------------
_g_quota_data = None
_g_quota_data_last_refreshed_stamps = {"d": None, "m": None}

USAGE_KEYS_PERIODIC = ["d_u", "m_u", "d_it_u", "m_it_u", "d_ot_u", "m_ot_u"]
MODEL_A_3H_CALLS_KEY = "model_A_3h_calls"

def _reset(block: dict, period_prefix: str, current_stamp: str, model_keys_zeros: dict) -> bool:
    data_changed = False
    period_stamp_key = period_prefix

    if block.get(period_stamp_key) != current_stamp:
        block[period_stamp_key] = current_stamp
        for usage_type_suffix in ["_u", "_it_u", "_ot_u"]:
            usage_dict_key = f"{period_prefix}{usage_type_suffix}"
            block[usage_dict_key] = model_keys_zeros.copy()
        data_changed = True
        logging.info(f"Quota period '{period_stamp_key}' reset for new stamp '{current_stamp}'.")
    else:
        # Ensure all current model keys exist even if period didn't reset
        for usage_type_suffix in ["_u", "_it_u", "_ot_u"]:
            usage_dict_key = f"{period_prefix}{usage_type_suffix}"
            if usage_dict_key not in block:
                block[usage_dict_key] = model_keys_zeros.copy()
                data_changed = True
                logging.info(f"Initialized missing usage dict '{usage_dict_key}' for stamp '{current_stamp}'.")
            else:
                current_period_usage_dict = block[usage_dict_key]
                for model_k_map in model_keys_zeros.keys():
                    if model_k_map not in current_period_usage_dict:
                        current_period_usage_dict[model_k_map] = 0
                        data_changed = True
                        logging.info(f"Added missing model '{model_k_map}' to usage dict '{usage_dict_key}' for stamp '{current_stamp}'.")
    return data_changed

def _ensure_quota_data_is_current():
    global _g_quota_data, _g_quota_data_last_refreshed_stamps
    now_d_stamp, now_m_stamp = _today(), _ymonth()
    needs_full_refresh_logic = False

    if _g_quota_data is None:
        needs_full_refresh_logic = True
        logging.info("Quota data not in memory. Performing initial load and refresh.")
    elif ((_g_quota_data_last_refreshed_stamps["d"] != now_d_stamp) or
          (_g_quota_data_last_refreshed_stamps["m"] != now_m_stamp)):
        needs_full_refresh_logic = True
        logging.info(f"Quota period change detected. Refreshing quota data.")

    if not needs_full_refresh_logic:
        # Prune 3-hour calls even without a full reset
        if MODEL_A_3H_CALLS_KEY in _g_quota_data and "A" in NEW_PLAN_CONFIG and NEW_PLAN_CONFIG["A"][7] > 0:
            _, _, _, _, _, _, _, three_hr_window_seconds = NEW_PLAN_CONFIG["A"]
            current_time = time.time()
            original_len = len(_g_quota_data.get(MODEL_A_3H_CALLS_KEY, [])) # Use get for safety
            _g_quota_data[MODEL_A_3H_CALLS_KEY] = [
                ts for ts in _g_quota_data.get(MODEL_A_3H_CALLS_KEY, []) # Use get for safety
                if current_time - ts < three_hr_window_seconds
            ]
            if len(_g_quota_data[MODEL_A_3H_CALLS_KEY]) != original_len:
                logging.info(f"Pruned Model A 3-hour call timestamps. Original: {original_len}, New: {len(_g_quota_data[MODEL_A_3H_CALLS_KEY])}.")
                _save(QUOTA_FILE, _g_quota_data) # Save if pruned
        return _g_quota_data

    q_loaded_data = _load(QUOTA_FILE, {})
    data_was_modified = _g_quota_data is None

    # Use NEW_PLAN_CONFIG keys as the source of truth for active models
    active_model_keys = set(NEW_PLAN_CONFIG.keys())
    cleaned_during_load = False

    # Clean obsolete model keys from existing usage dictionaries
    for usage_key_template in USAGE_KEYS_PERIODIC:
        if usage_key_template in q_loaded_data:
            current_period_usage_dict = q_loaded_data[usage_key_template]
            keys_in_usage = list(current_period_usage_dict.keys())
            for model_key_in_usage in keys_in_usage:
                if model_key_in_usage not in active_model_keys:
                    try:
                        del current_period_usage_dict[model_key_in_usage]
                        logging.info(f"Removed obsolete model key '{model_key_in_usage}' from quota usage '{usage_key_template}'.")
                        cleaned_during_load = True
                    except KeyError: pass
    if cleaned_during_load: data_was_modified = True

    # Remove obsolete top-level keys (e.g., old weekly format)
    obsolete_keys = ["w", "w_u"]
    for key in obsolete_keys:
        if key in q_loaded_data:
            del q_loaded_data[key]
            data_was_modified = True
            logging.info(f"Removed obsolete key '{key}' from quota data.")

    # Use NEW_PLAN_CONFIG keys for zero initialization in _reset
    current_model_zeros = {k: 0 for k in active_model_keys}
    reset_occurred_d = _reset(q_loaded_data, "d", now_d_stamp, current_model_zeros)
    reset_occurred_m = _reset(q_loaded_data, "m", now_m_stamp, current_model_zeros)
    if reset_occurred_d or reset_occurred_m: data_was_modified = True

    # Initialize or prune 3-hour calls for Model A
    if "A" in NEW_PLAN_CONFIG and NEW_PLAN_CONFIG["A"][6] > 0: # Check if Model A has a 3hr limit configured
        three_hr_config = NEW_PLAN_CONFIG["A"]
        three_hr_window_seconds = three_hr_config[7]
        if MODEL_A_3H_CALLS_KEY not in q_loaded_data:
            q_loaded_data[MODEL_A_3H_CALLS_KEY] = []
            data_was_modified = True # Initializing counts as modification
            logging.info(f"Initialized Model A 3-hour call list ({MODEL_A_3H_CALLS_KEY}).")

        current_time = time.time()
        original_len = len(q_loaded_data.get(MODEL_A_3H_CALLS_KEY, []))
        q_loaded_data[MODEL_A_3H_CALLS_KEY] = [
            ts for ts in q_loaded_data.get(MODEL_A_3H_CALLS_KEY, [])
            if current_time - ts < three_hr_window_seconds
        ]
        if len(q_loaded_data[MODEL_A_3H_CALLS_KEY]) != original_len:
             logging.info(f"Pruned Model A 3-hour call timestamps during full refresh. Original: {original_len}, New: {len(q_loaded_data[MODEL_A_3H_CALLS_KEY])}.")
             data_was_modified = True
    elif MODEL_A_3H_CALLS_KEY in q_loaded_data:
        # If Model A no longer has a 3hr limit configured, remove the tracking key
        del q_loaded_data[MODEL_A_3H_CALLS_KEY]
        data_was_modified = True
        logging.info(f"Removed obsolete Model A 3-hour call list ({MODEL_A_3H_CALLS_KEY}) as it's no longer configured in NEW_PLAN_CONFIG.")


    if data_was_modified:
        _save(QUOTA_FILE, q_loaded_data)
        logging.info("Quota data was modified (loaded/cleaned/reset/pruned) and saved to disk.")

    _g_quota_data = q_loaded_data
    _g_quota_data_last_refreshed_stamps = {"d": now_d_stamp, "m": now_m_stamp}
    return _g_quota_data


def get_quota_usage_and_limits(model_key: str):
    if model_key not in NEW_PLAN_CONFIG:
        logging.error(f"Model key '{model_key}' not in NEW_PLAN_CONFIG.")
        return {}

    current_q_data = _ensure_quota_data_is_current()
    plan = NEW_PLAN_CONFIG[model_key]

    limits = {
        "limit_daily_msg": plan[0], "limit_monthly_msg": plan[1],
        "limit_daily_in_tokens": plan[2], "limit_monthly_in_tokens": plan[3],
        "limit_daily_out_tokens": plan[4], "limit_monthly_out_tokens": plan[5],
        "limit_3hr_msg": plan[6] if plan[6] > 0 else float('inf')
    }

    usage = {
        "used_daily_msg": current_q_data.get("d_u", {}).get(model_key, 0),
        "used_monthly_msg": current_q_data.get("m_u", {}).get(model_key, 0),
        "used_daily_in_tokens": current_q_data.get("d_it_u", {}).get(model_key, 0),
        "used_monthly_in_tokens": current_q_data.get("m_it_u", {}).get(model_key, 0),
        "used_daily_out_tokens": current_q_data.get("d_ot_u", {}).get(model_key, 0),
        "used_monthly_out_tokens": current_q_data.get("m_ot_u", {}).get(model_key, 0),
        "used_3hr_msg": 0
    }

    # Specifically handle Model A 3-hour limit if configured
    if model_key == "A" and plan[6] > 0 and plan[7] > 0:
        current_time = time.time()
        three_hr_window_seconds = plan[7]
        recent_calls = [
            ts for ts in current_q_data.get(MODEL_A_3H_CALLS_KEY, []) # Safely get list
            if current_time - ts < three_hr_window_seconds
        ]
        usage["used_3hr_msg"] = len(recent_calls)
        # No need to re-save here, pruning happens in _ensure_quota_data_is_current

    return {**usage, **limits}

def is_model_available(model_key: str) -> bool:
    if model_key not in NEW_PLAN_CONFIG:
        logging.warning(f"is_model_available: Model key '{model_key}' not in NEW_PLAN_CONFIG. Assuming unavailable.")
        return False

    stats = get_quota_usage_and_limits(model_key)
    if not stats: return False # Should not happen if key is in NEW_PLAN_CONFIG

    # Check all standard quotas (only if limit > 0)
    if stats["limit_daily_msg"] > 0 and stats["used_daily_msg"] >= stats["limit_daily_msg"]: return False
    if stats["limit_monthly_msg"] > 0 and stats["used_monthly_msg"] >= stats["limit_monthly_msg"]: return False
    if stats["limit_daily_in_tokens"] > 0 and stats["used_daily_in_tokens"] >= stats["limit_daily_in_tokens"]: return False
    if stats["limit_monthly_in_tokens"] > 0 and stats["used_monthly_in_tokens"] >= stats["limit_monthly_in_tokens"]: return False
    if stats["limit_daily_out_tokens"] > 0 and stats["used_daily_out_tokens"] >= stats["limit_daily_out_tokens"]: return False
    if stats["limit_monthly_out_tokens"] > 0 and stats["used_monthly_out_tokens"] >= stats["limit_monthly_out_tokens"]: return False

    # Check specific Model A 3-hour limit if applicable
    if model_key == "A" and stats["limit_3hr_msg"] != float('inf'):
        if stats["used_3hr_msg"] >= stats["limit_3hr_msg"]: return False

    return True

def get_remaining_daily_messages(model_key: str) -> int:
    if model_key not in NEW_PLAN_CONFIG: return 0
    stats = get_quota_usage_and_limits(model_key)
    if not stats: return 0
    # Return 0 if the limit itself is 0
    if stats["limit_daily_msg"] == 0: return 0
    return max(0, stats["limit_daily_msg"] - stats["used_daily_msg"])

def record_use(model_key: str, prompt_tokens: int, completion_tokens: int):
    # Only record usage for models defined in NEW_PLAN_CONFIG (quota-tracked models)
    if model_key not in NEW_PLAN_CONFIG:
        logging.warning(f"Attempted to record usage for non-quota-tracked or unknown model key: {model_key}")
        return

    current_q_data = _ensure_quota_data_is_current()

    # Ensure all necessary usage dictionaries and the specific model key exist
    for period_prefix in ["d", "m"]:
        for usage_suffix in ["_u", "_it_u", "_ot_u"]:
            usage_key = f"{period_prefix}{usage_suffix}"
            current_q_data.setdefault(usage_key, {})
            current_q_data[usage_key].setdefault(model_key, 0)

    # Increment usage counts
    current_q_data["d_u"][model_key] += 1
    current_q_data["m_u"][model_key] += 1
    current_q_data["d_it_u"][model_key] += prompt_tokens
    current_q_data["m_it_u"][model_key] += prompt_tokens
    current_q_data["d_ot_u"][model_key] += completion_tokens
    current_q_data["m_ot_u"][model_key] += completion_tokens

    # Record timestamp for Model A 3-hour limit if applicable
    if model_key == "A" and NEW_PLAN_CONFIG["A"][6] > 0:
        current_q_data.setdefault(MODEL_A_3H_CALLS_KEY, []).append(time.time())
        # Pruning happens in _ensure_quota_data_is_current, just append here

    _save(QUOTA_FILE, current_q_data)
    logging.info(f"Recorded usage for model '{model_key}': 1 msg, {prompt_tokens}p, {completion_tokens}c tokens. Quotas saved.")


# --------------------- Session Management -----------------------
def _delete_unused_blank_sessions(keep_sid: str = None):
    sids_to_delete = []
    for sid, data in list(sessions.items()):
        if sid == keep_sid: continue
        # Check if title is default AND messages list is empty or None
        if data.get("title") == "New chat" and not data.get("messages"):
            sids_to_delete.append(sid)
    if sids_to_delete:
        for sid_del in sids_to_delete:
            logging.info(f"Auto-deleting blank session: {sid_del}")
            try: del sessions[sid_del]
            except KeyError: logging.warning(f"Session {sid_del} already deleted, skipping.")
        return True # Indicate deletion occurred
    return False

sessions = _load(SESS_FILE, {})

def _new_sid():
    sid = str(int(time.time() * 1000))
    sessions[sid] = {"title": "New chat", "messages": []}
    # Clean up *other* blank sessions immediately after creating a new one
    # This prevents multiple "New chat" entries from lingering if the user quickly switches
    _delete_unused_blank_sessions(keep_sid=sid)
    return sid


def _autoname(seed: str) -> str:
    words = seed.strip().split()
    cand = " ".join(words[:4]) or "Chat" # Use up to 4 words for title
    return (cand[:30] + "…") if len(cand) > 30 else cand # Max length 30


# --------------------------- Logging ----------------------------
# Ensure basic logging is configured early
# Set level to DEBUG to see detailed router prompts and responses
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
numeric_level = getattr(logging, log_level, logging.INFO)
if not logging.getLogger().handlers:
    logging.basicConfig(level=numeric_level, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)
    logging.info(f"Logging level set to: {log_level}")


def is_api_key_valid(api_key_value):
    return api_key_value and isinstance(api_key_value, str) and api_key_value.startswith("sk-or-")

# -------------------------- API Calls --------------------------
def api_post(payload: dict, *, stream: bool=False, timeout: int=DEFAULT_TIMEOUT):
    active_api_key = st.session_state.get("openrouter_api_key")
    if not is_api_key_valid(active_api_key):
        st.session_state.api_key_auth_failed = True
        raise ValueError("OpenRouter API Key is not set or syntactically invalid. Configure in Settings.")
    headers = {"Authorization": f"Bearer {active_api_key}", "Content-Type":  "application/json"}
    logging.info(f"POST /chat/completions -> model={payload.get('model')}, stream={stream}, max_tokens={payload.get('max_tokens')}")
    try:
        response = requests.post(f"{OPENROUTER_API_BASE}/chat/completions", headers=headers, json=payload, stream=stream, timeout=timeout)
        response.raise_for_status()
        st.session_state.api_key_auth_failed = False # Success means key is valid
        return response
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            st.session_state.api_key_auth_failed = True
            logging.error(f"API POST failed with 401 (Unauthorized): {e.response.text}")
        else: logging.error(f"API POST failed with {e.response.status_code}: {e.response.text}")
        raise
    except requests.exceptions.RequestException as e:
        # Handle network errors separately, don't assume auth failed
        logging.error(f"API POST failed with network error: {e}")
        raise

def streamed(model: str, messages: list, max_tokens_out: int):
    payload = {"model": model, "messages": messages, "stream": True, "max_tokens": max_tokens_out}
    st.session_state.pop("last_stream_usage", None) # Clear previous usage before new stream

    try:
        with api_post(payload, stream=True) as r:
            for line in r.iter_lines():
                if not line: continue
                line_str = line.decode("utf-8")
                if line_str.startswith(": OPENROUTER PROCESSING"): continue
                if not line_str.startswith("data: "):
                    logging.warning(f"Unexpected non-event-stream line: {line_str}"); continue
                data = line_str[6:].strip()
                if data == "[DONE]": break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    logging.error(f"Bad JSON chunk: {data}"); yield None, "Error decoding response chunk"; return

                # Check for API errors within the chunk
                if "error" in chunk:
                    error_info = chunk["error"]
                    # Check if it's an object or string, Anthropic sometimes returns string errors
                    if isinstance(error_info, dict):
                        msg = error_info.get("message", "Unknown API error")
                        code = error_info.get("code", "N/A")
                    elif isinstance(error_info, str):
                        msg = error_info
                        code = "N/A"
                    else:
                        msg = "Unknown API error format"
                        code = "N/A"
                    logging.error(f"API chunk error (Code: {code}): {msg}"); yield None, msg; return

                # Process successful chunk
                if chunk.get("choices") and isinstance(chunk["choices"], list) and len(chunk["choices"]) > 0:
                    # Store usage if present in this chunk
                    if "usage" in chunk and chunk["usage"] is not None:
                        st.session_state.last_stream_usage = chunk["usage"]

                    # Extract content delta
                    delta = chunk["choices"][0].get("delta", {}).get("content")
                    if delta is not None: yield delta, None
                # Handle potential empty chunks or chunks without choices/delta gracefully
                # else:
                #     logging.debug(f"Received chunk without expected content delta: {data}")

    except ValueError as ve: # Catches the API key validation error from api_post
        logging.error(f"ValueError during streamed call setup: {ve}"); yield None, str(ve)
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else 'N/A'; text = e.response.text if e.response else 'No response text'
        logging.error(f"Stream HTTPError {status_code}: {text}")
        # If 401 occurs during streaming, set the flag
        if status_code == 401: st.session_state.api_key_auth_failed = True
        yield None, f"HTTP {status_code}: An error occurred with the API provider. Details: {text}"
    except requests.exceptions.RequestException as e:
        logging.error(f"Stream Network Error: {e}"); yield None, f"Network Error: Failed to connect to API. {e}"
    except Exception as e:
        logging.exception(f"Unexpected error during streamed API call: {e}") # Use logging.exception to include traceback
        yield None, f"An unexpected error occurred: {e}"

# ------------------------- Model Routing (REVISED) -----------------------
def route_choice(user_msg: str, allowed: list[str], chat_history: list) -> str:
    # Determine fallback choice (logic unchanged, but now F is cheapest)
    if "F" in allowed: fallback_choice_letter = "F"
    elif allowed: fallback_choice_letter = allowed[0]
    elif "F" in MODEL_MAP: fallback_choice_letter = "F"
    elif MODEL_MAP: fallback_choice_letter = list(MODEL_MAP.keys())[0]
    else:
        logging.error("Router: No models configured. Using FALLBACK_MODEL_KEY.")
        return FALLBACK_MODEL_KEY

    if not allowed:
        logging.warning(f"Router: No models available due to quotas. Defaulting to fallback: '{fallback_choice_letter}' (or free if needed).")
        if is_model_available(fallback_choice_letter): return fallback_choice_letter
        else:
            logging.warning(f"Fallback choice '{fallback_choice_letter}' also unavailable. Using free fallback: {FALLBACK_MODEL_KEY}.")
            return FALLBACK_MODEL_KEY

    if len(allowed) == 1:
        logging.info(f"Router: Only one model allowed ('{allowed[0]}'), selecting it directly.")
        return allowed[0]

    # Prepare history context (Unchanged)
    history_segments = []
    current_chars = 0
    relevant_history_for_router = chat_history[:-1] if chat_history else []
    for msg in reversed(relevant_history_for_router):
        role = msg.get("role", "assistant").capitalize(); content = msg.get("content", "")
        if not isinstance(content, str): content = str(content)
        segment = f"{role}: {content}\n";
        if current_chars + len(segment) > MAX_HISTORY_CHARS_FOR_ROUTER: break
        history_segments.append(segment); current_chars += len(segment)
    history_context_str = "".join(reversed(history_segments)).strip() or "No prior conversation history."

    # --- Build the REVISED system prompt ---
    system_prompt_parts = [
        "You are an expert AI model routing assistant. Your task is to select the *single most appropriate and cost-effective* model letter from the 'Available Models' list to handle the 'Latest User Query'.",
        "Core Principles:",
        "1. **Assess Adequacy FIRST:** Before considering cost, determine the *minimum capability required* for the query. Is it simple, moderate, complex, creative, sensitive? Does it require high accuracy, deep reasoning, or specific stylistic output?",
        "2. **Maximize Cost-Effectiveness SECOND:** Once adequacy is assessed, choose the ***absolute cheapest*** model from the 'Available Models' list that meets the minimum capability requirement. Cost Order (cheapest to most expensive): **F < D < B < A < E < C**.",
        "3. **Prioritize Safety and Sensitivity:** For queries involving mental health, grief, safety concerns, or other sensitive topics, *err on the side of caution* and choose a more capable/nuanced model (likely B, A, or E minimum) even if a cheaper one *might* seem barely adequate. Avoid 'F' and 'D' for sensitive topics.", # Added safety emphasis
        "4. **Consider History:** Use 'Recent Conversation History' for context, but base the decision primarily on the *Latest User Query* requirements."
    ]
    system_prompt_parts.append("\nAvailable Models (Cost Order: F < D < B < A < E < C):")
    # Use the updated ROUTER_MODEL_GUIDANCE which includes negative constraints for F
    for k_model_key in allowed:
        description = ROUTER_MODEL_GUIDANCE.get(k_model_key, f"(Model {k_model_key} - Description not found).")
        system_prompt_parts.append(f"- {k_model_key}: {description}")

    # REVISED Specific Selection Guidance emphasizing adequacy *before* cost saving
    system_prompt_parts.append("\nDecision Process for 'Latest User Query':")
    system_prompt_parts.append("1. **Analyze Query:** Understand complexity, intent, required style, sensitivity.")
    system_prompt_parts.append("2. **Is 'F' sufficient?** (Available: {}). Check F's 'DO NOT USE' constraints in its description above. If the query is simple AND avoids those constraints, choose 'F'. ".format("Yes" if "F" in allowed else "No"))
    system_prompt_parts.append("3. **If not 'F', is 'D' sufficient?** (Available: {}). Suitable for technical/factual tasks if 'F' is too basic. Check sensitivity.".format("Yes" if "D" in allowed else "No"))
    system_prompt_parts.append("4. **If not 'D', is 'B' sufficient?** (Available: {}). Good for general moderate tasks, standard creativity, sensitive topics where F/D are inappropriate.".format("Yes" if "B" in allowed else "No"))
    system_prompt_parts.append("5. **If not 'B', is 'A' sufficient?** (Available: {}). Needed for higher complexity/reasoning/generation than 'B'.".format("Yes" if "A" in allowed else "No"))
    system_prompt_parts.append("6. **If not 'A', is 'E' sufficient?** (Available: {}). Needed for specific *novel/unique* creative style beyond 'A', or sensitive topics needing high nuance.".format("Yes" if "E" in allowed else "No"))
    system_prompt_parts.append("7. **If not 'E', consider 'C'?** (Available: {}). ***Extreme last resort*** only if *peak* polish/empathy is explicitly required and worth the highest cost.".format("Yes" if "C" in allowed else "No"))
    system_prompt_parts.append("8. **Select the *first* sufficient model encountered in the F -> D -> B -> A -> E -> C evaluation order that is also in the 'Available Models' list.**")

    system_prompt_parts.append("\nRecent Conversation History (Context):")
    system_prompt_parts.append(history_context_str)
    system_prompt_parts.append(f"\nAvailable Model Letters: {', '.join(sorted(allowed))}") # Explicitly list allowed letters
    # User message will be appended by the API call structure

    system_prompt_parts.append("\nINSTRUCTION: Based *strictly* on the adequacy assessment and cost-optimization process described (F->D->B->A->E->C), analyze the 'Latest User Query' (provided in the user role message). Respond with ONLY the single capital letter of the *cheapest adequate* model available. NO EXPLANATION.")
    final_system_message = "\n".join(system_prompt_parts)
    logging.debug(f"Router System Prompt:\n{final_system_message}") # Log the prompt for debugging

    router_messages = [{"role": "system", "content": final_system_message}, {"role": "user", "content": user_msg}]
    # Increased temperature slightly
    payload_r = {"model": ROUTER_MODEL_ID, "messages": router_messages, "max_tokens": 10, "temperature": 0.2}
    logging.debug(f"Router Payload: {json.dumps(payload_r, indent=2)}")

    try:
        r = api_post(payload_r)
        choice_data = r.json()
        logging.debug(f"Router Full Response JSON: {json.dumps(choice_data, indent=2)}") # Log full response
        raw_text_response = choice_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()
        logging.info(f"Router raw text response: '{raw_text_response}' for query: '{user_msg[:100]}...'")

        chosen_model_letter = None
        # Check the response *only* for allowed characters
        for char_in_response in raw_text_response:
            if char_in_response in allowed:
                chosen_model_letter = char_in_response
                logging.info(f"Router selected model '{chosen_model_letter}' (from response '{raw_text_response}', allowed: {allowed})")
                break # Found first allowed character

        if chosen_model_letter:
            return chosen_model_letter
        else:
            logging.warning(f"Router returned ('{raw_text_response}') - no allowed letter found or response invalid. Fallback to '{fallback_choice_letter}'.")
            if is_model_available(fallback_choice_letter): return fallback_choice_letter
            else:
                logging.warning(f"Fallback choice '{fallback_choice_letter}' also unavailable. Using free fallback: {FALLBACK_MODEL_KEY}.")
                return FALLBACK_MODEL_KEY

    # --- Error handling (same as before) ---
    except ValueError as ve: logging.error(f"Router call failed due to invalid API key or config: {ve}"); st.session_state.api_key_auth_failed = True; return None
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else 'N/A'; err_text = e.response.text if e.response else 'No response text'
        logging.error(f"Router HTTPError {status_code}: {err_text}")
        if status_code == 401: st.session_state.api_key_auth_failed = True; return None
    except requests.exceptions.RequestException as e: logging.error(f"Router Network Error: {e}")
    except (KeyError, IndexError, AttributeError, json.JSONDecodeError) as je:
        response_text_for_log = r.text if 'r' in locals() and hasattr(r, 'text') else "N/A"
        logging.error(f"Router JSON/structure error: {je}. Raw: {response_text_for_log}")
    except Exception as e: logging.exception(f"Router unexpected error: {e}")

    # Fallback if any error occurred above (except auth error which returns None)
    logging.warning(f"Router failed. Fallback to model letter: {fallback_choice_letter} (or free if needed).")
    if is_model_available(fallback_choice_letter): return fallback_choice_letter
    else:
        logging.warning(f"Fallback choice '{fallback_choice_letter}' also unavailable. Using free fallback: {FALLBACK_MODEL_KEY}.")
        return FALLBACK_MODEL_KEY

# --------------------- Credits Endpoint -----------------------
def get_credits():
    active_api_key = st.session_state.get("openrouter_api_key")
    if not is_api_key_valid(active_api_key):
        logging.warning("get_credits: API Key is not syntactically valid or not set."); return None, None, None
    try:
        # Use a shorter timeout for the credits check
        r = requests.get(f"{OPENROUTER_API_BASE}/credits", headers={"Authorization": f"Bearer {active_api_key}"}, timeout=15)
        r.raise_for_status()
        d = r.json().get("data") # Use .get for safety
        if d and "limit" in d and "usage" in d: # OpenRouter v1 uses "limit" and "usage"
            total_credits = float(d["limit"])
            total_usage = float(d["usage"])
            remaining_credits = total_credits - total_usage
            st.session_state.api_key_auth_failed = False # Success means key is valid for this endpoint
            return total_credits, total_usage, remaining_credits
        elif d and "total_credits" in d and "total_usage" in d: # Older/alternative format check
            total_credits = float(d["total_credits"])
            total_usage = float(d["total_usage"])
            remaining_credits = total_credits - total_usage
            st.session_state.api_key_auth_failed = False
            return total_credits, total_usage, remaining_credits
        else:
            logging.warning(f"Could not parse /credits response structure: {r.json()}")
            return None, None, None

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else 'N/A'; err_text = e.response.text if e.response else 'No response text'
        if status_code == 401:
            st.session_state.api_key_auth_failed = True
            logging.warning(f"Could not fetch /credits: HTTP {status_code} Unauthorized. {err_text}")
        else: logging.warning(f"Could not fetch /credits: HTTP {status_code}. {err_text}")
        return None, None, None
    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logging.warning(f"Could not fetch /credits due to network/parsing error: {e}"); return None, None, None


# ------------------------- UI Styling --------------------------
def load_custom_css():
    # CSS remains the same as provided previously
    css = """
    <style>
        :root {
            /* Core Colors */
            --app-bg-color: #F8F9FA; /* Lighter, softer background */
            --app-secondary-bg-color: #FFFFFF; /* White for secondary elements like sidebar, cards */
            --app-text-color: #212529; /* Darker grey for better contrast */
            --app-text-secondary-color: #6C757D; /* Lighter grey for secondary text */
            --app-primary-color: #007BFF; /* Standard Bootstrap blue */
            --app-primary-hover-color: #0056b3;
            --app-divider-color: #DEE2E6; /* Lighter divider */
            --app-border-color: #CED4DA; /* Softer border for inputs etc. */
            --app-success-color: #28A745;
            --app-warning-color: #FFC107;
            --app-danger-color: #DC3545;

            --border-radius-sm: 0.2rem;
            --border-radius-md: 0.25rem;
            --border-radius-lg: 0.3rem;

            --spacing-xs: 0.25rem;
            --spacing-sm: 0.5rem;
            --spacing-md: 1rem;
            --spacing-lg: 1.5rem;

            --shadow-sm: 0 .125rem .25rem rgba(0,0,0,.075);
            --shadow-md: 0 .5rem 1rem rgba(0,0,0,.15);

            --app-font: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
        }

        body, .stApp {
            font-family: var(--app-font) !important;
            background-color: var(--app-bg-color) !important;
            color: var(--app-text-color) !important;
        }
        .main .block-container {
            background-color: var(--app-bg-color);
            padding-top: var(--spacing-md);
            padding-bottom: var(--spacing-lg);
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: var(--app-secondary-bg-color);
            border-right: 1px solid var(--app-divider-color);
            padding: var(--spacing-md);
        }
        [data-testid="stSidebar"] .stImage > img { /* Sidebar Logo */
            border-radius: var(--border-radius-md);
            box-shadow: var(--shadow-sm);
            width: 40px !important; height: 40px !important;
            margin-right: var(--spacing-sm);
        }
        [data-testid="stSidebar"] h1 { /* App Title in Sidebar */
            font-size: 1.4rem !important;
            color: var(--app-text-color);
            font-weight: 600;
            margin-bottom: 0;
            line-height: 1.2;
            padding-top: 0.15rem; /* Align with logo better */
        }
        .sidebar-title-container { /* Custom container for logo and title */
            display: flex;
            align-items: center;
            margin-bottom: var(--spacing-md);
        }

        [data-testid="stSidebar"] .stButton > button {
            border-radius: var(--border-radius-md);
            border: 1px solid var(--app-border-color);
            padding: 0.5em 0.8em; font-size: 0.9rem;
            background-color: var(--app-secondary-bg-color);
            color: var(--app-text-color);
            transition: background-color 0.2s, border-color 0.2s;
            width: 100%; margin-bottom: var(--spacing-sm);
            text-align: left; font-weight: 500;
        }
        [data-testid="stSidebar"] .stButton > button:hover:not(:disabled) {
            border-color: var(--app-primary-color);
            background-color: color-mix(in srgb, var(--app-primary-color) 8%, transparent);
        }
        [data-testid="stSidebar"] .stButton > button:disabled { /* Active Chat Button */
            opacity: 1.0; cursor: default;
            background-color: color-mix(in srgb, var(--app-primary-color) 15%, transparent) !important;
            border-left: 3px solid var(--app-primary-color) !important;
            border-top-color: var(--app-border-color) !important;
            border-right-color: var(--app-border-color) !important;
            border-bottom-color: var(--app-border-color) !important;
            font-weight: 600; color: var(--app-text-color);
        }
        [data-testid="stSidebar"] [data-testid*="new_chat_button_top"] > button {
            background-color: var(--app-primary-color); color: white;
            border-color: var(--app-primary-color); font-weight: 600;
        }
        [data-testid="stSidebar"] [data-testid*="new_chat_button_top"] > button:hover {
            background-color: var(--app-primary-hover-color);
            border-color: var(--app-primary-hover-color);
        }
        [data-testid="stSidebar"] [data-testid*="new_chat_button_top"] > button:disabled {
            border-left-width: 1px !important; /* Reset active style for disabled new chat */
            background-color: var(--app-secondary-bg-color) !important; /* Make disabled look like normal */
            border-color: var(--app-border-color) !important;
            color: var(--app-text-secondary-color) !important;
            cursor: not-allowed;
        }

        [data-testid="stSidebar"] h3, [data-testid="stSidebar"] .stSubheader { /* Sidebar Section Headers */
            font-size: 0.75rem !important; text-transform: uppercase; font-weight: 600;
            color: var(--app-text-secondary-color);
            margin-top: var(--spacing-md); margin-bottom: var(--spacing-sm);
            letter-spacing: 0.03em;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            border: 1px solid var(--app-divider-color);
            border-radius: var(--border-radius-md);
            background-color: var(--app-secondary-bg-color); /* Match sidebar bg */
            margin-bottom: var(--spacing-sm);
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary {
            padding: 0.5rem var(--spacing-sm) !important;
            font-size: 0.8rem !important; font-weight: 500 !important;
            color: var(--app-text-color) !important;
            border-bottom: 1px solid var(--app-divider-color);
            border-top-left-radius: var(--border-radius-md); border-top-right-radius: var(--border-radius-md);
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
            background-color: color-mix(in srgb, var(--app-text-color) 4%, transparent);
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
            padding: var(--spacing-sm) !important;
             background-color: color-mix(in srgb, var(--app-bg-color) 50%, var(--app-secondary-bg-color) 50%); /* Slightly different from expander summary */
            border-bottom-left-radius: var(--border-radius-md); border-bottom-right-radius: var(--border-radius-md);
        }
        [data-testid="stSidebar"] [data-testid="stExpander"][aria-label^="⚡ DAILY MODEL QUOTAS"] div[data-testid="stExpanderDetails"] {
            padding: 0.4rem var(--spacing-xs) 0.1rem var(--spacing-xs) !important; /* More compact */
        }
        [data-testid="stSidebar"] [data-testid="stExpander"][aria-label^="⚡ DAILY MODEL QUOTAS"] div[data-testid="stHorizontalBlock"] {
            gap: 0.15rem !important; /* Tighter gap for quota items */
        }


        /* Compact Quota Item Styling */
        .compact-quota-item {
            display: flex; flex-direction: column; align-items: center;
            text-align: center; padding: var(--spacing-xs);
            background-color: color-mix(in srgb, var(--app-text-color) 2%, transparent);
            border-radius: var(--border-radius-sm);
            min-width: 30px; /* Ensure a minimum width */
        }
        .cq-info { font-size: 0.65rem; margin-bottom: 2px; line-height: 1; white-space: nowrap; color: var(--app-text-color); }
        .cq-bar-track {
            width: 100%; height: 6px;
            background-color: color-mix(in srgb, var(--app-text-color) 10%, transparent);
            border: 1px solid var(--app-divider-color);
            border-radius: 3px; overflow: hidden; margin-bottom: 3px;
        }
        .cq-bar-fill { height: 100%; border-radius: 3px; transition: width 0.3s ease-in-out, background-color 0.3s ease-in-out; }
        .cq-value { font-size: 0.65rem; font-weight: 600; line-height: 1; }

        /* Settings Panel in Sidebar */
        .settings-panel {
            border: 1px solid var(--app-divider-color);
            border-radius: var(--border-radius-md); padding: var(--spacing-sm);
            margin-top: var(--spacing-xs); margin-bottom: var(--spacing-md);
            background-color: var(--app-bg-color); /* Slightly different from sidebar bg */
        }
        .settings-panel .stTextInput input {
            border-color: var(--app-border-color) !important;
            background-color: var(--app-secondary-bg-color) !important;
            color: var(--app-text-color) !important;
            font-size: 0.85rem;
        }
        .settings-panel .stSubheader {
             color: var(--app-text-color) !important;
             font-weight: 600 !important; font-size: 0.9rem !important;
             margin-bottom: var(--spacing-xs) !important;
        }
        .settings-panel hr { border-top: 1px solid var(--app-divider-color); margin: var(--spacing-sm) 0; }
        .detailed-quota-modelname {
            font-weight: 600; font-size: 0.95em;
            margin-bottom: 0.2rem; display:block;
            color: var(--app-primary-color);
        }
        .detailed-quota-block { font-size: 0.8rem; line-height: 1.5; }
        .detailed-quota-block ul { list-style-type: none; padding-left: 0; margin-bottom: 0.3rem;}
        .detailed-quota-block li { margin-bottom: 0.1rem; }

        /* Chat Input Area */
        [data-testid="stChatInputContainer"] {
            background-color: var(--app-secondary-bg-color);
            border-top: 1px solid var(--app-divider-color);
            padding: var(--spacing-sm) var(--spacing-md);
            box-shadow: 0 -2px 5px rgba(0,0,0,0.03);
        }
        [data-testid="stChatInput"] textarea {
            border: 1px solid var(--app-border-color) !important;
            border-radius: var(--border-radius-md) !important;
            background-color: var(--app-secondary-bg-color) !important; /* Match container */
            color: var(--app-text-color) !important;
            box-shadow: var(--shadow-sm) inset;
        }
        [data-testid="stChatInput"] textarea:focus {
            border-color: var(--app-primary-color) !important;
            box-shadow: 0 0 0 0.2rem color-mix(in srgb, var(--app-primary-color) 25%, transparent) !important;
        }


        /* Chat Messages */
        [data-testid="stChatMessage"] {
            border-radius: var(--border-radius-lg);
            padding: 0.8rem 1rem;
            margin-bottom: var(--spacing-sm);
            box-shadow: var(--shadow-sm);
            border: 1px solid transparent;
            max-width: 80%; /* Slightly reduce max width */
            line-height: 1.5;
        }
        [data-testid="stChatMessage"] p { margin-bottom: 0.5em; } /* Spacing between paragraphs in a message */
        [data-testid="stChatMessage"] p:last-child { margin-bottom: 0; }

        [data-testid="stChatMessage"][data-testid^="stChatMessageUser"] {
            background-color: var(--app-primary-color);
            color: white;
            margin-left: auto;
            border-bottom-right-radius: var(--border-radius-sm); /* Pointy corner */
        }
        [data-testid="stChatMessage"][data-testid^="stChatMessageAssistant"] {
            background-color: var(--app-secondary-bg-color);
            color: var(--app-text-color);
            margin-right: auto;
            border: 1px solid var(--app-divider-color);
            border-bottom-left-radius: var(--border-radius-sm); /* Pointy corner */
        }
        /* Ensure avatars are vertically centered if they are taller than one line of text */
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            padding-top: 0.1rem;
            padding-bottom: 0.1rem; /* Adjust if avatars misalign */
        }


        .sidebar-divider {
             margin: var(--spacing-md) 0; /* More prominent spacing for dividers */
             border: 0; border-top: 1px solid var(--app-divider-color);
        }
        /* Utility for hiding Streamlit's default "Fork on GitHub" ribbon if desired */
        /* #GithubIcon { display: none; } */

        /* Improve general button styling if st.button is used in main area */
        .main .stButton > button:not([data-testid*="new_chat_button_top"]):not([data-testid*="toggle_settings_button_sidebar"]):not([data-testid*="session_button_"]) {
            border-radius: var(--border-radius-md);
            border: 1px solid var(--app-primary-color);
            background-color: var(--app-primary-color);
            color: white;
            padding: 0.5em 1em;
            font-weight: 500;
        }
        .main .stButton > button:not([data-testid*="new_chat_button_top"]):not([data-testid*="toggle_settings_button_sidebar"]):not([data-testid*="session_button_"]):hover {
            background-color: var(--app-primary-hover-color);
            border-color: var(--app-primary-hover-color);
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ----------------- API Key State Initialization -------------------
if "openrouter_api_key" not in st.session_state:
    app_conf = _load_app_config()
    st.session_state.openrouter_api_key = app_conf.get("openrouter_api_key", None)
if "api_key_auth_failed" not in st.session_state: st.session_state.api_key_auth_failed = False
api_key_is_syntactically_valid = is_api_key_valid(st.session_state.get("openrouter_api_key"))
# Determine if setup is needed: Key missing, OR invalid format, OR previous auth failed
app_requires_api_key_setup = (
    not st.session_state.get("openrouter_api_key") or
    not api_key_is_syntactically_valid or
    st.session_state.get("api_key_auth_failed", False)
)

# -------------------- Main Application Rendering -------------------
if app_requires_api_key_setup:
    st.set_page_config(page_title="OpenRouter API Key Setup", layout="centered")
    load_custom_css()
    st.title("🔒 OpenRouter API Key Required")
    st.markdown("---", unsafe_allow_html=True)

    # Provide specific feedback based on why setup is needed
    if st.session_state.get("api_key_auth_failed"):
        st.error("API Key Authentication Failed. Please verify your key on OpenRouter.ai and re-enter.")
    elif not api_key_is_syntactically_valid and st.session_state.get("openrouter_api_key") is not None:
        st.error("The configured API Key has an invalid format. It must start with `sk-or-`.")
    elif not st.session_state.get("openrouter_api_key"):
         st.info("Please configure your OpenRouter API Key to use the application.")
    else: # Should not happen based on app_requires_api_key_setup logic, but as a fallback:
         st.info("API Key configuration required.")

    st.markdown( "You can get a key from [OpenRouter.ai Keys](https://openrouter.ai/keys). Enter it below to continue." )
    new_key_input_val = st.text_input("Enter OpenRouter API Key", type="password", key="api_key_setup_input", value="", placeholder="sk-or-...")

    if st.button("Save and Validate API Key", key="save_api_key_setup_button", use_container_width=True, type="primary"):
        if is_api_key_valid(new_key_input_val):
            st.session_state.openrouter_api_key = new_key_input_val
            _save_app_config(new_key_input_val)
            st.session_state.api_key_auth_failed = False # Reset flag before validation attempt
            with st.spinner("Validating API Key..."):
                fetched_credits_data = get_credits() # This function now updates api_key_auth_failed on 401

            if st.session_state.get("api_key_auth_failed"):
                # get_credits already set the flag and logged
                st.error("Authentication failed with the provided API Key. Please check the key and try again.")
                # No rerun here, let the user try again
            elif fetched_credits_data == (None, None, None):
                # Handle cases where get_credits failed for reasons other than 401
                 st.error("Could not validate API Key. Network issue or OpenRouter API problem? Key saved, but functionality may be affected.")
                 time.sleep(1.5)
                 st.rerun() # Rerun to leave the setup page but show potential issues in main app
            else:
                st.success("API Key saved and validated! Initializing application...")
                if "credits" not in st.session_state: st.session_state.credits = {}
                st.session_state.credits["total"], st.session_state.credits["used"], st.session_state.credits["remaining"] = fetched_credits_data
                st.session_state.credits_ts = time.time()
                time.sleep(1.0)
                st.rerun() # Success, proceed to the main app
        elif not new_key_input_val:
            st.warning("API Key field cannot be empty.")
        else:
            st.error("Invalid API key format. It must start with 'sk-or-'.")

    st.markdown("---", unsafe_allow_html=True); st.caption("Your API key is stored locally in `app_config.json`.")

# --- Main App Logic (Executes only if API key setup is NOT required) ---
else:
    st.set_page_config(page_title="OpenRouter Chat", layout="wide", initial_sidebar_state="expanded")
    load_custom_css()

    # Initialize session state variables if they don't exist
    if "settings_panel_open" not in st.session_state: st.session_state.settings_panel_open = False
    if "credits" not in st.session_state: st.session_state.credits = {"total": None, "used": None, "remaining": None}; st.session_state.credits_ts = 0

    needs_save_session = False
    if "sid" not in st.session_state:
        st.session_state.sid = _new_sid() # Creates and cleans blank sessions
        needs_save_session = True
        logging.info(f"Initialized new session ID: {st.session_state.sid}")
    elif st.session_state.sid not in sessions:
        logging.warning(f"Session ID {st.session_state.sid} not found in loaded sessions. Creating new chat.")
        st.session_state.sid = _new_sid()
        needs_save_session = True

    if needs_save_session:
        _save(SESS_FILE, sessions)
        # Don't rerun immediately after creating session, let the page load normally

    # --- Credit Refresh Logic ---
    # Refresh credits if:
    # 1. They are older than 1 hour (3600s)
    # 2. They have never been fetched (credits_ts is 0)
    # 3. They are currently None (e.g., previous fetch failed)
    credits_are_stale = time.time() - st.session_state.get("credits_ts", 0) > 3600
    credits_never_fetched = st.session_state.get("credits_ts", 0) == 0
    credits_are_none = any(st.session_state.credits.get(k) is None for k in ["total", "used", "remaining"])

    if credits_are_stale or credits_never_fetched or credits_are_none:
        logging.info(f"Refreshing credits (Stale: {credits_are_stale}, Never Fetched: {credits_never_fetched}, Are None: {credits_are_none}).")
        credits_data = get_credits() # This function handles auth failure state internally

        if st.session_state.get("api_key_auth_failed"):
            logging.error("API Key auth failed during scheduled credit refresh. Credits remain unchanged.")
            # Keep existing credit values (might be None or old)
            if credits_are_none: # Ensure structure exists even if fetch failed
                 st.session_state.credits = {"total": None, "used": None, "remaining": None}
            # Update timestamp to prevent immediate re-fetch attempts on auth failure
            st.session_state.credits_ts = time.time()
        elif credits_data != (None, None, None):
            st.session_state.credits["total"], st.session_state.credits["used"], st.session_state.credits["remaining"] = credits_data
            st.session_state.credits_ts = time.time()
            logging.info("Credits refreshed successfully.")
        else:
             # Fetch failed for non-auth reasons (network, etc.)
             logging.warning("Scheduled credit refresh failed (non-auth). Credits remain unchanged.")
             if credits_are_none: # Ensure structure exists
                 st.session_state.credits = {"total": None, "used": None, "remaining": None}
             # Update timestamp to prevent rapid re-fetches on transient errors
             st.session_state.credits_ts = time.time()

    # --- Sidebar Rendering ---
    with st.sidebar:
        settings_button_label = "⚙️ Close Settings" if st.session_state.settings_panel_open else "⚙️ Settings"
        if st.button(settings_button_label, key="toggle_settings_button_sidebar", use_container_width=True):
            st.session_state.settings_panel_open = not st.session_state.settings_panel_open; st.rerun()

        # --- Settings Panel (Conditional) ---
        if st.session_state.get("settings_panel_open"):
            st.markdown("<div class='settings-panel'>", unsafe_allow_html=True)
            st.subheader("🔑 API Key Configuration")
            current_api_key_in_panel = st.session_state.get("openrouter_api_key")
            if current_api_key_in_panel and len(current_api_key_in_panel) > 8: key_display = f"Current key: `sk-or-...{current_api_key_in_panel[-4:]}`"
            elif current_api_key_in_panel: key_display = "Current key: `sk-or-...` (short key)"
            else: key_display = "Current key: Not set"
            st.caption(key_display)
            if st.session_state.get("api_key_auth_failed"): st.error("Current API Key failed authentication.")

            new_key_input_sidebar = st.text_input("New OpenRouter API Key (optional)", type="password", key="api_key_sidebar_input", placeholder="sk-or-...")
            if st.button("Save New API Key", key="save_api_key_sidebar_button", use_container_width=True):
                if is_api_key_valid(new_key_input_sidebar):
                    st.session_state.openrouter_api_key = new_key_input_sidebar
                    _save_app_config(new_key_input_sidebar)
                    st.session_state.api_key_auth_failed = False # Reset before validation
                    with st.spinner("Validating new API key..."): credits_data = get_credits()

                    if st.session_state.get("api_key_auth_failed"): st.error("New API Key failed authentication.")
                    elif credits_data == (None,None,None): st.warning("Could not validate new API key (network/API issue?). Saved, but functionality may be affected.")
                    else:
                        st.success("New API Key saved and validated!")
                        st.session_state.credits["total"],st.session_state.credits["used"],st.session_state.credits["remaining"] = credits_data
                        st.session_state.credits_ts = time.time()
                    time.sleep(0.8); st.rerun()
                elif not new_key_input_sidebar: st.warning("API Key field empty. No changes made.")
                else: st.error("Invalid API key format. Must start with 'sk-or-'.")

            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("📊 Detailed Model Quotas")
            # Ensure quota data is loaded/current before displaying details
            _ensure_quota_data_is_current()

            # Iterate through models defined in NEW_PLAN_CONFIG for detailed view
            for m_key_loop in sorted(NEW_PLAN_CONFIG.keys()):
                # Skip if model isn't fully defined (shouldn't happen with current setup)
                if m_key_loop not in MODEL_MAP or m_key_loop not in EMOJI or m_key_loop not in MODEL_DESCRIPTIONS:
                    logging.warning(f"Skipping detailed quota display for incompletely defined model key: {m_key_loop}")
                    continue

                stats = get_quota_usage_and_limits(m_key_loop)
                if not stats:
                    st.markdown(f"**{EMOJI.get(m_key_loop, '')} {m_key_loop} ({MODEL_MAP.get(m_key_loop, 'N/A').split('/')[-1]})**: Could not retrieve quota details.")
                    continue

                # Extract short name robustly
                model_desc_full = MODEL_DESCRIPTIONS.get(m_key_loop, "")
                try:
                    model_short_name = model_desc_full.split('(')[1].split(')')[0] if '(' in model_desc_full else MODEL_MAP.get(m_key_loop, "Unknown").split('/')[-1]
                except IndexError:
                    model_short_name = MODEL_MAP.get(m_key_loop, "Unknown").split('/')[-1]

                model_name_display = f"{EMOJI.get(m_key_loop, '')} <span class='detailed-quota-modelname'>{m_key_loop} ({model_short_name})</span>"
                st.markdown(f"{model_name_display}", unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="detailed-quota-block">
                    <ul>
                        <li><b>Daily Msgs:</b> {stats['used_daily_msg']}/{stats['limit_daily_msg'] if stats['limit_daily_msg'] > 0 else '∞'}</li>
                        <li><b>Daily In Tok:</b> {format_token_count(stats['used_daily_in_tokens'])}/{format_token_count(stats['limit_daily_in_tokens']) if stats['limit_daily_in_tokens'] > 0 else '∞'}</li>
                        <li><b>Daily Out Tok:</b> {format_token_count(stats['used_daily_out_tokens'])}/{format_token_count(stats['limit_daily_out_tokens']) if stats['limit_daily_out_tokens'] > 0 else '∞'}</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="detailed-quota-block">
                    <ul>
                        <li><b>Monthly Msgs:</b> {stats['used_monthly_msg']}/{stats['limit_monthly_msg'] if stats['limit_monthly_msg'] > 0 else '∞'}</li>
                        <li><b>Monthly In Tok:</b> {format_token_count(stats['used_monthly_in_tokens'])}/{format_token_count(stats['limit_monthly_in_tokens']) if stats['limit_monthly_in_tokens'] > 0 else '∞'}</li>
                        <li><b>Monthly Out Tok:</b> {format_token_count(stats['used_monthly_out_tokens'])}/{format_token_count(stats['limit_monthly_out_tokens']) if stats['limit_monthly_out_tokens'] > 0 else '∞'}</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)

                # Display 3-hour limit details only for Model A if configured
                if m_key_loop == "A" and stats["limit_3hr_msg"] != float('inf') and NEW_PLAN_CONFIG["A"][7] > 0:
                    time_until_next_msg_str = ""
                    # Use the already fetched/pruned list from _g_quota_data if available
                    active_model_a_calls = sorted(_g_quota_data.get(MODEL_A_3H_CALLS_KEY, []))
                    if len(active_model_a_calls) >= stats['limit_3hr_msg']:
                         # Find the timestamp that needs to expire for a new call to be allowed
                         if active_model_a_calls: # Ensure list is not empty
                             # Index from the end to find the oldest call that *counts* towards the limit
                             oldest_blocking_call_idx = max(0, len(active_model_a_calls) - int(stats['limit_3hr_msg']))
                             oldest_blocking_call_ts = active_model_a_calls[oldest_blocking_call_idx]
                             expiry_time = oldest_blocking_call_ts + NEW_PLAN_CONFIG["A"][7] # Use configured window
                             time_remaining_seconds = expiry_time - time.time()
                             if time_remaining_seconds > 0:
                                mins, secs = divmod(int(time_remaining_seconds), 60)
                                hrs, mins_rem = divmod(mins, 60)
                                if hrs > 0:
                                    time_until_next_msg_str = f" (Next in {hrs}h {mins_rem}m)"
                                elif mins_rem > 0:
                                    time_until_next_msg_str = f" (Next in {mins_rem}m {secs}s)"
                                else:
                                     time_until_next_msg_str = f" (Next in {secs}s)"

                    st.markdown(f"""
                    <div class="detailed-quota-block" style="margin-top: -0.5rem; margin-left:0.1rem;">
                    <ul>
                    <li><b>3-Hour Msgs:</b> {stats['used_3hr_msg']}/{int(stats['limit_3hr_msg'])}{time_until_next_msg_str}</li>
                    </ul>
                    </div>""", unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True) # End settings-panel

        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

        # --- Sidebar Header ---
        st.markdown("<div class='sidebar-title-container'>", unsafe_allow_html=True)
        logo_title_cols = st.columns([1, 5], gap="small")
        with logo_title_cols[0]: st.image("https://avatars.githubusercontent.com/u/130328222?s=200&v=4", width=40)
        with logo_title_cols[1]: st.title("OpenRouter Chat")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

        # --- Daily Quota Gauges ---
        with st.expander("⚡ DAILY MODEL QUOTAS", expanded=True):
            # Display gauges for models defined in NEW_PLAN_CONFIG
            active_model_keys_for_display = sorted([k for k in NEW_PLAN_CONFIG.keys() if k in MODEL_MAP and k in EMOJI]) # Ensure they are fully defined
            if not active_model_keys_for_display:
                st.caption("No quota-tracked models configured.")
            else:
                _ensure_quota_data_is_current() # Ensure data is fresh before display
                 # Adjust number of columns based on number of models for better layout
                num_models = len(active_model_keys_for_display)
                num_cols = min(num_models, 6) # Max 6 columns for quotas
                quota_cols = st.columns(num_cols)


                for i, m_key in enumerate(active_model_keys_for_display):
                    with quota_cols[i % num_cols]: # Cycle through columns
                        stats = get_quota_usage_and_limits(m_key)
                        if not stats: # Handle case where quota data couldn't be retrieved
                             left_d_msgs, lim_d_msgs = 0, 0
                        else:
                            left_d_msgs = max(0, stats["limit_daily_msg"] - stats["used_daily_msg"])
                            lim_d_msgs = stats["limit_daily_msg"]

                        if lim_d_msgs > 0:
                            pct_float = max(0.0, min(1.0, left_d_msgs / lim_d_msgs))
                            fill_width_val = int(pct_float * 100)
                            left_display = str(left_d_msgs)
                        else: # Handle models with zero daily limit (e.g., monthly only or unlimited)
                            pct_float, fill_width_val, left_display = 1.0, 100, "∞" # Display infinity if limit is 0

                        # Determine bar color based on percentage remaining
                        bar_color = "var(--app-danger-color)" # Default red
                        if lim_d_msgs <= 0: # If limit is 0 or less, show as green/full
                            bar_color = "var(--app-success-color)"
                        elif pct_float > 0.5: bar_color = "var(--app-success-color)"
                        elif pct_float > 0.25: bar_color = "var(--app-warning-color)"

                        emoji_char = EMOJI.get(m_key, "❔")
                        tooltip_text = f"{left_d_msgs} / {lim_d_msgs if lim_d_msgs > 0 else '∞'} Daily Messages Remaining"
                        st.markdown(f"""<div class="compact-quota-item" title="{tooltip_text}">
                                            <div class="cq-info">{emoji_char} <b>{m_key}</b></div>
                                            <div class="cq-bar-track"><div class="cq-bar-fill" style="width: {fill_width_val}%; background-color: {bar_color};"></div></div>
                                            <div class="cq-value" style="color: {bar_color};">{left_display}</div>
                                        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

        # --- New Chat Button ---
        # Disable 'New chat' only if the *current* session is truly blank
        current_session_is_truly_blank = (st.session_state.sid in sessions and
                                          sessions[st.session_state.sid].get("title") == "New chat" and
                                          not sessions[st.session_state.sid].get("messages"))
        if st.button("➕ New chat", key="new_chat_button_top", use_container_width=True, disabled=current_session_is_truly_blank):
            st.session_state.sid = _new_sid() # Creates new, saves sessions implicitly
            _save(SESS_FILE, sessions) # Explicit save might be redundant but ensures state
            st.rerun()

        # --- Chat History List ---
        st.subheader("Chats")
        # Filter and sort sessions robustly
        valid_sids = [s for s in sessions.keys() if isinstance(s, str) and s.isdigit()]
        sorted_sids = sorted(valid_sids, key=lambda s: int(s), reverse=True)

        for sid_key in sorted_sids:
            if sid_key not in sessions: continue # Should not happen, but safety check
            session_data = sessions[sid_key]
            title = session_data.get("title", f"Chat {sid_key}") # Fallback title
            # Use same truncation logic as autoname for display consistency
            display_title = (title[:30] + "…") if len(title) > 30 else title
            is_active_chat = (st.session_state.sid == sid_key)

            if st.button(display_title, key=f"session_button_{sid_key}", use_container_width=True, disabled=is_active_chat):
                if not is_active_chat:
                    # Before switching, delete any *other* blank sessions if the current one wasn't blank
                    current_session_data = sessions.get(st.session_state.sid, {})
                    current_session_was_blank = (current_session_data.get("title") == "New chat" and not current_session_data.get("messages"))
                    if not current_session_was_blank:
                         _delete_unused_blank_sessions(keep_sid=sid_key) # Delete other blanks
                    st.session_state.sid = sid_key
                    _save(SESS_FILE, sessions) # Save potentially cleaned sessions
                    st.rerun()

        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

        # --- Model Info ---
        st.subheader("Model Info & Costs") # Renamed for clarity
        st.caption(f"Router: {ROUTER_MODEL_ID.split('/')[-1]}")
        with st.expander("Cost Order: F < D < B < A < E < C", expanded=False): # Show cost order in title
            # Define the cost order explicitly
            cost_order = ["F", "D", "B", "A", "E", "C"]
            # Iterate through models in cost order
            for k_model in cost_order:
                 if k_model not in MODEL_MAP: continue # Skip if not configured

                 # Safely extract description parts
                 desc_full = MODEL_DESCRIPTIONS.get(k_model, MODEL_MAP.get(k_model, "N/A"))
                 try:
                     desc_parts = desc_full.split("(")
                     main_desc = desc_parts[0].strip()
                     model_name_in_desc = desc_parts[1].split(")")[0] if len(desc_parts) > 1 and ')' in desc_parts[1] else MODEL_MAP.get(k_model, "N/A").split('/')[-1]
                 except IndexError:
                     main_desc = desc_full # Use full description if parsing fails
                     model_name_in_desc = MODEL_MAP.get(k_model, "N/A").split('/')[-1]

                 max_tok = MAX_TOKENS.get(k_model, 0)
                 emoji_char = EMOJI.get(k_model, '') # Get emoji
                 st.markdown(f"**{emoji_char} {k_model}**: {main_desc} ({model_name_in_desc}) <br><small style='color:var(--app-text-secondary-color);'>Max Output: {max_tok:,} tokens</small>", unsafe_allow_html=True)

            # Add Fallback model info at the end
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(f"**{FALLBACK_MODEL_KEY}**: {FALLBACK_MODEL_EMOJI} {FALLBACK_MODEL_ID.split('/')[-1]} <br><small style='color:var(--app-text-secondary-color);'>Max Output: {FALLBACK_MODEL_MAX_TOKENS:,} tokens (Free; used when quotas exhausted or routing fails)</small>", unsafe_allow_html=True)

        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

        # --- Account Credits ---
        with st.expander("Account stats (credits)", expanded=False):
            if st.button("Refresh Credits", key="refresh_credits_button_sidebar"):
                 with st.spinner("Refreshing credits..."):
                     credits_data = get_credits() # Handles auth state internally
                 if st.session_state.get("api_key_auth_failed"):
                     st.error("API Key authentication failed. Cannot refresh credits.")
                 elif credits_data != (None,None,None):
                     st.session_state.credits["total"],st.session_state.credits["used"],st.session_state.credits["remaining"] = credits_data
                     st.session_state.credits_ts = time.time()
                     st.success("Credits refreshed!")
                     st.rerun() # Rerun to show updated values immediately
                 else:
                      st.warning("Could not refresh credits (network/API issue?).")
                 # No rerun on failure, keep expander open

            tot, used, rem = st.session_state.credits.get("total"), st.session_state.credits.get("used"), st.session_state.credits.get("remaining")

            if st.session_state.get("api_key_auth_failed"):
                 st.warning("Cannot display credits. API Key failed authentication.")
            elif tot is None or used is None or rem is None:
                 st.warning("Could not fetch/display credits (check API key and network).")
            else:
                 # Format credits, handling potential non-numeric values gracefully
                 try: rem_f = f"${float(rem):.2f} cr"
                 except (ValueError, TypeError): rem_f = "N/A"
                 try: used_f = f"${float(used):.2f} cr"
                 except (ValueError, TypeError): used_f = "N/A"
                 st.markdown(f"**Remaining:** {rem_f}<br>**Used:** {used_f}", unsafe_allow_html=True)

            ts = st.session_state.get("credits_ts", 0)
            last_updated_str = datetime.fromtimestamp(ts, TZ).strftime('%-d %b, %H:%M') if ts else "Never"
            st.caption(f"Last updated: {last_updated_str}")

    # ---- Main chat area ----
    if st.session_state.sid not in sessions:
        logging.error(f"CRITICAL: Current SID {st.session_state.sid} missing from sessions data. Resetting to new chat.")
        st.session_state.sid = _new_sid()
        _save(SESS_FILE, sessions)
        st.rerun()

    current_sid = st.session_state.sid
    # Ensure messages list exists for the current session
    if "messages" not in sessions[current_sid]:
         sessions[current_sid]["messages"] = []
         logging.warning(f"Initialized missing 'messages' list for session {current_sid}.")
         _save(SESS_FILE, sessions) # Save the fix

    chat_history = sessions[current_sid]["messages"]

    # --- Display Existing Chat Messages ---
    for msg_idx, msg in enumerate(chat_history): # Use enumerate for potential keys
        role = msg.get("role", "assistant")
        avatar_char = None
        if role == "user":
            avatar_char = "👤"
        elif role == "assistant":
            m_key = msg.get("model") # Get the model key used for this message
            if m_key == FALLBACK_MODEL_KEY: avatar_char = FALLBACK_MODEL_EMOJI
            elif m_key in EMOJI: avatar_char = EMOJI[m_key]
            else: avatar_char = "🤖" # Default assistant avatar if model unknown
        else: # Handle potential system messages or other roles if added later
             role="assistant" # Display as assistant for now
             avatar_char = "⚙️"

        with st.chat_message(role, avatar=avatar_char):
             # Removed the 'key' argument which caused the TypeError
             st.markdown(msg.get("content", "*empty message*"))

    # --- Chat Input Logic ---
    if prompt := st.chat_input("Ask anything…", key=f"chat_input_{current_sid}"):
        # Append user message immediately and display it
        user_message = {"role":"user","content":prompt}
        chat_history.append(user_message)
        with st.chat_message("user", avatar="👤"): st.markdown(prompt)
        _save(SESS_FILE, sessions) # Save user message immediately

        # Check API key status before attempting routing/API call
        if st.session_state.get("api_key_auth_failed"):
            st.error("API Key Authentication Failed. Please correct it in ⚙️ Settings.")
            st.stop() # Stop execution for this run
        if not is_api_key_valid(st.session_state.get("openrouter_api_key")):
             st.error("OpenRouter API Key is invalid or not configured. Please set it in ⚙️ Settings.")
             st.stop() # Stop execution

        # --- Model Selection Logic ---
        routing_start_time = time.time()
        with st.spinner("Selecting best model..."): # More specific spinner text
            _ensure_quota_data_is_current() # Refresh quota view

            # *** ADDED LOGGING HERE ***
            logging.info("--- Checking Model Availability Before Routing ---")
            allowed_standard_models = []
            # Check models in cost order F -> D -> B -> A -> E -> C
            cost_order_check = ["F", "D", "B", "A", "E", "C"]
            for k_map in cost_order_check:
                 if k_map in MODEL_MAP: # Only check models defined in our map
                     available = is_model_available(k_map)
                     logging.info(f"Model {k_map} ({MODEL_MAP.get(k_map,'?') }): Available = {available}") # Added get default
                     if available:
                          allowed_standard_models.append(k_map)
            logging.info(f"Final allowed models passed to router: {allowed_standard_models}")
            # *** END ADDED LOGGING ***

            use_fallback = False
            chosen_model_key = None
            model_id_to_use = None
            max_tokens_api = FALLBACK_MODEL_MAX_TOKENS
            avatar_resp = "🤖"

            # Call the router function
            routed_key_letter = route_choice(prompt, allowed_standard_models, chat_history)
            routing_end_time = time.time()
            logging.info(f"Routing took {routing_end_time - routing_start_time:.2f} seconds.")

            # Handle router failure due to auth (route_choice returns None)
            if routed_key_letter is None and st.session_state.get("api_key_auth_failed"):
                 st.error("API Authentication failed during model routing. Please check the key in ⚙️ Settings.")
                 st.stop()

            # Process the router's decision
            if routed_key_letter == FALLBACK_MODEL_KEY:
                logging.info(f"Router chose or fell back to {FALLBACK_MODEL_KEY}. Using free fallback: {FALLBACK_MODEL_ID}.")
                use_fallback = True
                chosen_model_key = FALLBACK_MODEL_KEY
                model_id_to_use = FALLBACK_MODEL_ID
                max_tokens_api = FALLBACK_MODEL_MAX_TOKENS
                avatar_resp = FALLBACK_MODEL_EMOJI
            elif routed_key_letter in MODEL_MAP: # Check if the returned key is a valid configured model
                # Final check: is the chosen model *still* available? (Quota might update mid-request?)
                if is_model_available(routed_key_letter):
                    chosen_model_key = routed_key_letter
                    model_id_to_use = MODEL_MAP[chosen_model_key]
                    max_tokens_api = MAX_TOKENS.get(chosen_model_key, FALLBACK_MODEL_MAX_TOKENS)
                    avatar_resp = EMOJI.get(chosen_model_key, "🤖")
                    logging.info(f"Router selected valid and available model: {chosen_model_key} ({model_id_to_use})")
                else:
                    # Router chose a model that's no longer available (e.g., quota hit just now)
                    logging.warning(f"Router chose '{routed_key_letter}', but it's no longer available (quota likely hit). Using free fallback {FALLBACK_MODEL_ID}.")
                    use_fallback = True
                    chosen_model_key = FALLBACK_MODEL_KEY
                    model_id_to_use = FALLBACK_MODEL_ID
                    max_tokens_api = FALLBACK_MODEL_MAX_TOKENS
                    avatar_resp = FALLBACK_MODEL_EMOJI
            else:
                 # Router returned something unexpected or failed silently (should have returned FALLBACK_MODEL_KEY in failure cases)
                 logging.error(f"Router returned unexpected key '{routed_key_letter}' or failed silently. Using free fallback {FALLBACK_MODEL_ID}.")
                 use_fallback = True
                 chosen_model_key = FALLBACK_MODEL_KEY
                 model_id_to_use = FALLBACK_MODEL_ID
                 max_tokens_api = FALLBACK_MODEL_MAX_TOKENS
                 avatar_resp = FALLBACK_MODEL_EMOJI

            # --- API Call and Response Streaming ---
            if model_id_to_use: # Ensure we have a model ID to use
                # Display the thinking animation with the chosen model's avatar
                with st.chat_message("assistant", avatar=avatar_resp):
                    response_placeholder = st.empty()
                    response_placeholder.markdown("Thinking... 💭") # Initial thinking message
                    full_response = ""
                    api_call_ok = True
                    error_message_from_stream = None
                    stream_start_time = time.time()

                    # Stream the response
                    for chunk_content, err_msg in streamed(model_id_to_use, chat_history, max_tokens_api):
                        if err_msg: # Handle errors reported by the stream function
                            logging.error(f"Error during streaming for model {model_id_to_use}: {err_msg}")
                            # Check if the error indicates auth failure - stream function already sets the flag
                            if st.session_state.get("api_key_auth_failed"):
                                 error_message_from_stream = "❗ **API Authentication Error**: Update Key in ⚙️ Settings."
                            else:
                                # Provide a more user-friendly error
                                if "rate limit" in err_msg.lower():
                                     error_message_from_stream = f"❗ **API Rate Limit**: The model provider reported a rate limit error. Please try again later. ({model_id_to_use})"
                                elif "context_length_exceeded" in err_msg.lower():
                                     error_message_from_stream = f"❗ **Context Length Exceeded**: The conversation history is too long for this model ({model_id_to_use}). Please start a new chat or try a model with a larger context window."
                                else:
                                    error_message_from_stream = f"❗ **API Error**: {err_msg}"
                            api_call_ok = False
                            break # Stop processing chunks on error
                        if chunk_content is not None:
                            full_response += chunk_content
                            # Add animated cursor effect while streaming
                            response_placeholder.markdown(full_response + "▌")

                    stream_end_time = time.time()
                    logging.info(f"Streaming took {stream_end_time - stream_start_time:.2f} seconds.")

                    # Final display of the message (error or full response)
                    if error_message_from_stream:
                         response_placeholder.markdown(error_message_from_stream)
                         full_response = error_message_from_stream # Store error as the content
                    elif not full_response and api_call_ok:
                         # Handle cases where the stream finished successfully but returned no content
                         response_placeholder.markdown("*Assistant returned an empty response.*")
                         full_response = "" # Ensure it's an empty string
                         logging.warning(f"Model {model_id_to_use} returned an empty response.")
                    else:
                         response_placeholder.markdown(full_response) # Display final response without cursor

                # --- Post-Response Processing ---
                # Retrieve usage data stored by the `streamed` function
                last_usage = st.session_state.pop("last_stream_usage", None)
                prompt_tokens_used = 0
                completion_tokens_used = 0

                if api_call_ok and last_usage:
                    prompt_tokens_used = last_usage.get("prompt_tokens", 0)
                    completion_tokens_used = last_usage.get("completion_tokens", 0) if full_response else 0 # Don't count completion tokens if response was empty
                    logging.info(f"API call completed for model {model_id_to_use}. Usage recorded: P={prompt_tokens_used}, C={completion_tokens_used}")
                elif api_call_ok and not last_usage:
                    logging.warning(f"Token usage info not found in stream response for model {model_id_to_use}.")
                # If api_call_ok is False, tokens remain 0

                # Append assistant message to history
                assistant_message = {
                    "role": "assistant",
                    "content": full_response, # Store the final content (could be error message or empty)
                    "model": chosen_model_key if api_call_ok else FALLBACK_MODEL_KEY, # Log the intended model if OK, else fallback
                    "prompt_tokens": prompt_tokens_used,
                    "completion_tokens": completion_tokens_used
                }
                chat_history.append(assistant_message)

                # Record quota usage ONLY if the API call was successful AND it wasn't the free fallback model
                if api_call_ok and not use_fallback and chosen_model_key in NEW_PLAN_CONFIG:
                     # Only record if completion tokens > 0 or prompt tokens > 0 (don't record for empty responses)
                     if prompt_tokens_used > 0 or completion_tokens_used > 0:
                          record_use(chosen_model_key, prompt_tokens_used, completion_tokens_used)
                     else:
                          logging.info(f"Skipping quota recording for {chosen_model_key} due to zero token usage (likely empty response).")


                # Auto-title the chat if it's new and we got a successful, non-error, non-empty response
                if api_call_ok and not error_message_from_stream and full_response and sessions[current_sid]["title"] == "New chat":
                   sessions[current_sid]["title"] = _autoname(prompt) # Title based on user prompt

                _save(SESS_FILE, sessions) # Save updated history, title, and usage
                st.rerun() # Rerun to refresh display (quota bars, chat list title, new message)

            # Handle case where no model_id_to_use was determined (should be rare)
            else:
                 if not st.session_state.get("api_key_auth_failed"): # Avoid double error message
                    st.error("Unexpected error: Could not determine a model to use. Please check logs or try again.")
                    logging.error("Failed to determine model_id_to_use after selection logic, and no API auth failure detected.")
                 _save(SESS_FILE, sessions) # Save history up to this point
                 st.stop() # Prevent potential infinite loop if rerun happens without state change
