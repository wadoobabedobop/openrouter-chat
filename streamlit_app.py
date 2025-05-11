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
    "F": "google/gemini-2.5-flash-preview"   # Gemini 2.5 Flash
}
ROUTER_MODEL_ID = "google/gemini-2.0-flash-exp:free"
MAX_HISTORY_CHARS_FOR_ROUTER = 3000  # Approx. 750 tokens for history context

MAX_TOKENS = { # Per-call max_tokens for API request (max output generation length)
    "A": 16_000, "B": 8_000, "C": 16_000,
    "D": 8_000, "F": 8_000
}

# NEW QUOTA CONFIGURATION based on the provided table
# (daily_msg, monthly_msg, daily_in_tokens, monthly_in_tokens, daily_out_tokens, monthly_out_tokens, 3hr_msg_limit, 3hr_window_seconds)
NEW_PLAN_CONFIG = {
    "A": (10, 200, 5000, 100000, 5000, 100000, 3, 3 * 3600),  # Gemini 2.5 Pro
    "B": (10, 200, 5000, 100000, 5000, 100000, 0, 0),          # GPT-4o mini
    "C": (10, 200, 5000, 100000, 5000, 100000, 0, 0),          # GPT-4o
    "D": (10, 200, 5000, 100000, 5000, 100000, 0, 0),          # DeepSeek R1
    "F": (150, 3000, 75000, 1500000, 75000, 1500000, 0, 0)     # Gemini 2.5 Flash
}
# Old PLAN dictionary is removed/replaced by NEW_PLAN_CONFIG

EMOJI = {
    "A": "🌟", "B": "🔷", "C": "🟥", "D": "🟢", "F": "🌀"
}

# User-facing descriptions (unchanged)
MODEL_DESCRIPTIONS = {
    "A": "🌟 (gemini-2.5-pro-preview) – top-quality, creative, expensive.",
    "B": "🔷 (o4-mini) – mid-stakes reasoning, cost-effective.",
    "C": "🟥 (chatgpt-4o-latest) – polished/empathetic, pricier.",
    "D": "🟢 (deepseek-r1) – cheap factual reasoning.",
    "F": "🌀 (gemini-2.5-flash-preview) – quick, free-tier, general purpose."
}

# *** MODIFIED FOR ROUTER ANONYMIZATION ***
# These descriptions are for the ROUTER_MODEL_ID. They are anonymized regarding specific model names.
ROUTER_MODEL_GUIDANCE = {
    "A": "(Model A: Top-Tier Quality & Capability) Use for EXTREMELY complex, multi-step reasoning; highly advanced creative generation (e.g., novel excerpts, sophisticated poetry); tasks demanding cutting-edge knowledge and deep nuanced understanding. HIGHEST COST. CHOOSE ONLY if query explicitly demands top-tier, 'genius-level' output AND cheaper models are CLEARLY insufficient. Avoid for anything less.",
    "B": "(Model B: Solid Mid-Tier All-Rounder) Use for general purpose chat; moderate complexity reasoning; summarization; drafting emails/content; brainstorming; standard instruction following. Good balance of capability and cost. MODERATE COST. CHOOSE if 'F' or 'D' are too basic, AND 'A' or 'C' are overkill/not strictly necessary for the task's core requirements.",
    "C": "(Model C: High Quality, Polished & Empathetic) Use for tasks requiring highly polished, empathetic, or very human-like conversational interactions; complex multi-turn instruction adherence where its specific stylistic strengths are key; creative content generation with a defined sophisticated tone. HIGHER COST. CHOOSE ONLY if query *specifically* benefits from its unique interaction style or demands exceptional refinement AND 'B' is clearly inadequate.",
    "D": "(Model D: Cost-Effective Factual & Technical) Use for factual Q&A; code generation/explanation/debugging; data extraction; straightforward logical reasoning; technical or scientific queries. LOW COST. CHOOSE for tasks that are well-defined, benefit from specialized reasoning, and do not require broad world knowledge, deep creativity, or nuanced conversation. Very slow responses. Prefer over B for these specific tasks if cost is a factor.",
    "F": "(Model F: Fast & Economical for Simple Tasks) Use for very quick, simple Q&A; fast summarization of short texts; basic classification; brief translations; or when speed is paramount and task complexity is very low. LOWEST COST. Default starting point for most simple requests."
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
# def _yweek():    return datetime.now(TZ).strftime("%G-%V") # Weekly no longer used
def _ymonth():   return datetime.now(TZ).strftime("%Y-%m")

def _load_app_config():
    return _load(CONFIG_FILE, {})

def _save_app_config(api_key_value: str):
    config_data = _load_app_config()
    config_data["openrouter_api_key"] = api_key_value
    _save(CONFIG_FILE, config_data)


# --------------------- Quota Management (Revised) ------------------------
_g_quota_data = None
_g_quota_data_last_refreshed_stamps = {"d": None, "m": None} # Removed "w"

# Define all usage keys for clarity
USAGE_KEYS_PERIODIC = ["d_u", "m_u", "d_it_u", "m_it_u", "d_ot_u", "m_ot_u"]
MODEL_A_3H_CALLS_KEY = "model_A_3h_calls"


def _reset(block: dict, period_prefix: str, current_stamp: str, model_keys_zeros: dict) -> bool:
    """ Resets usage data for a given period (d, m) if the timestamp has changed. """
    data_changed = False
    period_stamp_key = period_prefix # e.g., "d" or "m"
    
    if block.get(period_stamp_key) != current_stamp:
        block[period_stamp_key] = current_stamp
        # Reset all usage types for this period
        for usage_type_suffix in ["_u", "_it_u", "_ot_u"]: # messages, input tokens, output tokens
            usage_dict_key = f"{period_prefix}{usage_type_suffix}"
            block[usage_dict_key] = model_keys_zeros.copy()
        data_changed = True
        logging.info(f"Quota period '{period_stamp_key}' reset for new stamp '{current_stamp}'.")
    else:
        # Ensure all usage dictionaries exist and have all current models
        for usage_type_suffix in ["_u", "_it_u", "_ot_u"]:
            usage_dict_key = f"{period_prefix}{usage_type_suffix}"
            if usage_dict_key not in block:
                block[usage_dict_key] = model_keys_zeros.copy()
                data_changed = True
                logging.info(f"Initialized missing usage dict '{usage_dict_key}' for stamp '{current_stamp}'.")
            else:
                # Ensure all models are in this existing usage dict
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
        # Still prune Model A's 3h calls even if no full refresh
        if MODEL_A_3H_CALLS_KEY in _g_quota_data and "A" in NEW_PLAN_CONFIG and NEW_PLAN_CONFIG["A"][7] > 0:
            _, _, _, _, _, _, _, three_hr_window_seconds = NEW_PLAN_CONFIG["A"]
            current_time = time.time()
            original_len = len(_g_quota_data[MODEL_A_3H_CALLS_KEY])
            _g_quota_data[MODEL_A_3H_CALLS_KEY] = [
                ts for ts in _g_quota_data[MODEL_A_3H_CALLS_KEY]
                if current_time - ts < three_hr_window_seconds
            ]
            if len(_g_quota_data[MODEL_A_3H_CALLS_KEY]) != original_len:
                logging.info(f"Pruned Model A 3-hour call timestamps. Original: {original_len}, New: {len(_g_quota_data[MODEL_A_3H_CALLS_KEY])}.")
                # No save here, will be saved on next record_use or if other changes occur
        return _g_quota_data

    q_loaded_data = _load(QUOTA_FILE, {})
    data_was_modified = _g_quota_data is None # Mark modified if loaded for the first time
    
    active_model_keys = set(MODEL_MAP.keys())
    cleaned_during_load = False

    # Clean obsolete models from all usage dictionaries
    for usage_key_template in USAGE_KEYS_PERIODIC: # e.g., "d_u", "m_it_u" etc.
        if usage_key_template in q_loaded_data:
            current_period_usage_dict = q_loaded_data[usage_key_template]
            keys_in_usage = list(current_period_usage_dict.keys()) # Iterate over a copy
            for model_key_in_usage in keys_in_usage:
                if model_key_in_usage not in active_model_keys:
                    try:
                        del current_period_usage_dict[model_key_in_usage]
                        logging.info(f"Removed obsolete model key '{model_key_in_usage}' from quota usage '{usage_key_template}'.")
                        cleaned_during_load = True
                    except KeyError: pass # Should not happen if keys_in_usage is from the dict
    if cleaned_during_load: data_was_modified = True

    # Remove old weekly quota fields if they exist
    if "w" in q_loaded_data: del q_loaded_data["w"]; data_was_modified = True; logging.info("Removed obsolete 'w' field from quota data.")
    if "w_u" in q_loaded_data: del q_loaded_data["w_u"]; data_was_modified = True; logging.info("Removed obsolete 'w_u' field from quota data.")

    current_model_zeros = {k: 0 for k in MODEL_MAP.keys()}
    reset_occurred_d = _reset(q_loaded_data, "d", now_d_stamp, current_model_zeros)
    reset_occurred_m = _reset(q_loaded_data, "m", now_m_stamp, current_model_zeros)
    if reset_occurred_d or reset_occurred_m: data_was_modified = True

    # Initialize or prune Model A's 3h call list
    if MODEL_A_3H_CALLS_KEY not in q_loaded_data:
        q_loaded_data[MODEL_A_3H_CALLS_KEY] = []
        # data_was_modified = True # Not strictly modification if it's just init
    if "A" in NEW_PLAN_CONFIG and NEW_PLAN_CONFIG["A"][7] > 0:
        _, _, _, _, _, _, _, three_hr_window_seconds = NEW_PLAN_CONFIG["A"]
        current_time = time.time()
        original_len = len(q_loaded_data.get(MODEL_A_3H_CALLS_KEY, []))
        q_loaded_data[MODEL_A_3H_CALLS_KEY] = [
            ts for ts in q_loaded_data.get(MODEL_A_3H_CALLS_KEY, [])
            if current_time - ts < three_hr_window_seconds
        ]
        if len(q_loaded_data[MODEL_A_3H_CALLS_KEY]) != original_len:
             logging.info(f"Pruned Model A 3-hour call timestamps during full refresh. Original: {original_len}, New: {len(q_loaded_data[MODEL_A_3H_CALLS_KEY])}.")
             data_was_modified = True


    if data_was_modified:
        _save(QUOTA_FILE, q_loaded_data)
        logging.info("Quota data was modified (loaded/cleaned/reset/pruned) and saved to disk.")
    
    _g_quota_data = q_loaded_data
    _g_quota_data_last_refreshed_stamps = {"d": now_d_stamp, "m": now_m_stamp}
    return _g_quota_data

def get_quota_usage_and_limits(model_key: str):
    """
    Returns a dictionary with current usage and limits for a model.
    Keys: 'used_daily_msg', 'limit_daily_msg', 'used_monthly_msg', 'limit_monthly_msg',
          'used_daily_in_tokens', 'limit_daily_in_tokens', etc.
          'used_3hr_msg', 'limit_3hr_msg' (if applicable)
    """
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

    if model_key == "A" and plan[6] > 0: # Model A with 3hr cap
        # Pruning is handled in _ensure_quota_data_is_current
        usage["used_3hr_msg"] = len(current_q_data.get(MODEL_A_3H_CALLS_KEY, []))

    return {**usage, **limits}


def is_model_available(model_key: str) -> bool:
    """Checks if a model is available based on all its quotas."""
    if model_key not in NEW_PLAN_CONFIG:
        logging.warning(f"is_model_available: Model key '{model_key}' not in NEW_PLAN_CONFIG. Assuming unavailable.")
        return False

    stats = get_quota_usage_and_limits(model_key)
    if not stats: return False # Error getting stats

    # Check all limits
    if stats["used_daily_msg"] >= stats["limit_daily_msg"]: return False
    if stats["used_monthly_msg"] >= stats["limit_monthly_msg"]: return False
    if stats["used_daily_in_tokens"] >= stats["limit_daily_in_tokens"]: return False
    if stats["used_monthly_in_tokens"] >= stats["limit_monthly_in_tokens"]: return False
    if stats["used_daily_out_tokens"] >= stats["limit_daily_out_tokens"]: return False
    if stats["used_monthly_out_tokens"] >= stats["limit_monthly_out_tokens"]: return False
    
    if model_key == "A" and stats["limit_3hr_msg"] != float('inf'): # Check 3hr cap for Model A
        if stats["used_3hr_msg"] >= stats["limit_3hr_msg"]: return False
            
    return True

def get_remaining_daily_messages(model_key: str) -> int:
    """Helper for UI display, returns remaining daily messages."""
    if model_key not in NEW_PLAN_CONFIG: return 0
    stats = get_quota_usage_and_limits(model_key)
    if not stats: return 0
    return max(0, stats["limit_daily_msg"] - stats["used_daily_msg"])


def record_use(model_key: str, prompt_tokens: int, completion_tokens: int):
    if model_key not in MODEL_MAP: # Only track for standard, quota-enabled models
        logging.warning(f"Attempted to record usage for non-standard or unknown model key: {model_key}")
        return

    current_q_data = _ensure_quota_data_is_current() # Ensures data is fresh and dicts exist

    # Increment message counts
    current_q_data.setdefault("d_u", {}).setdefault(model_key, 0)
    current_q_data["d_u"][model_key] += 1
    current_q_data.setdefault("m_u", {}).setdefault(model_key, 0)
    current_q_data["m_u"][model_key] += 1

    # Increment token counts
    current_q_data.setdefault("d_it_u", {}).setdefault(model_key, 0)
    current_q_data["d_it_u"][model_key] += prompt_tokens
    current_q_data.setdefault("m_it_u", {}).setdefault(model_key, 0)
    current_q_data["m_it_u"][model_key] += prompt_tokens

    current_q_data.setdefault("d_ot_u", {}).setdefault(model_key, 0)
    current_q_data["d_ot_u"][model_key] += completion_tokens
    current_q_data.setdefault("m_ot_u", {}).setdefault(model_key, 0)
    current_q_data["m_ot_u"][model_key] += completion_tokens

    # Handle Model A 3-hour cap
    if model_key == "A" and NEW_PLAN_CONFIG["A"][6] > 0:
        current_q_data.setdefault(MODEL_A_3H_CALLS_KEY, []).append(time.time())
        # Pruning of this list is done in _ensure_quota_data_is_current

    _save(QUOTA_FILE, current_q_data)
    logging.info(f"Recorded usage for model '{model_key}': 1 msg, {prompt_tokens}p, {completion_tokens}c tokens. Quotas saved.")

# --------------------- Session Management -----------------------
def _delete_unused_blank_sessions(keep_sid: str = None):
    sids_to_delete = []
    for sid, data in list(sessions.items()):
        if sid == keep_sid: continue
        if data.get("title") == "New chat" and not data.get("messages"):
            sids_to_delete.append(sid)
    if sids_to_delete:
        for sid_del in sids_to_delete:
            logging.info(f"Auto-deleting blank session: {sid_del}")
            try: del sessions[sid_del]
            except KeyError: logging.warning(f"Session {sid_del} already deleted, skipping.")
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

# --------------------------- Logging ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)
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
        return response
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            st.session_state.api_key_auth_failed = True
            logging.error(f"API POST failed with 401 (Unauthorized): {e.response.text}")
        else: logging.error(f"API POST failed with {e.response.status_code}: {e.response.text}")
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
                
                if "error" in chunk:
                    msg = chunk["error"].get("message", "Unknown API error")
                    logging.error(f"API chunk error: {msg}"); yield None, msg; return

                # Capture usage info if present in the chunk (OpenRouter sends it in the last data message)
                if "usage" in chunk and chunk["usage"] is not None:
                    st.session_state.last_stream_usage = chunk["usage"]
                    logging.info(f"Captured stream usage: {chunk['usage']}")

                delta = chunk["choices"][0]["delta"].get("content")
                if delta is not None: yield delta, None
    except ValueError as ve: # From api_post if key is invalid
        logging.error(f"ValueError during streamed call setup: {ve}"); yield None, str(ve)
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code; text = e.response.text
        logging.error(f"Stream HTTPError {status_code}: {text}")
        yield None, f"HTTP {status_code}: An error occurred with the API provider. Details: {text}"
    except Exception as e:
        logging.error(f"Streamed API call failed: {e}"); yield None, f"Failed to connect or make request: {e}"

# ------------------------- Model Routing -----------------------
# The ROUTER_MODEL_GUIDANCE dictionary is now defined in the Configuration section with anonymized descriptions

def route_choice(user_msg: str, allowed: list[str], chat_history: list) -> str:
    # Determine a fallback choice from allowed models, or F, or any model, or true fallback
    if "F" in allowed: fallback_choice_letter = "F"
    elif allowed: fallback_choice_letter = allowed[0]
    elif "F" in MODEL_MAP: fallback_choice_letter = "F" # F might be available but not in `allowed` due to quota
    elif MODEL_MAP: fallback_choice_letter = list(MODEL_MAP.keys())[0]
    else: # No models in MODEL_MAP, must use true fallback
        logging.error("Router: No models available in MODEL_MAP for fallback. Using FALLBACK_MODEL_KEY.")
        return FALLBACK_MODEL_KEY

    if not allowed: # No models have quota
        logging.warning(f"route_choice called with empty allowed list. Defaulting to FALLBACK_MODEL_KEY.")
        return FALLBACK_MODEL_KEY # True fallback if no allowed models
    if len(allowed) == 1:
        logging.info(f"Router: Only one model allowed ('{allowed[0]}'), selecting it directly.")
        return allowed[0]

    # Construct history context string (already anonymizes specific model used for past assistant responses)
    history_segments = []
    current_chars = 0
    relevant_history_for_router = chat_history[:-1] # Exclude current user_msg
    for msg in reversed(relevant_history_for_router):
        role = msg.get("role", "assistant").capitalize() # "User" or "Assistant"
        content = msg.get("content", "")
        segment = f"{role}: {content}\n" # Specific model choice (e.g., "A", "F") is NOT in this segment
        if current_chars + len(segment) > MAX_HISTORY_CHARS_FOR_ROUTER: break
        history_segments.append(segment)
        current_chars += len(segment)
    history_context_str = "".join(reversed(history_segments)).strip()
    if not history_context_str: history_context_str = "No prior conversation history for this session."

    system_prompt_parts = [
        "You are an expert AI model routing assistant. Your task is to select the *single most appropriate and cost-effective* model letter from the 'Available Models' list to handle the 'Latest User Query' provided at the end. Consider the 'Recent Conversation History' for context.",
        "Strictly adhere to these decision-making principles in order of importance:",
        "1. Maximize Cost-Effectiveness: This is your PRIMARY GOAL. Always prefer a cheaper model (F > D > B > C > A in general cost order) if it can adequately perform the task. Do NOT select expensive models (A, C) unless explicitly justified by the query's extreme complexity and specific requirements that cheaper models demonstrably cannot meet.",
        "2. Analyze Latest User Query Intent (in context of history): Deeply understand what the user is trying to achieve with their *latest query*, the complexity involved (simple, moderate, high, extreme), the desired output style (factual, creative, conversational), and any implicit needs, considering the flow of the conversation so far.",
        "3. Match to Model Strengths and Weaknesses as described below."
    ]
    system_prompt_parts.append("\nAvailable Models (select one letter):")
    # Uses the anonymized ROUTER_MODEL_GUIDANCE from the top of the file
    for k_model_key in allowed:
        description = ROUTER_MODEL_GUIDANCE.get(k_model_key, f"(Model {k_model_key} - General purpose description; details not found).")
        system_prompt_parts.append(f"- {k_model_key}: {description}")

    system_prompt_parts.append("\nSpecific Selection Guidance (apply rigorously to the 'Latest User Query'):")
    if "F" in allowed: system_prompt_parts.append("  - If 'F' is available AND the 'Latest User Query' is simple (e.g., basic factual question, quick definition, short summary of <200 words, simple classification), CHOOSE 'F'.")
    if "D" in allowed: system_prompt_parts.append("  - If 'D' is available AND the 'Latest User Query' is primarily factual, technical, code-related, or requires straightforward logical deduction, AND 'F' (if available) is too basic, STRONGLY PREFER 'D'.")
    if "B" in allowed: system_prompt_parts.append("  - If 'B' is available, AND 'F'/'D' (if available) are insufficient for the 'Latest User Query's' general reasoning, drafting, or moderate creative needs, 'B' is a good general-purpose choice. Prefer 'B' over 'A'/'C' if peak quality/style isn't explicitly demanded.")
    system_prompt_parts.append("\n  - Guidance for Expensive Models (A, C) - Use Sparingly for 'Latest User Query':")
    if "C" in allowed: system_prompt_parts.append("    - CHOOSE 'C' ONLY if the 'Latest User Query' *explicitly requires or strongly implies a need for* a highly polished, empathetic, human-like conversational tone, or involves nuanced, multi-turn creative collaboration where its specific stylistic strengths are indispensable AND 'B' (if available) is clearly inadequate.")
    if "A" in allowed: system_prompt_parts.append("    - CHOOSE 'A' ONLY if the 'Latest User Query' involves *exceptionally* complex, multi-layered reasoning, requires generation of extensive, high-stakes creative content, or tasks demanding the absolute frontier of AI capability that *no other available model can credibly handle*.")

    system_prompt_parts.append("\nRecent Conversation History (context for the 'Latest User Query'):")
    system_prompt_parts.append(history_context_str)
    system_prompt_parts.append("\nINSTRUCTIONS: Based on all the above guidance and the provided conversation history, analyze the 'Latest User Query' (which will be the next message from the 'user' role). Then, respond with ONLY the single capital letter of your chosen model (e.g., A, B, C, D, or F). NO EXPLANATION, NO EXTRA TEXT, JUST THE LETTER.")
    final_system_message = "\n".join(system_prompt_parts)

    router_messages = [{"role": "system", "content": final_system_message}, {"role": "user", "content": user_msg}]
    payload_r = {"model": ROUTER_MODEL_ID, "messages": router_messages, "max_tokens": 10, "temperature": 0.1}

    try:
        r = api_post(payload_r)
        choice_data = r.json()
        raw_text_response = choice_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()
        logging.info(f"Router raw response: '{raw_text_response}' for query: '{user_msg}' with history context.")
        
        chosen_model_letter = None
        for char_in_response in raw_text_response:
            if char_in_response in allowed:
                chosen_model_letter = char_in_response; break
        if chosen_model_letter:
            logging.info(f"Router selected model: '{chosen_model_letter}'")
            return chosen_model_letter
        else: # Router's response didn't contain an allowed letter
            logging.warning(f"Router returned ('{raw_text_response}') - no allowed letter found. Fallback to '{fallback_choice_letter}'.")
            return fallback_choice_letter # Use the determined fallback letter
    except ValueError as ve: logging.error(f"ValueError in router call: {ve}") # API key issue
    except requests.exceptions.HTTPError as e: logging.error(f"Router HTTPError {e.response.status_code}: {e.response.text}")
    except (KeyError, IndexError, AttributeError, json.JSONDecodeError) as je:
        response_text_for_log = r.text if 'r' in locals() and hasattr(r, 'text') else "N/A"
        logging.error(f"Router JSON/structure error: {je}. Raw: {response_text_for_log}")
    except Exception as e: logging.error(f"Router unexpected error: {e}")

    logging.warning(f"Router failed. Fallback to model letter: {fallback_choice_letter}")
    return fallback_choice_letter # Use the determined fallback letter

# --------------------- Credits Endpoint -----------------------
def get_credits():
    active_api_key = st.session_state.get("openrouter_api_key")
    if not is_api_key_valid(active_api_key):
        logging.warning("get_credits: API Key is not syntactically valid or not set."); return None, None, None
    try:
        r = requests.get(f"{OPENROUTER_API_BASE}/credits", headers={"Authorization": f"Bearer {active_api_key}"}, timeout=10)
        r.raise_for_status(); d = r.json()["data"]
        st.session_state.api_key_auth_failed = False
        return d["total_credits"], d["total_usage"], d["total_credits"] - d["total_usage"]
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code; err_text = e.response.text
        if status_code == 401:
            st.session_state.api_key_auth_failed = True
            logging.warning(f"Could not fetch /credits: HTTP {status_code} Unauthorized. {err_text}")
        else: logging.warning(f"Could not fetch /credits: HTTP {status_code}. {err_text}")
        return None, None, None
    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
        logging.warning(f"Could not fetch /credits due to network/parsing error: {e}"); return None, None, None

# ------------------------- UI Styling --------------------------
def load_custom_css():
    css = """
    <style>
        :root { --border-radius-sm: 4px; --border-radius-md: 8px; --border-radius-lg: 12px; --spacing-sm: 0.5rem; --spacing-md: 1rem; --spacing-lg: 1.5rem; --shadow-light: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1); --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1); } body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji"; }
        [data-testid="stSidebar"] { background-color: var(--secondaryBackgroundColor); padding: var(--spacing-lg) var(--spacing-md); border-right: 1px solid var(--divider-color, #262730); } [data-testid="stSidebar"] .stImage > img { border-radius: 50%; box-shadow: var(--shadow-light); width: 48px !important; height: 48px !important; margin-right: var(--spacing-sm); } [data-testid="stSidebar"] h1 { font-size: 1.5rem !important; color: var(--primaryColor); font-weight: 600; margin-bottom: 0; padding-top: 0.2rem; }
        [data-testid="stSidebar"] .stButton > button { border-radius: var(--border-radius-md); border: 1px solid var(--divider-color, #333); padding: 0.6em 1em; font-size: 0.9em; background-color: transparent; color: var(--textColor); transition: background-color 0.2s, border-color 0.2s; width: 100%; margin-bottom: var(--spacing-sm); text-align: left; font-weight: 500; } [data-testid="stSidebar"] .stButton > button:hover:not(:disabled) { border-color: var(--primaryColor); background-color: color-mix(in srgb, var(--primaryColor) 15%, transparent); }
        [data-testid="stSidebar"] .stButton > button:disabled { opacity: 1.0; cursor: default; background-color: color-mix(in srgb, var(--primaryColor) 25%, transparent) !important; border-left: 3px solid var(--primaryColor) !important; border-top-color: var(--divider-color, #333) !important; border-right-color: var(--divider-color, #333) !important; border-bottom-color: var(--divider-color, #333) !important; font-weight: 600; color: var(--textColor); }
        [data-testid="stSidebar"] [data-testid="stButton-new_chat_button_top"] > button { background-color: var(--primaryColor); color: white; border-color: var(--primaryColor); font-weight: 600; } [data-testid="stSidebar"] [data-testid="stButton-new_chat_button_top"] > button:hover { background-color: color-mix(in srgb, var(--primaryColor) 85%, black); border-color: color-mix(in srgb, var(--primaryColor) 85%, black); } [data-testid="stSidebar"] [data-testid="stButton-new_chat_button_top"] > button:disabled { background-color: var(--primaryColor) !important; color: white !important; border-color: var(--primaryColor) !important; opacity: 0.6 !important; cursor: not-allowed !important; border-left: 1px solid var(--primaryColor) !important; }
        [data-testid="stSidebar"] h3, [data-testid="stSidebar"] .stSubheader { font-size: 0.8rem !important; text-transform: uppercase; font-weight: 700; color: var(--text-color-secondary, #A0A0A0); margin-top: var(--spacing-lg); margin-bottom: var(--spacing-sm); letter-spacing: 0.05em; }
        [data-testid="stSidebar"] [data-testid="stExpander"] { border: 1px solid var(--divider-color, #262730); border-radius: var(--border-radius-md); background-color: transparent; margin-bottom: var(--spacing-md); } [data-testid="stSidebar"] [data-testid="stExpander"] summary { padding: 0.6rem var(--spacing-md) !important; font-size: 0.85rem !important; font-weight: 600 !important; text-transform: uppercase; color: var(--textColor) !important; border-bottom: 1px solid var(--divider-color, #262730); border-top-left-radius: var(--border-radius-md); border-top-right-radius: var(--border-radius-md); } [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover { background-color: color-mix(in srgb, var(--textColor) 5%, transparent); }
        [data-testid="stSidebar"] [data-testid="stExpander"] div[data-testid="stExpanderDetails"] { padding: var(--spacing-sm) var(--spacing-md) !important; background-color: var(--secondaryBackgroundColor); border-bottom-left-radius: var(--border-radius-md); border-bottom-right-radius: var(--border-radius-md); } [data-testid="stSidebar"] [data-testid="stExpander"][aria-label^="⚡ DAILY MODEL QUOTAS"] div[data-testid="stExpanderDetails"] { padding-top: 0.6rem !important; padding-bottom: 0.2rem !important; padding-left: 0.1rem !important; padding-right: 0.1rem !important; } [data-testid="stSidebar"] [data-testid="stExpander"][aria-label^="⚡ DAILY MODEL QUOTAS"] div[data-testid="stHorizontalBlock"] { gap: 0.25rem !important; }
        .compact-quota-item { display: flex; flex-direction: column; align-items: center; text-align: center; padding: 0px 4px; } .cq-info { font-size: 0.7rem; margin-bottom: 3px; line-height: 1.1; white-space: nowrap; color: var(--textColor); } .cq-bar-track { width: 100%; height: 8px; background-color: color-mix(in srgb, var(--textColor) 10%, transparent); border: 1px solid var(--divider-color, #333); border-radius: var(--border-radius-sm); overflow: hidden; margin-bottom: 5px; } .cq-bar-fill { height: 100%; border-radius: var(--border-radius-sm); transition: width 0.3s ease-in-out, background-color 0.3s ease-in-out; } .cq-value { font-size: 0.7rem; font-weight: bold; line-height: 1; }
        .settings-panel { border: 1px solid var(--divider-color, #333); border-radius: var(--border-radius-md); padding: var(--spacing-md); margin-top: var(--spacing-sm); margin-bottom: var(--spacing-md); background-color: color-mix(in srgb, var(--backgroundColor) 50%, var(--secondaryBackgroundColor)); } .settings-panel .stTextInput input { border-color: var(--divider-color, #444) !important; }
        [data-testid="stChatInputContainer"] { background-color: var(--secondaryBackgroundColor); border-top: 1px solid var(--divider-color, #262730); padding: var(--spacing-sm) var(--spacing-md); } [data-testid="stChatInput"] textarea { border-color: var(--divider-color, #444) !important; border-radius: var(--border-radius-md) !important; background-color: var(--backgroundColor) !important; color: var(--textColor) !important; }
        [data-testid="stChatMessage"] { border-radius: var(--border-radius-lg); padding: var(--spacing-md) 1.25rem; margin-bottom: var(--spacing-md); box-shadow: var(--shadow-light); border: 1px solid transparent; max-width: 85%; } [data-testid="stChatMessage"][data-testid^="stChatMessageUser"] { background-color: var(--primaryColor); color: white; margin-left: auto; border-top-right-radius: var(--border-radius-sm); } [data-testid="stChatMessage"][data-testid^="stChatMessageAssistant"] { background-color: var(--secondaryBackgroundColor); color: var(--textColor); margin-right: auto; border-top-left-radius: var(--border-radius-sm); }
        hr { margin-top: var(--spacing-md); margin-bottom: var(--spacing-md); border: 0; border-top: 1px solid var(--divider-color, #262730); }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ----------------- API Key State Initialization -------------------
if "openrouter_api_key" not in st.session_state:
    app_conf = _load_app_config()
    st.session_state.openrouter_api_key = app_conf.get("openrouter_api_key", None)
if "api_key_auth_failed" not in st.session_state: st.session_state.api_key_auth_failed = False
api_key_is_syntactically_valid = is_api_key_valid(st.session_state.get("openrouter_api_key"))
app_requires_api_key_setup = not api_key_is_syntactically_valid or st.session_state.api_key_auth_failed

# -------------------- Main Application Rendering -------------------
if app_requires_api_key_setup:
    st.set_page_config(page_title="OpenRouter API Key Setup", layout="centered")
    load_custom_css()
    st.title("🔒 OpenRouter API Key Required"); st.markdown("---")
    if st.session_state.api_key_auth_failed: st.error("API Key Authentication Failed. Please verify your key on OpenRouter.ai and re-enter.")
    elif not api_key_is_syntactically_valid and st.session_state.get("openrouter_api_key") is not None: st.error("The previously configured API Key has an invalid format. It must start with `sk-or-`.")
    else: st.info("Please configure your OpenRouter API Key to use the application.")
    st.markdown( "You can get a key from [OpenRouter.ai Keys](https://openrouter.ai/keys). Enter it below to continue." )
    new_key_input_val = st.text_input("Enter OpenRouter API Key", type="password", key="api_key_setup_input", value="", placeholder="sk-or-...")
    if st.button("Save and Validate API Key", key="save_api_key_setup_button", use_container_width=True, type="primary"):
        if is_api_key_valid(new_key_input_val):
            st.session_state.openrouter_api_key = new_key_input_val
            _save_app_config(new_key_input_val); st.session_state.api_key_auth_failed = False
            with st.spinner("Validating API Key..."): fetched_credits_data = get_credits()
            if st.session_state.api_key_auth_failed: st.error("Authentication failed with the provided API Key."); time.sleep(0.5); st.rerun()
            elif fetched_credits_data == (None, None, None): st.error("Could not validate API Key. Network or API provider issue.")
            else:
                st.success("API Key saved and validated! Initializing application...")
                if "credits" not in st.session_state: st.session_state.credits = {}
                st.session_state.credits["total"], st.session_state.credits["used"], st.session_state.credits["remaining"] = fetched_credits_data
                st.session_state.credits_ts = time.time(); time.sleep(1.0); st.rerun()
        elif not new_key_input_val: st.warning("API Key field cannot be empty.")
        else: st.error("Invalid API key format. It must start with 'sk-or-'.")
    st.markdown("---"); st.caption("Your API key is stored locally in `app_config.json`.")
else:
    st.set_page_config(page_title="OpenRouter Chat", layout="wide", initial_sidebar_state="expanded")
    load_custom_css()
    if "settings_panel_open" not in st.session_state: st.session_state.settings_panel_open = False
    needs_save_session = False
    if "sid" not in st.session_state: st.session_state.sid = _new_sid(); needs_save_session = True
    elif st.session_state.sid not in sessions:
        logging.warning(f"Session ID {st.session_state.sid} not found. Creating new chat."); st.session_state.sid = _new_sid(); needs_save_session = True
    if _delete_unused_blank_sessions(keep_sid=st.session_state.sid): needs_save_session = True
    if needs_save_session: _save(SESS_FILE, sessions); st.rerun()

    if "credits" not in st.session_state: st.session_state.credits = {"total": 0.0, "used": 0.0, "remaining": 0.0}; st.session_state.credits_ts = 0
    credits_are_stale = time.time() - st.session_state.get("credits_ts", 0) > 3600
    credits_are_default_and_old = (st.session_state.credits.get("total") == 0.0 and st.session_state.credits.get("used") == 0.0 and st.session_state.credits.get("remaining") == 0.0 and st.session_state.credits_ts != 0 and time.time() - st.session_state.credits_ts > 300)
    credits_never_fetched = st.session_state.credits_ts == 0
    if credits_are_stale or credits_are_default_and_old or credits_never_fetched:
        logging.info("Refreshing credits (stale, default, or never fetched).")
        credits_data = get_credits()
        if st.session_state.get("api_key_auth_failed"): logging.error("API Key auth failed during credit refresh.")
        if credits_data != (None, None, None):
            st.session_state.credits["total"], st.session_state.credits["used"], st.session_state.credits["remaining"] = credits_data
            st.session_state.credits_ts = time.time()
        else:
            st.session_state.credits_ts = time.time()
            if not all(isinstance(st.session_state.credits.get(k), (int,float)) for k in ["total", "used", "remaining"]):
                 st.session_state.credits = {"total": 0.0, "used": 0.0, "remaining": 0.0}

    with st.sidebar:
        settings_button_label = "⚙️ Close Settings" if st.session_state.settings_panel_open else "⚙️ Settings"
        if st.button(settings_button_label, key="toggle_settings_button_sidebar", use_container_width=True):
            st.session_state.settings_panel_open = not st.session_state.settings_panel_open; st.rerun()
        if st.session_state.get("settings_panel_open"):
            st.markdown("<div class='settings-panel'>", unsafe_allow_html=True); st.subheader("API Key Configuration")
            current_api_key_in_panel = st.session_state.get("openrouter_api_key")
            if current_api_key_in_panel and len(current_api_key_in_panel) > 8: key_display = f"Current key: `sk-or-...{current_api_key_in_panel[-4:]}`"
            elif current_api_key_in_panel: key_display = "Current key: `sk-or-...` (short key)"
            else: key_display = "Current key: Not set"
            st.caption(key_display)
            new_key_input_sidebar = st.text_input("New OpenRouter API Key (optional)", type="password", key="api_key_sidebar_input", placeholder="sk-or-...")
            if st.button("Save New API Key", key="save_api_key_sidebar_button", use_container_width=True):
                if is_api_key_valid(new_key_input_sidebar):
                    st.session_state.openrouter_api_key = new_key_input_sidebar
                    _save_app_config(new_key_input_sidebar); st.session_state.api_key_auth_failed = False
                    with st.spinner("Validating new API key..."): credits_data = get_credits()
                    if st.session_state.api_key_auth_failed: st.error("New API Key failed authentication.")
                    elif credits_data == (None,None,None): st.warning("Could not validate new API key. Saved, but functionality may be affected.")
                    else:
                        st.success("New API Key saved and validated!")
                        st.session_state.credits["total"],st.session_state.credits["used"],st.session_state.credits["remaining"] = credits_data
                        st.session_state.credits_ts = time.time()
                    st.session_state.settings_panel_open = False; time.sleep(0.8); st.rerun()
                elif not new_key_input_sidebar: st.warning("API Key field empty. No changes.")
                else: st.error("Invalid API key format. Must start with 'sk-or-'.")
            st.markdown("</div>", unsafe_allow_html=True)
        st.divider()
        logo_title_cols = st.columns([1, 4], gap="small")
        with logo_title_cols[0]: st.image("https://avatars.githubusercontent.com/u/130328222?s=200&v=4", width=48)
        with logo_title_cols[1]: st.title("OpenRouter Chat")
        st.divider()
        with st.expander("⚡ DAILY MODEL QUOTAS", expanded=True):
            active_model_keys_for_display = sorted(MODEL_MAP.keys())
            if not active_model_keys_for_display: st.caption("No models configured for quota tracking.")
            else:
                _ensure_quota_data_is_current() # Make sure global quota data is fresh
                quota_cols = st.columns(len(active_model_keys_for_display))
                for i, m_key in enumerate(active_model_keys_for_display):
                    with quota_cols[i]:
                        # Display remaining daily messages
                        left_d_msgs = get_remaining_daily_messages(m_key)
                        lim_d_msgs = NEW_PLAN_CONFIG.get(m_key, (0,))[0]

                        is_unlimited = lim_d_msgs >= 900_000 # Arbitrary large number for "unlimited"
                        if is_unlimited:
                            pct_float, fill_width_val, left_display = 1.0, 100, "∞"
                        elif lim_d_msgs > 0:
                            pct_float = max(0.0, min(1.0, left_d_msgs / lim_d_msgs))
                            fill_width_val = int(pct_float * 100)
                            left_display = str(left_d_msgs)
                        else: # limit is 0 or not defined
                            pct_float, fill_width_val, left_display = 0.0, 0, "0"
                        
                        bar_color = "#f44336" # Red
                        if pct_float > 0.5: bar_color = "#4caf50" # Green
                        elif pct_float > 0.25: bar_color = "#ffc107" # Yellow
                        if is_unlimited: bar_color = "var(--primaryColor)"
                        
                        emoji_char = EMOJI.get(m_key, "❔")
                        st.markdown(f"""<div class="compact-quota-item"><div class="cq-info">{emoji_char} <b>{m_key}</b></div><div class="cq-bar-track"><div class="cq-bar-fill" style="width: {fill_width_val}%; background-color: {bar_color};"></div></div><div class="cq-value" style="color: {bar_color};">{left_display}</div></div>""", unsafe_allow_html=True)
        st.divider()
        current_session_is_truly_blank = (st.session_state.sid in sessions and sessions[st.session_state.sid].get("title") == "New chat" and not sessions[st.session_state.sid].get("messages"))
        if st.button("➕ New chat", key="new_chat_button_top", use_container_width=True, disabled=current_session_is_truly_blank):
            st.session_state.sid = _new_sid(); _delete_unused_blank_sessions(keep_sid=st.session_state.sid)
            _save(SESS_FILE, sessions); st.rerun()
        st.subheader("Chats")
        valid_sids = [s for s in sessions.keys() if isinstance(s, str) and s.isdigit()]
        sorted_sids = sorted(valid_sids, key=lambda s: int(s), reverse=True)
        for sid_key in sorted_sids:
            if sid_key not in sessions: continue
            title = sessions[sid_key].get("title", "Untitled")
            display_title = title[:25] + ("…" if len(title) > 25 else "")
            is_active_chat = st.session_state.sid == sid_key
            if st.button(display_title, key=f"session_button_{sid_key}", use_container_width=True, disabled=is_active_chat):
                if not is_active_chat:
                    _delete_unused_blank_sessions(keep_sid=sid_key); st.session_state.sid = sid_key
                    _save(SESS_FILE, sessions); st.rerun()
        st.divider()
        st.subheader("Model-Routing Map")
        st.caption(f"Router: {ROUTER_MODEL_ID}")
        with st.expander("Letters → Models", expanded=False): # Uses user-facing MODEL_DESCRIPTIONS
            for k_model in sorted(MODEL_MAP.keys()):
                desc = MODEL_DESCRIPTIONS.get(k_model, MODEL_MAP.get(k_model, "N/A"))
                max_tok = MAX_TOKENS.get(k_model, 0)
                st.markdown(f"**{k_model}**: {desc} (max_out={max_tok:,})")
        st.divider()
        with st.expander("Account stats (credits)", expanded=False):
            if st.button("Refresh Credits", key="refresh_credits_button_sidebar"):
                 with st.spinner("Refreshing credits..."): credits_data = get_credits()
                 if not st.session_state.get("api_key_auth_failed"):
                    if credits_data != (None,None,None):
                        st.session_state.credits["total"],st.session_state.credits["used"],st.session_state.credits["remaining"] = credits_data
                        st.session_state.credits_ts = time.time(); st.success("Credits refreshed!")
                    else: st.warning("Could not refresh credits.")
                 else: st.error("API Key authentication failed. Cannot refresh credits.")
                 st.rerun()
            tot, used, rem = st.session_state.credits.get("total"), st.session_state.credits.get("used"), st.session_state.credits.get("remaining")
            if tot is None or used is None or rem is None : st.warning("Could not fetch/display credits.")
            else: st.markdown(f"**Remaining:** ${float(rem):.2f} cr"); st.markdown(f"**Used:** ${float(used):.2f} cr")
            ts = st.session_state.get("credits_ts", 0)
            last_updated_str = datetime.fromtimestamp(ts, TZ).strftime('%-d %b, %H:%M:%S') if ts else "N/A"
            st.caption(f"Last updated: {last_updated_str}")

    if st.session_state.sid not in sessions:
        logging.error(f"CRITICAL: SID {st.session_state.sid} missing. Resetting."); st.session_state.sid = _new_sid()
        _save(SESS_FILE, sessions); st.rerun(); st.stop()
    current_sid = st.session_state.sid
    chat_history = sessions[current_sid]["messages"]
    for msg in chat_history:
        role = msg.get("role", "assistant"); avatar_char = "👤" if role == "user" else None
        if role == "assistant":
            m_key = msg.get("model")
            if m_key == FALLBACK_MODEL_KEY: avatar_char = FALLBACK_MODEL_EMOJI
            elif m_key in EMOJI: avatar_char = EMOJI[m_key]
            else: avatar_char = "🤖"
        with st.chat_message(role, avatar=avatar_char): st.markdown(msg.get("content", "*empty*"))

    if prompt := st.chat_input("Ask anything…", key=f"chat_input_{current_sid}"):
        chat_history.append({"role":"user","content":prompt})
        with st.chat_message("user", avatar="👤"): st.markdown(prompt)
        if not is_api_key_valid(st.session_state.get("openrouter_api_key")) or st.session_state.get("api_key_auth_failed"):
            st.error("API Key not configured or failed. Set in ⚙️ Settings.")
            if st.session_state.get("api_key_auth_failed"): time.sleep(0.5); st.rerun()
        else:
            _ensure_quota_data_is_current() # Refresh quota data before checking
            allowed_standard_models = [k for k in MODEL_MAP if is_model_available(k)] # Use new availability check

            use_fallback, chosen_model_key, model_id_to_use, max_tokens_api, avatar_resp = (False, None, None, None, "🤖")

            if not allowed_standard_models:
                logging.info(f"All standard model quotas (messages, tokens, or 3hr cap) exhausted. Using fallback: {FALLBACK_MODEL_ID}")
                st.info(f"{FALLBACK_MODEL_EMOJI} All model quotas exhausted. Using free fallback.")
                use_fallback, chosen_model_key, model_id_to_use, max_tokens_api, avatar_resp = (True, FALLBACK_MODEL_KEY, FALLBACK_MODEL_ID, FALLBACK_MODEL_MAX_TOKENS, FALLBACK_MODEL_EMOJI)
            else:
                routed_key_letter = route_choice(prompt, allowed_standard_models, chat_history)
                if st.session_state.get("api_key_auth_failed"): st.error("API Auth failed during model routing. Check Key in Settings.")
                elif routed_key_letter == FALLBACK_MODEL_KEY: # Router explicitly chose true fallback
                    logging.warning(f"Router chose FALLBACK_MODEL_KEY. Using free fallback: {FALLBACK_MODEL_ID}.")
                    st.warning(f"{FALLBACK_MODEL_EMOJI} Router determined no standard models suitable. Using free fallback.")
                    use_fallback, chosen_model_key, model_id_to_use, max_tokens_api, avatar_resp = (True, FALLBACK_MODEL_KEY, FALLBACK_MODEL_ID, FALLBACK_MODEL_MAX_TOKENS, FALLBACK_MODEL_EMOJI)
                elif routed_key_letter not in MODEL_MAP or not is_model_available(routed_key_letter): # Router chose a letter, but it's invalid or now unavailable
                    logging.warning(f"Router chose '{routed_key_letter}' (invalid or no quota/unavailable after check). Using free fallback {FALLBACK_MODEL_ID}.")
                    st.warning(f"{FALLBACK_MODEL_EMOJI} Model routing issue or chosen model '{routed_key_letter}' unavailable. Using free fallback.")
                    use_fallback, chosen_model_key, model_id_to_use, max_tokens_api, avatar_resp = (True, FALLBACK_MODEL_KEY, FALLBACK_MODEL_ID, FALLBACK_MODEL_MAX_TOKENS, FALLBACK_MODEL_EMOJI)
                else: # Valid standard model letter chosen and still available
                    chosen_model_key = routed_key_letter
                    model_id_to_use = MODEL_MAP[chosen_model_key]
                    max_tokens_api = MAX_TOKENS[chosen_model_key]
                    avatar_resp = EMOJI.get(chosen_model_key, "🤖")
            
            if not model_id_to_use and not st.session_state.get("api_key_auth_failed"):
                 logging.error("No model_id_to_use, but API key auth not flagged. Using fallback.")
                 st.warning(f"{FALLBACK_MODEL_EMOJI} Unexpected issue selecting model. Using free fallback.")
                 use_fallback, chosen_model_key, model_id_to_use, max_tokens_api, avatar_resp = (True, FALLBACK_MODEL_KEY, FALLBACK_MODEL_ID, FALLBACK_MODEL_MAX_TOKENS, FALLBACK_MODEL_EMOJI)

            if model_id_to_use:
                with st.chat_message("assistant", avatar=avatar_resp):
                    response_placeholder, full_response = st.empty(), ""
                    api_call_ok = True
                    for chunk_content, err_msg in streamed(model_id_to_use, chat_history, max_tokens_api):
                        if st.session_state.get("api_key_auth_failed"):
                            full_response = "❗ **API Authentication Error**: Update Key in ⚙️ Settings."
                            api_call_ok = False; break
                        if err_msg: full_response = f"❗ **API Error**: {err_msg}"; api_call_ok = False; break
                        if chunk_content: full_response += chunk_content; response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)
                
                # Retrieve token usage from session state
                last_usage = st.session_state.pop("last_stream_usage", None)
                prompt_tokens_used = 0
                completion_tokens_used = 0
                if last_usage:
                    prompt_tokens_used = last_usage.get("prompt_tokens", 0)
                    completion_tokens_used = last_usage.get("completion_tokens", 0)
                    logging.info(f"API call completed. Tokens used: Prompt={prompt_tokens_used}, Completion={completion_tokens_used}")
                else:
                    logging.warning(f"Token usage information not found in session state after stream for model {model_id_to_use}.")

                chat_history.append({
                    "role": "assistant",
                    "content": full_response,
                    "model": chosen_model_key if api_call_ok else FALLBACK_MODEL_KEY,
                    "prompt_tokens": prompt_tokens_used if api_call_ok else 0,
                    "completion_tokens": completion_tokens_used if api_call_ok else 0
                })

                if api_call_ok:
                    if not use_fallback and chosen_model_key and chosen_model_key in MODEL_MAP:
                       record_use(chosen_model_key, prompt_tokens_used, completion_tokens_used)
                    if sessions[current_sid]["title"] == "New chat" and prompt:
                       sessions[current_sid]["title"] = _autoname(prompt)
                       _delete_unused_blank_sessions(keep_sid=current_sid) # This might save sessions
                _save(SESS_FILE, sessions); st.rerun() # Save sessions after append and potential record_use
            elif st.session_state.get("api_key_auth_failed"): time.sleep(0.5); st.rerun()
            else: st.error("Unexpected error: Could not determine a model."); logging.error("Reached unexpected state: no model_id and no API auth failure.")
