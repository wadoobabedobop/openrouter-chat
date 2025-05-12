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

ROUTER_MODEL_GUIDANCE = {
    "A": "(Model A: Top-Tier Quality & Capability) Use for EXTREMELY complex, multi-step reasoning; highly advanced creative generation (e.g., novel excerpts, sophisticated poetry); tasks demanding cutting-edge knowledge and deep nuanced understanding. HIGHEST COST. CHOOSE ONLY if query explicitly demands top-tier, 'genius-level' output AND cheaper models are CLEARLY insufficient. Avoid for anything less.",
    "B": "(Model B: Solid Mid-Tier All-Rounder) Use for general purpose chat; moderate complexity reasoning; summarization; drafting emails/content; brainstorming; standard instruction following. Good balance of capability and cost. MODERATE COST. CHOOSE if 'F' or 'D' are too basic, AND 'A' or 'C' are overkill/not strictly necessary for the task's core requirements.",
    "C": "(Model C: High Quality, Polished & Empathetic) Use for tasks requiring highly polished, empathetic, or very human-like conversational interactions; complex multi-turn instruction adherence where its specific stylistic strengths are key; creative content generation with a defined sophisticated tone. HIGHER COST. CHOOSE ONLY if query *specifically* benefits from its unique interaction style or demands exceptional refinement AND 'B' (if available) is clearly inadequate.",
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
            original_len = len(_g_quota_data[MODEL_A_3H_CALLS_KEY])
            _g_quota_data[MODEL_A_3H_CALLS_KEY] = [
                ts for ts in _g_quota_data[MODEL_A_3H_CALLS_KEY]
                if current_time - ts < three_hr_window_seconds
            ]
            if len(_g_quota_data[MODEL_A_3H_CALLS_KEY]) != original_len:
                logging.info(f"Pruned Model A 3-hour call timestamps. Original: {original_len}, New: {len(_g_quota_data[MODEL_A_3H_CALLS_KEY])}.")
                _save(QUOTA_FILE, _g_quota_data) # Save if pruned
        return _g_quota_data

    q_loaded_data = _load(QUOTA_FILE, {})
    data_was_modified = _g_quota_data is None

    active_model_keys = set(MODEL_MAP.keys())
    cleaned_during_load = False

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

    # Remove obsolete fields (weekly)
    obsolete_keys = ["w", "w_u"]
    for key in obsolete_keys:
        if key in q_loaded_data:
            del q_loaded_data[key]
            data_was_modified = True
            logging.info(f"Removed obsolete key '{key}' from quota data.")


    current_model_zeros = {k: 0 for k in MODEL_MAP.keys()}
    reset_occurred_d = _reset(q_loaded_data, "d", now_d_stamp, current_model_zeros)
    reset_occurred_m = _reset(q_loaded_data, "m", now_m_stamp, current_model_zeros)
    if reset_occurred_d or reset_occurred_m: data_was_modified = True

    # Initialize or prune 3-hour calls for Model A
    if MODEL_A_3H_CALLS_KEY not in q_loaded_data:
        q_loaded_data[MODEL_A_3H_CALLS_KEY] = []
        data_was_modified = True # Initializing counts as modification
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

    if model_key == "A" and plan[6] > 0:
        # Check count within the current 3-hour window
        current_time = time.time()
        three_hr_window_seconds = plan[7]
        recent_calls = [
            ts for ts in current_q_data.get(MODEL_A_3H_CALLS_KEY, [])
            if current_time - ts < three_hr_window_seconds
        ]
        usage["used_3hr_msg"] = len(recent_calls)


    return {**usage, **limits}

def is_model_available(model_key: str) -> bool:
    if model_key not in NEW_PLAN_CONFIG:
        logging.warning(f"is_model_available: Model key '{model_key}' not in NEW_PLAN_CONFIG. Assuming unavailable.")
        return False

    stats = get_quota_usage_and_limits(model_key)
    if not stats: return False # Should not happen if key is in NEW_PLAN_CONFIG

    if stats["used_daily_msg"] >= stats["limit_daily_msg"]: return False
    if stats["used_monthly_msg"] >= stats["limit_monthly_msg"]: return False
    if stats["used_daily_in_tokens"] >= stats["limit_daily_in_tokens"]: return False
    if stats["used_monthly_in_tokens"] >= stats["limit_monthly_in_tokens"]: return False
    if stats["used_daily_out_tokens"] >= stats["limit_daily_out_tokens"]: return False
    if stats["used_monthly_out_tokens"] >= stats["limit_monthly_out_tokens"]: return False

    if model_key == "A" and stats["limit_3hr_msg"] != float('inf'):
        if stats["used_3hr_msg"] >= stats["limit_3hr_msg"]: return False

    return True

def get_remaining_daily_messages(model_key: str) -> int:
    if model_key not in NEW_PLAN_CONFIG: return 0
    stats = get_quota_usage_and_limits(model_key)
    if not stats: return 0
    return max(0, stats["limit_daily_msg"] - stats["used_daily_msg"])

def record_use(model_key: str, prompt_tokens: int, completion_tokens: int):
    if model_key not in MODEL_MAP:
        logging.warning(f"Attempted to record usage for non-standard or unknown model key: {model_key}")
        return

    current_q_data = _ensure_quota_data_is_current()

    # Ensure keys exist with default values if not present
    current_q_data.setdefault("d_u", {})
    current_q_data["d_u"].setdefault(model_key, 0)
    current_q_data.setdefault("m_u", {})
    current_q_data["m_u"].setdefault(model_key, 0)

    current_q_data.setdefault("d_it_u", {})
    current_q_data["d_it_u"].setdefault(model_key, 0)
    current_q_data.setdefault("m_it_u", {})
    current_q_data["m_it_u"].setdefault(model_key, 0)

    current_q_data.setdefault("d_ot_u", {})
    current_q_data["d_ot_u"].setdefault(model_key, 0)
    current_q_data.setdefault("m_ot_u", {})
    current_q_data["m_ot_u"].setdefault(model_key, 0)

    # Increment usage
    current_q_data["d_u"][model_key] += 1
    current_q_data["m_u"][model_key] += 1
    current_q_data["d_it_u"][model_key] += prompt_tokens
    current_q_data["m_it_u"][model_key] += prompt_tokens
    current_q_data["d_ot_u"][model_key] += completion_tokens
    current_q_data["m_ot_u"][model_key] += completion_tokens

    # Record timestamp for Model A 3-hour limit if applicable
    if model_key == "A" and NEW_PLAN_CONFIG["A"][6] > 0:
        current_q_data.setdefault(MODEL_A_3H_CALLS_KEY, []).append(time.time())

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
    sid = str(int(time.time() * 1000))
    sessions[sid] = {"title": "New chat", "messages": []}
    _delete_unused_blank_sessions(keep_sid=sid) # Clean up *other* blank sessions
    return sid


def _autoname(seed: str) -> str:
    words = seed.strip().split()
    cand = " ".join(words[:3]) or "Chat"
    return (cand[:25] + "…") if len(cand) > 25 else cand

# --------------------------- Logging ----------------------------
# Ensure basic logging is configured early
if not logging.getLogger().handlers:
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
    st.session_state.pop("last_stream_usage", None)

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

                if "usage" in chunk and chunk["usage"] is not None:
                    st.session_state.last_stream_usage = chunk["usage"]

                delta = chunk["choices"][0]["delta"].get("content")
                if delta is not None: yield delta, None
    except ValueError as ve:
        logging.error(f"ValueError during streamed call setup: {ve}"); yield None, str(ve)
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code; text = e.response.text
        logging.error(f"Stream HTTPError {status_code}: {text}")
        yield None, f"HTTP {status_code}: An error occurred with the API provider. Details: {text}"
    except Exception as e:
        logging.error(f"Streamed API call failed: {e}"); yield None, f"Failed to connect or make request: {e}"

# ------------------------- Model Routing -----------------------
def route_choice(user_msg: str, allowed: list[str], chat_history: list) -> str:
    if "F" in allowed: fallback_choice_letter = "F"
    elif allowed: fallback_choice_letter = allowed[0]
    elif "F" in MODEL_MAP: fallback_choice_letter = "F"
    elif MODEL_MAP: fallback_choice_letter = list(MODEL_MAP.keys())[0]
    else:
        logging.error("Router: No models available in MODEL_MAP for fallback. Using FALLBACK_MODEL_KEY.")
        return FALLBACK_MODEL_KEY

    if not allowed:
        logging.warning(f"route_choice called with empty allowed list. Defaulting to FALLBACK_MODEL_KEY.")
        return FALLBACK_MODEL_KEY
    if len(allowed) == 1:
        logging.info(f"Router: Only one model allowed ('{allowed[0]}'), selecting it directly.")
        return allowed[0]

    history_segments = []
    current_chars = 0
    relevant_history_for_router = chat_history[:-1]
    for msg in reversed(relevant_history_for_router):
        role = msg.get("role", "assistant").capitalize()
        content = msg.get("content", "")
        if not isinstance(content, str): content = str(content)
        segment = f"{role}: {content}\n"
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
        else:
            logging.warning(f"Router returned ('{raw_text_response}') - no allowed letter found. Fallback to '{fallback_choice_letter}'.")
            return fallback_choice_letter
    except ValueError as ve: logging.error(f"ValueError in router call: {ve}")
    except requests.exceptions.HTTPError as e: logging.error(f"Router HTTPError {e.response.status_code}: {e.response.text}")
    except (KeyError, IndexError, AttributeError, json.JSONDecodeError) as je:
        response_text_for_log = r.text if 'r' in locals() and hasattr(r, 'text') else "N/A"
        logging.error(f"Router JSON/structure error: {je}. Raw: {response_text_for_log}")
    except Exception as e: logging.error(f"Router unexpected error: {e}")

    logging.warning(f"Router failed. Fallback to model letter: {fallback_choice_letter}")
    return fallback_choice_letter

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
    css = f"""
    <style>
        :root {{
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
        }}

        body, .stApp {{
            font-family: var(--app-font) !important;
            background-color: var(--app-bg-color) !important;
            color: var(--app-text-color) !important;
        }}
        .main .block-container {{
            background-color: var(--app-bg-color);
            padding-top: var(--spacing-md);
            padding-bottom: var(--spacing-lg);
        }}

        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background-color: var(--app-secondary-bg-color);
            border-right: 1px solid var(--app-divider-color);
            padding: var(--spacing-md);
        }}
        [data-testid="stSidebar"] .stImage > img {{ /* Sidebar Logo */
            border-radius: var(--border-radius-md);
            box-shadow: var(--shadow-sm);
            width: 40px !important; height: 40px !important;
            margin-right: var(--spacing-sm);
        }}
        [data-testid="stSidebar"] h1 {{ /* App Title in Sidebar */
            font-size: 1.4rem !important;
            color: var(--app-text-color);
            font-weight: 600;
            margin-bottom: 0;
            line-height: 1.2;
            padding-top: 0.15rem; /* Align with logo better */
        }}
        .sidebar-title-container {{ /* Custom container for logo and title */
            display: flex;
            align-items: center;
            margin-bottom: var(--spacing-md);
        }}

        [data-testid="stSidebar"] .stButton > button {{
            border-radius: var(--border-radius-md);
            border: 1px solid var(--app-border-color);
            padding: 0.5em 0.8em; font-size: 0.9rem;
            background-color: var(--app-secondary-bg-color);
            color: var(--app-text-color);
            transition: background-color 0.2s, border-color 0.2s;
            width: 100%; margin-bottom: var(--spacing-sm);
            text-align: left; font-weight: 500;
        }}
        [data-testid="stSidebar"] .stButton > button:hover:not(:disabled) {{
            border-color: var(--app-primary-color);
            background-color: color-mix(in srgb, var(--app-primary-color) 8%, transparent);
        }}
        [data-testid="stSidebar"] .stButton > button:disabled {{ /* Active Chat Button */
            opacity: 1.0; cursor: default;
            background-color: color-mix(in srgb, var(--app-primary-color) 15%, transparent) !important;
            border-left: 3px solid var(--app-primary-color) !important;
            border-top-color: var(--app-border-color) !important;
            border-right-color: var(--app-border-color) !important;
            border-bottom-color: var(--app-border-color) !important;
            font-weight: 600; color: var(--app-text-color);
        }}
        [data-testid="stSidebar"] [data-testid*="new_chat_button_top"] > button {{
            background-color: var(--app-primary-color); color: white;
            border-color: var(--app-primary-color); font-weight: 600;
        }}
        [data-testid="stSidebar"] [data-testid*="new_chat_button_top"] > button:hover {{
            background-color: var(--app-primary-hover-color);
            border-color: var(--app-primary-hover-color);
        }}
        [data-testid="stSidebar"] [data-testid*="new_chat_button_top"] > button:disabled {{
            border-left-width: 1px !important; /* Reset active style for disabled new chat */
        }}

        [data-testid="stSidebar"] h3, [data-testid="stSidebar"] .stSubheader {{ /* Sidebar Section Headers */
            font-size: 0.75rem !important; text-transform: uppercase; font-weight: 600;
            color: var(--app-text-secondary-color);
            margin-top: var(--spacing-md); margin-bottom: var(--spacing-sm);
            letter-spacing: 0.03em;
        }}
        [data-testid="stSidebar"] [data-testid="stExpander"] {{
            border: 1px solid var(--app-divider-color);
            border-radius: var(--border-radius-md);
            background-color: var(--app-secondary-bg-color); /* Match sidebar bg */
            margin-bottom: var(--spacing-sm);
        }}
        [data-testid="stSidebar"] [data-testid="stExpander"] summary {{
            padding: 0.5rem var(--spacing-sm) !important;
            font-size: 0.8rem !important; font-weight: 500 !important;
            color: var(--app-text-color) !important;
            border-bottom: 1px solid var(--app-divider-color);
            border-top-left-radius: var(--border-radius-md); border-top-right-radius: var(--border-radius-md);
        }}
        [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {{
            background-color: color-mix(in srgb, var(--app-text-color) 4%, transparent);
        }}
        [data-testid="stSidebar"] [data-testid="stExpander"] div[data-testid="stExpanderDetails"] {{
            padding: var(--spacing-sm) !important;
             background-color: color-mix(in srgb, var(--app-bg-color) 50%, var(--app-secondary-bg-color) 50%); /* Slightly different from expander summary */
            border-bottom-left-radius: var(--border-radius-md); border-bottom-right-radius: var(--border-radius-md);
        }}
        [data-testid="stSidebar"] [data-testid="stExpander"][aria-label^="⚡ DAILY MODEL QUOTAS"] div[data-testid="stExpanderDetails"] {{
            padding: 0.4rem var(--spacing-xs) 0.1rem var(--spacing-xs) !important; /* More compact */
        }}
        [data-testid="stSidebar"] [data-testid="stExpander"][aria-label^="⚡ DAILY MODEL QUOTAS"] div[data-testid="stHorizontalBlock"] {{
            gap: 0.15rem !important; /* Tighter gap for quota items */
        }}


        /* Compact Quota Item Styling */
        .compact-quota-item {{
            display: flex; flex-direction: column; align-items: center;
            text-align: center; padding: var(--spacing-xs);
            background-color: color-mix(in srgb, var(--app-text-color) 2%, transparent);
            border-radius: var(--border-radius-sm);
            min-width: 30px; /* Ensure a minimum width */
        }}
        .cq-info {{ font-size: 0.65rem; margin-bottom: 2px; line-height: 1; white-space: nowrap; color: var(--app-text-color); }}
        .cq-bar-track {{
            width: 100%; height: 6px;
            background-color: color-mix(in srgb, var(--app-text-color) 10%, transparent);
            border: 1px solid var(--app-divider-color);
            border-radius: 3px; overflow: hidden; margin-bottom: 3px;
        }}
        .cq-bar-fill {{ height: 100%; border-radius: 3px; transition: width 0.3s ease-in-out, background-color 0.3s ease-in-out; }}
        .cq-value {{ font-size: 0.65rem; font-weight: 600; line-height: 1; }}

        /* Settings Panel in Sidebar */
        .settings-panel {{
            border: 1px solid var(--app-divider-color);
            border-radius: var(--border-radius-md); padding: var(--spacing-sm);
            margin-top: var(--spacing-xs); margin-bottom: var(--spacing-md);
            background-color: var(--app-bg-color); /* Slightly different from sidebar bg */
        }}
        .settings-panel .stTextInput input {{
            border-color: var(--app-border-color) !important;
            background-color: var(--app-secondary-bg-color) !important;
            color: var(--app-text-color) !important;
            font-size: 0.85rem;
        }}
        .settings-panel .stSubheader {{
             color: var(--app-text-color) !important;
             font-weight: 600 !important; font-size: 0.9rem !important;
             margin-bottom: var(--spacing-xs) !important;
        }}
        .settings-panel hr {{ border-top: 1px solid var(--app-divider-color); margin: var(--spacing-sm) 0; }}
        .detailed-quota-modelname {{
            font-weight: 600; font-size: 0.95em;
            margin-bottom: 0.2rem; display:block;
            color: var(--app-primary-color);
        }}
        .detailed-quota-block {{ font-size: 0.8rem; line-height: 1.5; }}
        .detailed-quota-block ul {{ list-style-type: none; padding-left: 0; margin-bottom: 0.3rem;}}
        .detailed-quota-block li {{ margin-bottom: 0.1rem; }}

        /* Chat Input Area */
        [data-testid="stChatInputContainer"] {{
            background-color: var(--app-secondary-bg-color);
            border-top: 1px solid var(--app-divider-color);
            padding: var(--spacing-sm) var(--spacing-md);
            box-shadow: 0 -2px 5px rgba(0,0,0,0.03);
        }}
        [data-testid="stChatInput"] textarea {{
            border: 1px solid var(--app-border-color) !important;
            border-radius: var(--border-radius-md) !important;
            background-color: var(--app-secondary-bg-color) !important; /* Match container */
            color: var(--app-text-color) !important;
            box-shadow: var(--shadow-sm) inset;
        }}
        [data-testid="stChatInput"] textarea:focus {{
            border-color: var(--app-primary-color) !important;
            box-shadow: 0 0 0 0.2rem color-mix(in srgb, var(--app-primary-color) 25%, transparent) !important;
        }}


        /* Chat Messages */
        [data-testid="stChatMessage"] {{
            border-radius: var(--border-radius-lg);
            padding: 0.8rem 1rem;
            margin-bottom: var(--spacing-sm);
            box-shadow: var(--shadow-sm);
            border: 1px solid transparent;
            max-width: 80%; /* Slightly reduce max width */
            line-height: 1.5;
        }}
        [data-testid="stChatMessage"] p {{ margin-bottom: 0.5em; }} /* Spacing between paragraphs in a message */
        [data-testid="stChatMessage"] p:last-child {{ margin-bottom: 0; }}

        [data-testid="stChatMessage"][data-testid^="stChatMessageUser"] {{
            background-color: var(--app-primary-color);
            color: white;
            margin-left: auto;
            border-bottom-right-radius: var(--border-radius-sm); /* Pointy corner */
        }}
        [data-testid="stChatMessage"][data-testid^="stChatMessageAssistant"] {{
            background-color: var(--app-secondary-bg-color);
            color: var(--app-text-color);
            margin-right: auto;
            border: 1px solid var(--app-divider-color);
            border-bottom-left-radius: var(--border-radius-sm); /* Pointy corner */
        }}
        /* Ensure avatars are vertically centered if they are taller than one line of text */
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            padding-top: 0.1rem; padding-bottom: 0.1rem; /* Adjust if avatars misalign */
        }


        .sidebar-divider {{
             margin: var(--spacing-md) 0; /* More prominent spacing for dividers */
             border: 0; border-top: 1px solid var(--app-divider-color);
        }}
        /* Utility for hiding Streamlit's default "Fork on GitHub" ribbon if desired */
        /* #GithubIcon {{ display: none; }} */
        
        /* Improve general button styling if st.button is used in main area */
        .main .stButton > button:not([data-testid*="new_chat_button_top"]):not([data-testid*="toggle_settings_button_sidebar"]):not([data-testid*="session_button_"]) {{
            border-radius: var(--border-radius-md);
            border: 1px solid var(--app-primary-color);
            background-color: var(--app-primary-color);
            color: white;
            padding: 0.5em 1em;
            font-weight: 500;
        }}
        .main .stButton > button:not([data-testid*="new_chat_button_top"]):not([data-testid*="toggle_settings_button_sidebar"]):not([data-testid*="session_button_"]):hover {{
            background-color: var(--app-primary-hover-color);
            border-color: var(--app-primary-hover-color);
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ----------------- API Key State Initialization -------------------
if "openrouter_api_key" not in st.session_state:
    app_conf = _load_app_config()
    st.session_state.openrouter_api_key = app_conf.get("openrouter_api_key", None)
if "api_key_auth_failed" not in st.session_state: st.session_state.api_key_auth_failed = False
api_key_is_syntactically_valid = is_api_key_valid(st.session_state.get("openrouter_api_key"))
app_requires_api_key_setup = not api_key_is_syntactically_valid or st.session_state.get("api_key_auth_failed", False)

# -------------------- Main Application Rendering -------------------
if app_requires_api_key_setup:
    st.set_page_config(page_title="OpenRouter API Key Setup", layout="centered")
    load_custom_css()
    st.title("🔒 OpenRouter API Key Required")
    st.markdown("---", unsafe_allow_html=True)
    if st.session_state.get("api_key_auth_failed"): st.error("API Key Authentication Failed. Please verify your key on OpenRouter.ai and re-enter.")
    elif not api_key_is_syntactically_valid and st.session_state.get("openrouter_api_key") is not None: st.error("The previously configured API Key has an invalid format. It must start with `sk-or-`.")
    else: st.info("Please configure your OpenRouter API Key to use the application.")
    st.markdown( "You can get a key from [OpenRouter.ai Keys](https://openrouter.ai/keys). Enter it below to continue." )
    new_key_input_val = st.text_input("Enter OpenRouter API Key", type="password", key="api_key_setup_input", value="", placeholder="sk-or-...")
    if st.button("Save and Validate API Key", key="save_api_key_setup_button", use_container_width=True, type="primary"):
        if is_api_key_valid(new_key_input_val):
            st.session_state.openrouter_api_key = new_key_input_val
            _save_app_config(new_key_input_val); st.session_state.api_key_auth_failed = False
            with st.spinner("Validating API Key..."): fetched_credits_data = get_credits()
            if st.session_state.get("api_key_auth_failed"): st.error("Authentication failed with the provided API Key."); time.sleep(0.5); st.rerun()
            elif fetched_credits_data == (None, None, None): st.error("Could not validate API Key. Network or API provider issue.")
            else:
                st.success("API Key saved and validated! Initializing application...")
                if "credits" not in st.session_state: st.session_state.credits = {}
                st.session_state.credits["total"], st.session_state.credits["used"], st.session_state.credits["remaining"] = fetched_credits_data
                st.session_state.credits_ts = time.time(); time.sleep(1.0); st.rerun()
        elif not new_key_input_val: st.warning("API Key field cannot be empty.")
        else: st.error("Invalid API key format. It must start with 'sk-or-'.")
    st.markdown("---", unsafe_allow_html=True); st.caption("Your API key is stored locally in `app_config.json`.")
else:
    st.set_page_config(page_title="OpenRouter Chat", layout="wide", initial_sidebar_state="expanded")
    load_custom_css()
    if "settings_panel_open" not in st.session_state: st.session_state.settings_panel_open = False
    needs_save_session = False
    if "sid" not in st.session_state: st.session_state.sid = _new_sid(); needs_save_session = True
    elif st.session_state.sid not in sessions:
        logging.warning(f"Session ID {st.session_state.sid} not found. Creating new chat."); st.session_state.sid = _new_sid(); needs_save_session = True

    if needs_save_session: _save(SESS_FILE, sessions); st.rerun()

    if "credits" not in st.session_state: st.session_state.credits = {"total": 0.0, "used": 0.0, "remaining": 0.0}; st.session_state.credits_ts = 0
    credits_are_stale = time.time() - st.session_state.get("credits_ts", 0) > 3600
    credits_are_default_and_old = (st.session_state.credits.get("total") == 0.0 and st.session_state.credits.get("used") == 0.0 and st.session_state.credits.get("remaining") == 0.0 and st.session_state.get("credits_ts", 0) != 0 and time.time() - st.session_state.get("credits_ts", 0) > 10)
    credits_never_fetched = st.session_state.get("credits_ts", 0) == 0
    if credits_are_stale or credits_are_default_and_old or credits_never_fetched:
        logging.info("Refreshing credits (stale, default/old, or never fetched).")
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
            st.markdown("<div class='settings-panel'>", unsafe_allow_html=True)
            st.subheader("🔑 API Key Configuration")
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
                    if st.session_state.get("api_key_auth_failed"): st.error("New API Key failed authentication.")
                    elif credits_data == (None,None,None): st.warning("Could not validate new API key. Saved, but functionality may be affected.")
                    else:
                        st.success("New API Key saved and validated!")
                        st.session_state.credits["total"],st.session_state.credits["used"],st.session_state.credits["remaining"] = credits_data
                        st.session_state.credits_ts = time.time()
                    time.sleep(0.8); st.rerun()
                elif not new_key_input_sidebar: st.warning("API Key field empty. No changes.")
                else: st.error("Invalid API key format. Must start with 'sk-or-'.")

            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("📊 Detailed Model Quotas")
            _ensure_quota_data_is_current()

            for m_key_loop in sorted(NEW_PLAN_CONFIG.keys()):
                if m_key_loop not in MODEL_MAP: continue

                stats = get_quota_usage_and_limits(m_key_loop)
                if not stats:
                    st.markdown(f"**{EMOJI.get(m_key_loop, '')} {m_key_loop} ({MODEL_MAP[m_key_loop].split('/')[-1]})**: Could not retrieve quota details.")
                    continue

                model_short_name = MODEL_DESCRIPTIONS.get(m_key_loop, "").split('(')[1].split(')')[0] if '(' in MODEL_DESCRIPTIONS.get(m_key_loop, "") else MODEL_MAP[m_key_loop].split('/')[-1]
                model_name_display = f"{EMOJI.get(m_key_loop, '')} <span class='detailed-quota-modelname'>{m_key_loop} ({model_short_name})</span>"
                st.markdown(f"{model_name_display}", unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="detailed-quota-block">
                    <ul>
                        <li><b>Daily Msgs:</b> {stats['used_daily_msg']}/{stats['limit_daily_msg']}</li>
                        <li><b>Daily In Tok:</b> {format_token_count(stats['used_daily_in_tokens'])}/{format_token_count(stats['limit_daily_in_tokens'])}</li>
                        <li><b>Daily Out Tok:</b> {format_token_count(stats['used_daily_out_tokens'])}/{format_token_count(stats['limit_daily_out_tokens'])}</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="detailed-quota-block">
                    <ul>
                        <li><b>Monthly Msgs:</b> {stats['used_monthly_msg']}/{stats['limit_monthly_msg']}</li>
                        <li><b>Monthly In Tok:</b> {format_token_count(stats['used_monthly_in_tokens'])}/{format_token_count(stats['limit_monthly_in_tokens'])}</li>
                        <li><b>Monthly Out Tok:</b> {format_token_count(stats['used_monthly_out_tokens'])}/{format_token_count(stats['limit_monthly_out_tokens'])}</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)

                if m_key_loop == "A" and stats["limit_3hr_msg"] != float('inf'):
                    time_until_next_msg_str = ""
                    active_model_a_calls = sorted(_g_quota_data.get(MODEL_A_3H_CALLS_KEY, []))
                    if len(active_model_a_calls) >= stats['limit_3hr_msg']:
                         oldest_blocking_call_ts = active_model_a_calls[0]
                         expiry_time = oldest_blocking_call_ts + NEW_PLAN_CONFIG["A"][7]
                         time_remaining_seconds = expiry_time - time.time()
                         if time_remaining_seconds > 0:
                            mins, secs = divmod(int(time_remaining_seconds), 60)
                            hrs, mins_rem = divmod(mins, 60)
                            if hrs > 0:
                                time_until_next_msg_str = f" (Next in {hrs}h {mins_rem}m)"
                            else:
                                time_until_next_msg_str = f" (Next in {mins_rem}m {secs}s)"
                    st.markdown(f"""
                    <div class="detailed-quota-block" style="margin-top: -0.5rem; margin-left:0.1rem;">
                    <ul>
                    <li><b>3-Hour Msgs:</b> {stats['used_3hr_msg']}/{int(stats['limit_3hr_msg'])}{time_until_next_msg_str}</li>
                    </ul>
                    </div>""", unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True) # End settings-panel

        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
        # Create a container for logo and title for better alignment
        st.markdown("<div class='sidebar-title-container'>", unsafe_allow_html=True)
        logo_title_cols = st.columns([1, 5], gap="small") # Adjusted column ratio
        with logo_title_cols[0]: st.image("https://avatars.githubusercontent.com/u/130328222?s=200&v=4", width=40)
        with logo_title_cols[1]: st.title("OpenRouter Chat")
        st.markdown("</div>", unsafe_allow_html=True) # Close sidebar-title-container
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)


        with st.expander("⚡ DAILY MODEL QUOTAS", expanded=True):
            active_model_keys_for_display = sorted([k for k in MODEL_MAP if k in NEW_PLAN_CONFIG])
            if not active_model_keys_for_display: st.caption("No models configured.")
            else:
                _ensure_quota_data_is_current()
                if active_model_keys_for_display:
                    quota_cols = st.columns(len(active_model_keys_for_display))
                else: quota_cols = [st.container()]

                for i, m_key in enumerate(active_model_keys_for_display):
                    with quota_cols[i]:
                        left_d_msgs = get_remaining_daily_messages(m_key)
                        lim_d_msgs = NEW_PLAN_CONFIG.get(m_key, (0,))[0]
                        if lim_d_msgs > 0:
                            pct_float = max(0.0, min(1.0, left_d_msgs / lim_d_msgs))
                            fill_width_val = int(pct_float * 100)
                            left_display = str(left_d_msgs)
                        else: pct_float, fill_width_val, left_display = 0.0, 0, "0"

                        bar_color = "var(--app-danger-color)"
                        if pct_float > 0.5: bar_color = "var(--app-success-color)"
                        elif pct_float > 0.25: bar_color = "var(--app-warning-color)"
                        emoji_char = EMOJI.get(m_key, "❔")
                        st.markdown(f"""<div class="compact-quota-item"><div class="cq-info">{emoji_char} <b>{m_key}</b></div><div class="cq-bar-track"><div class="cq-bar-fill" style="width: {fill_width_val}%; background-color: {bar_color};"></div></div><div class="cq-value" style="color: {bar_color};">{left_display}</div></div>""", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
        current_session_is_truly_blank = (st.session_state.sid in sessions and sessions[st.session_state.sid].get("title") == "New chat" and not sessions[st.session_state.sid].get("messages"))
        if st.button("➕ New chat", key="new_chat_button_top", use_container_width=True, disabled=current_session_is_truly_blank):
            st.session_state.sid = _new_sid();
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
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
        st.subheader("Model-Routing Map")
        st.caption(f"Router: {ROUTER_MODEL_ID.split('/')[-1]}") # Shorter router name
        with st.expander("Letters → Models", expanded=False):
            for k_model in sorted(MODEL_MAP.keys()):
                desc_parts = MODEL_DESCRIPTIONS.get(k_model, MODEL_MAP.get(k_model, "N/A")).split("(")
                main_desc = desc_parts[0].strip()
                model_name_in_desc = desc_parts[1].split(")")[0] if len(desc_parts) > 1 else MODEL_MAP.get(k_model, "N/A").split('/')[-1]
                max_tok = MAX_TOKENS.get(k_model, 0)
                st.markdown(f"**{k_model}**: {main_desc} ({model_name_in_desc}) <br><small style='color:var(--app-text-secondary-color);'>Max Output: {max_tok:,} tokens</small>", unsafe_allow_html=True)
            st.markdown(f"**{FALLBACK_MODEL_KEY}**: {FALLBACK_MODEL_EMOJI} {FALLBACK_MODEL_ID.split('/')[-1]} <br><small style='color:var(--app-text-secondary-color);'>Max Output: {FALLBACK_MODEL_MAX_TOKENS:,} tokens</small>", unsafe_allow_html=True)

        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
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
            if tot is None or used is None or rem is None or st.session_state.get("api_key_auth_failed"):
                 st.warning("Could not fetch/display credits.")
            else: st.markdown(f"**Remaining:** ${float(rem):.2f} cr <br>**Used:** ${float(used):.2f} cr", unsafe_allow_html=True)
            ts = st.session_state.get("credits_ts", 0)
            last_updated_str = datetime.fromtimestamp(ts, TZ).strftime('%-d %b, %H:%M') if ts else "N/A" # Shorter time
            st.caption(f"Last updated: {last_updated_str}")


    # ---- Main chat area ----
    if st.session_state.sid not in sessions:
        logging.error(f"CRITICAL: SID {st.session_state.sid} missing. Resetting."); st.session_state.sid = _new_sid()
        _save(SESS_FILE, sessions); st.rerun();
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
        else:
            _ensure_quota_data_is_current()
            allowed_standard_models = [k for k in MODEL_MAP if is_model_available(k)]

            use_fallback, chosen_model_key, model_id_to_use, max_tokens_api, avatar_resp = (False, None, None, None, "🤖")

            routed_key_letter = route_choice(prompt, allowed_standard_models, chat_history)

            if st.session_state.get("api_key_auth_failed"):
                st.error("API Auth failed during model routing. Check Key in Settings.")
            elif routed_key_letter == FALLBACK_MODEL_KEY:
                logging.warning(f"Router chose FALLBACK_MODEL_KEY. Using free fallback: {FALLBACK_MODEL_ID}.")
                # st.warning(f"{FALLBACK_MODEL_EMOJI} Router determined no standard models suitable. Using free fallback.") # User doesn't need to see this detail always
                use_fallback, chosen_model_key, model_id_to_use, max_tokens_api, avatar_resp = (True, FALLBACK_MODEL_KEY, FALLBACK_MODEL_ID, FALLBACK_MODEL_MAX_TOKENS, FALLBACK_MODEL_EMOJI)
            elif routed_key_letter not in MODEL_MAP or not is_model_available(routed_key_letter):
                logging.warning(f"Router chose '{routed_key_letter}' (invalid/no quota). Using fallback {FALLBACK_MODEL_ID}.")
                # st.warning(f"{FALLBACK_MODEL_EMOJI} Model routing issue or chosen model '{routed_key_letter}' unavailable. Using free fallback.")
                use_fallback, chosen_model_key, model_id_to_use, max_tokens_api, avatar_resp = (True, FALLBACK_MODEL_KEY, FALLBACK_MODEL_ID, FALLBACK_MODEL_MAX_TOKENS, FALLBACK_MODEL_EMOJI)
            else:
                chosen_model_key = routed_key_letter
                model_id_to_use = MODEL_MAP[chosen_model_key]
                max_tokens_api = MAX_TOKENS.get(chosen_model_key, FALLBACK_MODEL_MAX_TOKENS)
                avatar_resp = EMOJI.get(chosen_model_key, "🤖")

            if model_id_to_use and not st.session_state.get("api_key_auth_failed"):
                with st.chat_message("assistant", avatar=avatar_resp):
                    response_placeholder, full_response = st.empty(), ""
                    api_call_ok = True
                    for chunk_content, err_msg in streamed(model_id_to_use, chat_history, max_tokens_api):
                        if st.session_state.get("api_key_auth_failed"):
                            full_response = "❗ **API Authentication Error**: Update Key in ⚙️ Settings."
                            api_call_ok = False; break
                        if err_msg:
                            full_response = f"❗ **API Error**: {err_msg}"
                            api_call_ok = False; break
                        if chunk_content:
                            full_response += chunk_content
                            response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)

                last_usage = st.session_state.pop("last_stream_usage", None)
                prompt_tokens_used = 0
                completion_tokens_used = 0

                if last_usage:
                    prompt_tokens_used = last_usage.get("prompt_tokens", 0)
                    completion_tokens_used = last_usage.get("completion_tokens", 0)
                    logging.info(f"API call completed for model {model_id_to_use}. Tokens: P={prompt_tokens_used}, C={completion_tokens_used}")
                else:
                    logging.warning(f"Token usage info not found for model {model_id_to_use}.")

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
                    if sessions[current_sid]["title"] == "New chat" and prompt and full_response:
                       sessions[current_sid]["title"] = _autoname(prompt)

                _save(SESS_FILE, sessions)
                st.rerun()

            elif st.session_state.get("api_key_auth_failed"):
                time.sleep(0.5); st.rerun()
            else:
                st.error("Unexpected error: Could not determine a model to use.")
                logging.error("No model_id and no API auth failure after selection logic.")
                _save(SESS_FILE, sessions); st.rerun()
