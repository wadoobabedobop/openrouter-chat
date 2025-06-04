#!/usr/bin/env python3
# -*- coding: utf-8 -*- # THIS SHOULD BE LINE 2
"""
OpenRouter Streamlit Chat — Full Edition + Search
• Persistent chat sessions
• Daily/weekly/monthly quotas
• Pretty ‘token-jar’ gauges (fixed at top)
• Detailed model-routing panel (Mistral router) + Perplexity Search Models
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
# Order roughly reflects increasing cost/capability for *non-search* models
MODEL_MAP = {
    "A": "google/gemini-2.5-pro-preview",    # Gemini 2.5 Pro (High Cap, Mod Cost)
    "B": "openai/o4-mini",                   # GPT-4o mini (Mid-stakes, Cost-effective)
    "C": "openai/chatgpt-4o-latest",         # GPT-4o (Polished, Highest Cost)
    "D": "deepseek/deepseek-r1",             # DeepSeek R1 (Cheap Factual/Technical)
    "E": "anthropic/claude-3.7-sonnet",      # Claude 3.7 Sonnet (Novel/Creative, High Cost)
    "F": "google/gemini-2.5-flash-preview",   # Gemini 2.5 Flash (Fast, Cheapest)
    # --- NEW SEARCH MODELS ---
    "G": "perplexity/sonar",                 # Sonar (Cheap Search)
    "H": "perplexity/sonar-reasoning-pro",   # Sonar Reasoning Pro (Adv Search/Reasoning, Expensive)
    "I": "perplexity/sonar-deep-research",   # Sonar Deep Research (Deep Investigation, Expensive)
}
# ROUTER_MODEL_ID = "mistralai/mistral-7b-instruct:free"
# ROUTER_MODEL_ID = "nousresearch/deephermes-3-mistral-24b-preview:free" # Seems reliable
ROUTER_MODEL_ID = "nousresearch/deephermes-3-mistral-24b-preview:free" # Try the best router, cost is minimal
MAX_HISTORY_CHARS_FOR_ROUTER = 3000  # Approx. 750 tokens for history context

MAX_TOKENS = { # Per-call max_tokens for API request (max output generation length)
    "A": 16_000, "B": 8_000, "C": 16_000,
    "D": 8_000, "E": 8_000, "F": 8_000,
    # --- Search Model Max Tokens (Check OpenRouter/Perplexity for exact limits, using estimates) ---
    "G": 16_000, # Sonar (often handles long contexts)
    "H": 16_000, # Sonar Pro
    "I": 16_000  # Sonar Deep
}

# QUOTA CONFIGURATION (Added G, H, I)
# Format: (DailyMsgs, MonthlyMsgs, DailyInTok, MonthlyInTok, DailyOutTok, MonthlyOutTok, 3hrMsgs, 3hrWindowSec)
# Costs roughly: F < G < D < B < A ≈ H ≈ I < E < C
NEW_PLAN_CONFIG = {
    "A": (5, 200, 5000, 100000, 5000, 100000, 3, 3 * 3600), # Gemini Pro
    "B": (10, 200, 5000, 100000, 5000, 100000, 0, 0),       # o4-mini
    "C": (5, 200, 5000, 100000, 5000, 100000, 0, 0),       # GPT-4o
    "D": (10, 200, 5000, 100000, 5000, 100000, 0, 0),       # DeepSeek R1
    "E": (5, 200, 5000, 100000, 5000, 100000, 0, 0),       # Sonnet
    "F": (150, 3000, 75000, 1500000, 75000, 1500000, 0, 0), # Flash
    # --- Search Model Quotas ---
    "G": (50, 1000, 20000, 400000, 20000, 400000, 0, 0),    # Sonar (cheaper, higher volume)
    "H": (10, 200, 5000, 100000, 5000, 100000, 0, 0),       # Sonar Pro (expensive, like A/E)
    "I": (5, 100, 5000, 100000, 5000, 100000, 0, 0)         # Sonar Deep (expensive, lower vol)
}

EMOJI = {
    "A": "🌟", "B": "🔷", "C": "🟥", "D": "🟢", "E": "🖋️", "F": "🌀",
    # --- Search Emojis ---
    "G": "🔎", "H": "💡", "I": "📚"
}

# MODEL_DESCRIPTIONS (Reflects cost F < G < D < B < A ≈ H ≈ I < E < C)
MODEL_DESCRIPTIONS = {
    "A": "🌟 (gemini-2.5-pro) – High capability, moderate cost.",
    "B": "🔷 (o4-mini) – Mid-stakes reasoning, cost-effective.",
    "C": "🟥 (chatgpt-4o) – Polished/empathetic, HIGHEST cost.",
    "D": "🟢 (deepseek-r1) – Cheap factual/technical reasoning.",
    "E": "🖋️ (claude-3.7-sonnet) – Novel, creative, high cost.",
    "F": "🌀 (gemini-2.5-flash) – Quick, CHEAPEST, simple NON-SEARCH tasks.",
    # --- Search Descriptions ---
    "G": "🔎 (sonar) – Cheap web search, simple queries + response.",
    "H": "💡 (sonar-pro) – Adv. search + reasoning, complex search Qs.",
    "I": "📚 (sonar-deep) – Deep multi-step search investigation.",
}

# ROUTER_MODEL_GUIDANCE (Reflects cost F < G < D < B < A ≈ H ≈ I < E < C, focuses on adequacy, includes Search)
ROUTER_MODEL_GUIDANCE_SENSITIVE = {
    "A": "(Model A: High Capability [Cost Rank 5/9]) Use for complex non-search tasks. **Suitable for sensitive/crisis topics if E is unavailable.** Cheaper than E, C.",
    "B": "(Model B: Mid-Tier [Cost Rank 4/9]) Use for general moderate non-search tasks. Suitable for *mild-to-moderate* sensitivity. **Generally AVOID for severe crisis/self-harm unless E, A, C are all unavailable.** Cheaper than A, H, I, E, C.",
    "C": "(Model C: Polished, HIGHEST COST [Cost Rank 9/9]) Avoid unless extreme polish is essential AND cheaper options inadequate. **Can be a fallback for crisis if E and A are unavailable.**",
    "D": "(Model D: Factual/Technical [Cost Rank 3/9]) Use for factual/code tasks if F/G are insufficient. **NOT suitable for sensitive topics.** Slow.",
    "E": "(Model E: Novel & Nuanced, High Cost [Cost Rank 8/9]) Use for unique creative non-search tasks OR **handling serious sensitive topics/crisis situations.** **Preferred choice for crisis if available.** Cheaper than C.",
    "F": "(Model F: CHEAPEST [Cost Rank 1/9]) Use ONLY for simple, low-stakes, **NON-SEARCH**, non-sensitive tasks. ***DO NOT USE 'F' IF*** query involves: search, complexity, sensitivity (esp. crisis/safety), math, deep analysis, high accuracy needs.",
    # --- Search Guidance ---
    "G": "(Model G: Cheap Search [Cost Rank 2/9]) **Use ONLY for queries requiring web search.** Best for *simple* search needs (e.g., 'latest news', 'weather', 'define X'). If query is complex *beyond* the search itself, prefer H. **NOT suitable for sensitive topics.**",
    "H": "(Model H: Adv Search+Reasoning [Cost Rank 6/9]) **Use ONLY for queries requiring web search.** Use for *complex* search queries needing reasoning *over search results* or answering nuanced questions based on current info. Prefer over G if query complexity warrants the cost. **NOT suitable for sensitive topics.**",
    "I": "(Model I: Deep Search [Cost Rank 7/9]) **Use ONLY for queries requiring extensive web research.** Use for explicit requests for *deep investigation*, multi-step research, or comprehensive reports based on searching multiple sources. High cost. **NOT suitable for sensitive topics.**"
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

def _save_app_config(api_key_value: str = None, dark_mode_value: bool = None):
    config_data = _load_app_config()
    if api_key_value is not None:
        config_data["openrouter_api_key"] = api_key_value
    if dark_mode_value is not None:
        config_data["dark_mode"] = bool(dark_mode_value)
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

# --------------------- Quota Management (Should adapt to new keys) ------------------------
_g_quota_data = None
_g_quota_data_last_refreshed_stamps = {"d": None, "m": None}

USAGE_KEYS_PERIODIC = ["d_u", "m_u", "d_it_u", "m_it_u", "d_ot_u", "m_ot_u"]
MODEL_A_3H_CALLS_KEY = "model_A_3h_calls" # Specific to Model A

# --- Quota functions (_reset, _ensure_quota_data_is_current, get_quota_usage_and_limits, is_model_available, record_use) ---
# These functions *should* automatically handle the new model keys (G, H, I)
# as they primarily rely on iterating over NEW_PLAN_CONFIG.keys() and accessing
# the global quota data structure (_g_quota_data) which gets updated by _reset.
# No structural changes should be needed here unless a new *type* of quota (e.g., weekly) is added.
# We'll just verify the logging confirms they are picked up.

def _reset(block: dict, period_prefix: str, current_stamp: str, model_keys_zeros: dict) -> bool:
    data_changed = False
    period_stamp_key = period_prefix

    if block.get(period_stamp_key) != current_stamp:
        block[period_stamp_key] = current_stamp
        for usage_type_suffix in ["_u", "_it_u", "_ot_u"]:
            usage_dict_key = f"{period_prefix}{usage_type_suffix}"
            block[usage_dict_key] = model_keys_zeros.copy() # Initialize with ALL current model keys
        data_changed = True
        logging.info(f"Quota period '{period_stamp_key}' reset for new stamp '{current_stamp}'. Initialized models: {list(model_keys_zeros.keys())}")
    else:
        # Ensure all current model keys exist even if period didn't reset
        for usage_type_suffix in ["_u", "_it_u", "_ot_u"]:
            usage_dict_key = f"{period_prefix}{usage_type_suffix}"
            if usage_dict_key not in block:
                block[usage_dict_key] = model_keys_zeros.copy()
                data_changed = True
                logging.info(f"Initialized missing usage dict '{usage_dict_key}' for stamp '{current_stamp}'. Models: {list(model_keys_zeros.keys())}")
            else:
                current_period_usage_dict = block[usage_dict_key]
                # Add any *new* model keys defined in config but missing from this dict
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
        logging.info(f"Quota period change detected (Day: {_g_quota_data_last_refreshed_stamps['d']}->{now_d_stamp}, Month: {_g_quota_data_last_refreshed_stamps['m']}->{now_m_stamp}). Refreshing quota data.")

    if not needs_full_refresh_logic:
        # Prune 3-hour calls for Model A even without a full reset
        if MODEL_A_3H_CALLS_KEY in _g_quota_data and "A" in NEW_PLAN_CONFIG and NEW_PLAN_CONFIG["A"][7] > 0:
            _, _, _, _, _, _, _, three_hr_window_seconds = NEW_PLAN_CONFIG["A"]
            current_time = time.time()
            original_len = len(_g_quota_data.get(MODEL_A_3H_CALLS_KEY, [])) # Use get for safety
            _g_quota_data[MODEL_A_3H_CALLS_KEY] = [
                ts for ts in _g_quota_data.get(MODEL_A_3H_CALLS_KEY, []) # Use get for safety
                if current_time - ts < three_hr_window_seconds
            ]
            if len(_g_quota_data.get(MODEL_A_3H_CALLS_KEY, [])) != original_len:
                logging.info(f"Pruned Model A 3-hour call timestamps. Original: {original_len}, New: {len(_g_quota_data[MODEL_A_3H_CALLS_KEY])}.")
                _save(QUOTA_FILE, _g_quota_data) # Save if pruned
        return _g_quota_data

    q_loaded_data = _load(QUOTA_FILE, {})
    data_was_modified = _g_quota_data is None # Track if any changes occur

    # Use NEW_PLAN_CONFIG keys as the source of truth for active models
    active_model_keys = set(NEW_PLAN_CONFIG.keys())
    logging.info(f"Active models defined in NEW_PLAN_CONFIG: {sorted(list(active_model_keys))}")
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

    # Remove other potentially obsolete top-level keys (e.g., old weekly format)
    obsolete_keys = ["w", "w_u"] # Add others if needed
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
    # Check if Model A exists in config and has a 3hr limit configured
    if "A" in NEW_PLAN_CONFIG and len(NEW_PLAN_CONFIG["A"]) > 7 and NEW_PLAN_CONFIG["A"][6] > 0 and NEW_PLAN_CONFIG["A"][7] > 0:
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
    logging.debug(f"Quota data refreshed. Current state: {json.dumps(_g_quota_data, indent=2)}")
    return _g_quota_data

def get_quota_usage_and_limits(model_key: str):
    if model_key not in NEW_PLAN_CONFIG:
        # Handle FALLBACK_MODEL_KEY explicitly - it has no quotas
        if model_key == FALLBACK_MODEL_KEY:
            return {"used_daily_msg": 0, "limit_daily_msg": float('inf'), # No limits
                    "used_monthly_msg": 0, "limit_monthly_msg": float('inf'),
                    "used_daily_in_tokens": 0, "limit_daily_in_tokens": float('inf'),
                    "used_monthly_in_tokens": 0, "limit_monthly_in_tokens": float('inf'),
                    "used_daily_out_tokens": 0, "limit_daily_out_tokens": float('inf'),
                    "used_monthly_out_tokens": 0, "limit_monthly_out_tokens": float('inf'),
                    "used_3hr_msg": 0, "limit_3hr_msg": float('inf')}
        logging.error(f"Model key '{model_key}' not in NEW_PLAN_CONFIG. Cannot get quotas.")
        return {} # Return empty dict for unknown quota-tracked models

    current_q_data = _ensure_quota_data_is_current()
    plan = NEW_PLAN_CONFIG[model_key]

    # Check if plan has enough elements (including 3hr limits)
    has_3hr_limit = len(plan) > 7 and plan[6] > 0 and plan[7] > 0

    limits = {
        "limit_daily_msg": plan[0], "limit_monthly_msg": plan[1],
        "limit_daily_in_tokens": plan[2], "limit_monthly_in_tokens": plan[3],
        "limit_daily_out_tokens": plan[4], "limit_monthly_out_tokens": plan[5],
        "limit_3hr_msg": plan[6] if has_3hr_limit else float('inf')
    }

    usage = {
        "used_daily_msg": current_q_data.get("d_u", {}).get(model_key, 0),
        "used_monthly_msg": current_q_data.get("m_u", {}).get(model_key, 0),
        "used_daily_in_tokens": current_q_data.get("d_it_u", {}).get(model_key, 0),
        "used_monthly_in_tokens": current_q_data.get("m_it_u", {}).get(model_key, 0),
        "used_daily_out_tokens": current_q_data.get("d_ot_u", {}).get(model_key, 0),
        "used_monthly_out_tokens": current_q_data.get("m_ot_u", {}).get(model_key, 0),
        "used_3hr_msg": 0 # Default to 0
    }

    # Specifically handle Model A 3-hour limit if configured
    if model_key == "A" and has_3hr_limit:
        current_time = time.time()
        three_hr_window_seconds = plan[7]
        # Get the list safely, default to empty list if key doesn't exist yet
        recent_calls = [
            ts for ts in current_q_data.get(MODEL_A_3H_CALLS_KEY, [])
            if current_time - ts < three_hr_window_seconds
        ]
        usage["used_3hr_msg"] = len(recent_calls)
        # No need to re-save here, pruning happens in _ensure_quota_data_is_current

    return {**usage, **limits}

def is_model_available(model_key: str) -> bool:
    # Free fallback is always considered available quota-wise
    if model_key == FALLBACK_MODEL_KEY:
        return True
    # Check if model exists in quota config
    if model_key not in NEW_PLAN_CONFIG:
        logging.warning(f"is_model_available: Model key '{model_key}' not in NEW_PLAN_CONFIG. Assuming unavailable.")
        return False

    stats = get_quota_usage_and_limits(model_key)
    if not stats: return False # Should not happen if key is in NEW_PLAN_CONFIG

    # Check all standard quotas (only if limit > 0 and is not infinity)
    if stats["limit_daily_msg"] != float('inf') and stats["limit_daily_msg"] > 0 and stats["used_daily_msg"] >= stats["limit_daily_msg"]: return False
    if stats["limit_monthly_msg"] != float('inf') and stats["limit_monthly_msg"] > 0 and stats["used_monthly_msg"] >= stats["limit_monthly_msg"]: return False
    if stats["limit_daily_in_tokens"] != float('inf') and stats["limit_daily_in_tokens"] > 0 and stats["used_daily_in_tokens"] >= stats["limit_daily_in_tokens"]: return False
    if stats["limit_monthly_in_tokens"] != float('inf') and stats["limit_monthly_in_tokens"] > 0 and stats["used_monthly_in_tokens"] >= stats["limit_monthly_in_tokens"]: return False
    if stats["limit_daily_out_tokens"] != float('inf') and stats["limit_daily_out_tokens"] > 0 and stats["used_daily_out_tokens"] >= stats["limit_daily_out_tokens"]: return False
    if stats["limit_monthly_out_tokens"] != float('inf') and stats["limit_monthly_out_tokens"] > 0 and stats["used_monthly_out_tokens"] >= stats["limit_monthly_out_tokens"]: return False

    # Check specific Model A 3-hour limit if applicable and finite
    if model_key == "A" and stats["limit_3hr_msg"] != float('inf'):
        if stats["used_3hr_msg"] >= stats["limit_3hr_msg"]: return False

    return True

def get_remaining_daily_messages(model_key: str) -> int:
    if model_key == FALLBACK_MODEL_KEY: return float('inf')
    if model_key not in NEW_PLAN_CONFIG: return 0
    stats = get_quota_usage_and_limits(model_key)
    if not stats: return 0
    # Return infinity if limit is infinite or 0
    if stats["limit_daily_msg"] == float('inf') or stats["limit_daily_msg"] <= 0: return float('inf')
    return max(0, stats["limit_daily_msg"] - stats["used_daily_msg"])

def record_use(model_key: str, prompt_tokens: int, completion_tokens: int):
    # Do not record usage for the free fallback model or unknown models
    if model_key == FALLBACK_MODEL_KEY or model_key not in NEW_PLAN_CONFIG:
        logging.debug(f"Skipping usage recording for non-quota-tracked model key: {model_key}")
        return

    current_q_data = _ensure_quota_data_is_current()

    # Ensure all necessary usage dictionaries and the specific model key exist
    # setdefault should handle this gracefully if the key already exists
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
    # Check if Model A is in config and has the 3hr limit configured
    if model_key == "A" and "A" in NEW_PLAN_CONFIG and len(NEW_PLAN_CONFIG["A"]) > 7 and NEW_PLAN_CONFIG["A"][6] > 0:
        current_q_data.setdefault(MODEL_A_3H_CALLS_KEY, []).append(time.time())
        # Pruning happens in _ensure_quota_data_is_current, just append here

    _save(QUOTA_FILE, current_q_data)
    logging.info(f"Recorded usage for model '{model_key}': 1 msg, {prompt_tokens}p, {completion_tokens}c tokens. Quotas saved.")

# --------------------- Session Management (Unchanged) -----------------------
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
    _delete_unused_blank_sessions(keep_sid=sid)
    return sid

def _autoname(seed: str) -> str:
    words = seed.strip().split()
    cand = " ".join(words[:4]) or "Chat"
    return (cand[:30] + "…") if len(cand) > 30 else cand

# --------------------------- Logging (Unchanged) ----------------------------
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
numeric_level = getattr(logging, log_level, logging.INFO)
if not logging.getLogger().handlers:
    logging.basicConfig(level=numeric_level, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)
    logging.info(f"Logging level set to: {log_level}")

def is_api_key_valid(api_key_value):
    return api_key_value and isinstance(api_key_value, str) and api_key_value.startswith("sk-or-")

# -------------------------- API Calls (Unchanged) --------------------------
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
        logging.error(f"API POST failed with network error: {e}")
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
                    error_info = chunk["error"]
                    if isinstance(error_info, dict): msg = error_info.get("message", "Unknown API error"); code = error_info.get("code", "N/A")
                    elif isinstance(error_info, str): msg = error_info; code = "N/A"
                    else: msg = "Unknown API error format"; code = "N/A"
                    logging.error(f"API chunk error (Code: {code}): {msg}"); yield None, msg; return

                if chunk.get("choices") and isinstance(chunk["choices"], list) and len(chunk["choices"]) > 0:
                    if "usage" in chunk and chunk["usage"] is not None: st.session_state.last_stream_usage = chunk["usage"]
                    delta = chunk["choices"][0].get("delta", {}).get("content")
                    if delta is not None: yield delta, None
                # else: logging.debug(f"Received chunk without expected content delta: {data}") # Optional debug

    except ValueError as ve: logging.error(f"ValueError during streamed call setup: {ve}"); yield None, str(ve)
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else 'N/A'; text = e.response.text if e.response else 'No response text'
        logging.error(f"Stream HTTPError {status_code}: {text}")
        if status_code == 401: st.session_state.api_key_auth_failed = True
        yield None, f"HTTP {status_code}: An error occurred with the API provider. Details: {text}"
    except requests.exceptions.RequestException as e: logging.error(f"Stream Network Error: {e}"); yield None, f"Network Error: Failed to connect to API. {e}"
    except Exception as e: logging.exception(f"Unexpected error during streamed API call: {e}"); yield None, f"An unexpected error occurred: {e}"

# ------------------------- Model Routing (REVISED V4 - Incorporates Search) -----------------------

def _pick_fallback_model(allowed: list[str]) -> str:
    """Return the best fallback model letter from the allowed list."""
    allowed = list(dict.fromkeys(allowed))
    non_search = [k for k in allowed if k in "ABCDEF"]
    search = [k for k in allowed if k in "GHI"]

    if "F" in non_search:
        return "F"
    if "G" in search:
        return "G"
    if non_search:
        return non_search[0]
    if search:
        return search[0]
    if "F" in MODEL_MAP:
        return "F"
    if "G" in MODEL_MAP:
        return "G"
    return next(iter(MODEL_MAP), FALLBACK_MODEL_KEY)


def _extract_letter(resp: str, allowed: list[str]) -> str | None:
    """Parse router response text and return the first allowed letter if any."""
    resp = (resp or "").upper()
    if resp in allowed:
        return resp
    for char in resp:
        if char in allowed:
            return char
    return None


def route_choice(user_msg: str, allowed: list[str], chat_history: list) -> str:
    # Determine fallback choice using helper
    fallback_choice_letter = _pick_fallback_model(allowed)

    if not MODEL_MAP:
        logging.error("Router: No models configured. Using FALLBACK_MODEL_KEY.")
        return FALLBACK_MODEL_KEY

    if not allowed:
        logging.warning(f"Router: No models available due to quotas. Defaulting to free fallback: {FALLBACK_MODEL_KEY}.")
        # No need to check is_model_available(fallback_choice_letter) here, free is assumed available
        return FALLBACK_MODEL_KEY

    # If only one model is allowed, use it (covers case where only free is left implicitly)
    if len(allowed) == 1:
        logging.info(f"Router: Only one model allowed ('{allowed[0]}'), selecting it directly.")
        return allowed[0]

    # Prepare history context (Unchanged)
    history_segments = []; current_chars = 0
    relevant_history_for_router = chat_history[:-1] if chat_history else []
    for msg in reversed(relevant_history_for_router):
        role = msg.get("role", "assistant").capitalize(); content = msg.get("content", "")
        if not isinstance(content, str): content = str(content)
        segment = f"{role}: {content}\n";
        if current_chars + len(segment) > MAX_HISTORY_CHARS_FOR_ROUTER: break
        history_segments.append(segment); current_chars += len(segment)
    history_context_str = "".join(reversed(history_segments)).strip() or "No prior conversation history."

    # --- Build the REVISED V4 system prompt ---
    system_prompt_parts = [
        "You are an expert AI model routing assistant. Your task is to select the *single most appropriate and cost-effective* model letter from the 'Available Models' list to handle the 'Latest User Query'.",
        "Core Principles:",
        "1. **Assess Need: Crisis -> Search -> General:** First, check for CRITICAL SENSITIVITY (self-harm, crisis). Second, check if WEB SEARCH is explicitly or implicitly required. Third, assess general capability needed.",
        "2. **Prioritize Safety (CRISIS):** If the query mentions/implies self-harm, suicide, immediate danger, or severe crisis, **YOU MUST CHOOSE** a highly capable, safe model. Order: **E > A > C**. If none available, use B ONLY as a last resort. **IGNORE search/cost** for crisis. Models F, D, G, H, I are **NOT SUITABLE**.",
        "3. **Prioritize Functionality (SEARCH):** If the query requires up-to-date info, current events, or specific web lookups (and is NOT a crisis), **YOU MUST CHOOSE** a Search model (G, H, I). Select based on complexity:",
        "   - **Deep Research ('I'):** For explicit multi-step investigations, comprehensive reports needing broad search. Use if 'investigate', 'deep dive', 'comprehensive report on' are used. (Available: {})".format("Yes" if "I" in allowed else "No"),
        "   - **Advanced Search + Reasoning ('H'):** For complex questions needing search AND reasoning on results, comparisons based on current data. (Available: {})".format("Yes" if "H" in allowed else "No"),
        "   - **Simple Search ('G'):** For basic lookups (news, weather, definitions, simple facts). (Available: {})".format("Yes" if "G" in allowed else "No"),
        "   - **Search Fallback:** If the ideal search model (I/H/G) is unavailable, choose the next best *available* search model (I -> H -> G). If NO search models (G, H, I) are available, select the best *general* model (A, B, E, C...) but acknowledge search wasn't performed.",
        "4. **Maximize Cost-Effectiveness (GENERAL / NON-CRISIS / NON-SEARCH):** If NOT crisis and NOT search, choose the ***absolute cheapest*** available general model (F, D, B, A, E, C) that meets the capability needed. General Cost Order: **F < G < D < B < A ≈ H ≈ I < E < C**.",
        "5. **Consider History:** Use 'Recent Conversation History' for context."
    ]
    system_prompt_parts.append("\nAvailable Models (Cost Order: F < G < D < B < A ≈ H ≈ I < E < C - may be overridden by crisis/search):")

    # Use the updated guidance dict including search models G, H, I
    for k_model_key in sorted(allowed): # Display available models alphabetically for clarity
        description = ROUTER_MODEL_GUIDANCE_SENSITIVE.get(k_model_key, f"(Model {k_model_key} - Description not found).")
        system_prompt_parts.append(f"- {k_model_key}: {description}")

    # REVISED Decision Process V4
    system_prompt_parts.append("\nDecision Process for 'Latest User Query':")
    system_prompt_parts.append("1. **CRISIS CHECK:** Does the query mention/imply self-harm, suicide, immediate danger, severe crisis?")
    system_prompt_parts.append("   - **IF YES (CRISIS):**")
    system_prompt_parts.append("     - Is 'E' available? CHOOSE 'E'.")
    system_prompt_parts.append("     - Else, is 'A' available? CHOOSE 'A'.")
    system_prompt_parts.append("     - Else, is 'C' available? CHOOSE 'C'.")
    system_prompt_parts.append("     - Else, is 'B' available? CHOOSE 'B' (last resort).")
    system_prompt_parts.append("     - If none of E,A,C,B available, STOP (system handles fallback).")
    system_prompt_parts.append("   - **IF NO (NOT CRISIS):** Proceed to step 2.")
    system_prompt_parts.append("2. **SEARCH CHECK:** Does the query require web search / current info (latest news, specific lookup, current status)?")
    system_prompt_parts.append("   - **IF YES (SEARCH NEEDED):**")
    system_prompt_parts.append("     - Does it need deep multi-step research ('I' territory)? Is 'I' available? CHOOSE 'I'.")
    system_prompt_parts.append("     - Else, does it need complex search + reasoning ('H' territory)? Is 'H' available? CHOOSE 'H'.")
    system_prompt_parts.append("     - Else, is it a simple lookup ('G' territory)? Is 'G' available? CHOOSE 'G'.")
    system_prompt_parts.append("     - **Search Fallback:** If chosen search model (I/H/G) unavailable, pick the next best *available* search model (I->H->G).")
    system_prompt_parts.append("     - **No Search Models Available:** If G, H, AND I are *all* unavailable, proceed to Step 3 (Standard Routing) to pick the best *general* model, understanding search cannot be performed.")
    system_prompt_parts.append("   - **IF NO (SEARCH NOT NEEDED):** Proceed to step 3.")
    system_prompt_parts.append("3. **Standard Routing (Non-Crisis, Non-Search):** Select the CHEAPEST sufficient model from available F, D, B, A, E, C.")
    system_prompt_parts.append("   - Is 'F' sufficient (simple, low-stakes)? (Available: {}) CHOOSE 'F'.".format("Yes" if "F" in allowed else "No"))
    system_prompt_parts.append("   - Else, is 'D' sufficient (technical/factual)? (Available: {}) CHOOSE 'D'.".format("Yes" if "D" in allowed else "No"))
    system_prompt_parts.append("   - Else, is 'B' sufficient (general moderate tasks)? (Available: {}) CHOOSE 'B'.".format("Yes" if "B" in allowed else "No"))
    system_prompt_parts.append("   - Else, is 'A' sufficient (complex reasoning)? (Available: {}) CHOOSE 'A'.".format("Yes" if "A" in allowed else "No"))
    system_prompt_parts.append("   - Else, is 'E' sufficient (novel creative)? (Available: {}) CHOOSE 'E'.".format("Yes" if "E" in allowed else "No"))
    system_prompt_parts.append("   - Else, consider 'C' (peak polish)? (Available: {}) CHOOSE 'C'.".format("Yes" if "C" in allowed else "No"))
    system_prompt_parts.append("   - Select the *first* sufficient model found in the F->D->B->A->E->C order.")

    system_prompt_parts.append("\nRecent Conversation History (Context):")
    system_prompt_parts.append(history_context_str)
    system_prompt_parts.append(f"\nAvailable Model Letters: {', '.join(sorted(allowed))}")
    system_prompt_parts.append("\nINSTRUCTION: Analyze the 'Latest User Query' (provided in the user role message) using the 3-step process (Crisis -> Search -> Standard). Respond with ONLY the single capital letter of your chosen model. NO EXPLANATION.")

    final_system_message = "\n".join(system_prompt_parts)
    # Use DEBUG level to see the full prompt if needed
    logging.debug(f"Router System Prompt V4:\n{final_system_message}")

    router_messages = [{"role": "system", "content": final_system_message}, {"role": "user", "content": user_msg}]
    payload_r = {"model": ROUTER_MODEL_ID, "messages": router_messages, "max_tokens": 10, "temperature": 0.1} # Lower temp slightly
    logging.debug(f"Router Payload: {json.dumps(payload_r, indent=2)}")

    try:
        r = api_post(payload_r)
        choice_data = r.json()
        logging.debug(f"Router Full Response JSON: {json.dumps(choice_data, indent=2)}")
        raw_text_response = choice_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()
        logging.info(f"Router raw text response: '{raw_text_response}' for query: '{user_msg[:100]}...'")

        chosen_model_letter = _extract_letter(raw_text_response, allowed)
        if chosen_model_letter:
            logging.info(f"Router selected model '{chosen_model_letter}' from response '{raw_text_response}'")

        if chosen_model_letter:
            # Final check: if the chosen model is NOT available (e.g., quota hit *during* routing), fall back.
            if not is_model_available(chosen_model_letter):
                 logging.warning(f"Router chose '{chosen_model_letter}' but it became unavailable. Using free fallback: {FALLBACK_MODEL_KEY}.")
                 return FALLBACK_MODEL_KEY
            else:
                 return chosen_model_letter
        else:
            logging.warning(f"Router returned ('{raw_text_response}') - no allowed letter found or response invalid. Using defined fallback logic -> '{fallback_choice_letter}' or free.")
            # Use the pre-calculated fallback_choice_letter, checking its availability
            if is_model_available(fallback_choice_letter):
                return fallback_choice_letter
            else:
                logging.warning(f"Fallback choice '{fallback_choice_letter}' also unavailable. Using free fallback: {FALLBACK_MODEL_KEY}.")
                return FALLBACK_MODEL_KEY

    # --- Error handling (same as before) ---
    except ValueError as ve: logging.error(f"Router call failed due to invalid API key or config: {ve}"); st.session_state.api_key_auth_failed = True; return None
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else 'N/A'; err_text = e.response.text if e.response else 'No response text'
        logging.error(f"Router HTTPError {status_code}: {err_text}")
        if status_code == 401: st.session_state.api_key_auth_failed = True; return None # Special return for auth failure
        # For other HTTP errors, proceed to fallback below
    except requests.exceptions.RequestException as e: logging.error(f"Router Network Error: {e}")
    except (KeyError, IndexError, AttributeError, json.JSONDecodeError) as je:
        response_text_for_log = r.text if 'r' in locals() and hasattr(r, 'text') else "N/A"
        logging.error(f"Router JSON/structure error: {je}. Raw: {response_text_for_log}")
    except Exception as e: logging.exception(f"Router unexpected error: {e}")

    # Fallback if any error occurred above (except auth error which returns None)
    logging.warning(f"Router failed. Falling back based on availability -> '{fallback_choice_letter}' or free.")
    if is_model_available(fallback_choice_letter):
        return fallback_choice_letter
    else:
        logging.warning(f"Fallback choice '{fallback_choice_letter}' also unavailable. Using free fallback: {FALLBACK_MODEL_KEY}.")
        return FALLBACK_MODEL_KEY

# --------------------- Credits Endpoint (Unchanged) -----------------------
def get_credits():
    active_api_key = st.session_state.get("openrouter_api_key")
    if not is_api_key_valid(active_api_key):
        logging.warning("get_credits: API Key is not syntactically valid or not set."); return None, None, None
    try:
        r = requests.get(f"{OPENROUTER_API_BASE}/credits", headers={"Authorization": f"Bearer {active_api_key}"}, timeout=15)
        r.raise_for_status()
        d = r.json().get("data")
        if d and "limit" in d and "usage" in d: # Standard OpenRouter v1 format
            total_credits = float(d["limit"])
            total_usage = float(d["usage"])
            remaining_credits = total_credits - total_usage
            st.session_state.api_key_auth_failed = False
            return total_credits, total_usage, remaining_credits
        # Add checks for other potential structures if needed
        else:
            logging.warning(f"Could not parse /credits response structure: {r.json()}")
            return None, None, None

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else 'N/A'; err_text = e.response.text if e.response else 'No response text'
        if status_code == 401: st.session_state.api_key_auth_failed = True; logging.warning(f"Could not fetch /credits: HTTP {status_code} Unauthorized. {err_text}")
        else: logging.warning(f"Could not fetch /credits: HTTP {status_code}. {err_text}")
        return None, None, None
    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logging.warning(f"Could not fetch /credits due to network/parsing error: {e}"); return None, None, None


# ------------------------- UI Styling (Unchanged - CSS handles general elements) --------------------------
def load_custom_css():
    """Inject custom CSS. Uses dark or light theme based on session state."""
    dark = st.session_state.get("dark_mode", False)
    if dark:
        root_colors = """
            --app-bg-color: #121212;
            --app-secondary-bg-color: #1f1f1f;
            --app-text-color: #E4E4E4;
            --app-text-secondary-color: #A1A1A1;
            --app-primary-color: #0d6efd;
            --app-primary-hover-color: #0b5ed7;
            --app-divider-color: #343a40;
            --app-border-color: #495057;
            --app-success-color: #198754;
            --app-warning-color: #FFC107;
            --app-danger-color: #DC3545;
        """
    else:
        root_colors = """
            --app-bg-color: #F8F9FA;
            --app-secondary-bg-color: #FFFFFF;
            --app-text-color: #212529;
            --app-text-secondary-color: #6C757D;
            --app-primary-color: #007BFF;
            --app-primary-hover-color: #0056b3;
            --app-divider-color: #DEE2E6;
            --app-border-color: #CED4DA;
            --app-success-color: #28A745;
            --app-warning-color: #FFC107;
            --app-danger-color: #DC3545;
        """

    css_template = """
    <style>
        :root {
            /* Core Colors */
PLACEHOLDER_ROOT_COLORS
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
            max-width: 1200px; /* Adjust max width if needed */
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: var(--app-secondary-bg-color);
            border-right: 1px solid var(--app-divider-color);
            padding: var(--spacing-md);
            width: 300px !important; /* Slightly wider sidebar maybe? */
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
        .sidebar-title-container h1 {
            background: linear-gradient(90deg, var(--app-primary-color), var(--app-primary-hover-color));
            -webkit-background-clip: text;
            color: transparent;
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
            background-color: color-mix(in srgb, var(--app-bg-color) 50%, var(--app-secondary-bg-color) 50%);
            border-bottom-left-radius: var(--border-radius-md); border-bottom-right-radius: var(--border-radius-md);
        }
        /* Specific styling for compact quota gauges */
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
            flex-grow: 1; /* Allow items to grow slightly */
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
            max-width: 85%; /* Increase max width slightly */
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
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            padding-top: 0.1rem; padding-bottom: 0.1rem;
        }
        [data-testid="stChatMessage"] code { /* Improve code block styling */
           font-size: 0.85em;
           padding: 0.2em 0.4em;
           margin: 0;
           background-color: color-mix(in srgb, var(--app-text-color) 5%, transparent);
           border-radius: var(--border-radius-sm);
        }
        [data-testid="stChatMessage"] pre > code { /* Full code blocks */
           background-color: initial; /* Let streamlit handle block background */
           padding: 0;
           border-radius: 0;
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
    css = css_template.replace("PLACEHOLDER_ROOT_COLORS", root_colors)
    st.markdown(css, unsafe_allow_html=True)

# ----------------- State Initialization (API Key & Theme) -------------------
app_conf = _load_app_config()
if "openrouter_api_key" not in st.session_state:
    st.session_state.openrouter_api_key = app_conf.get("openrouter_api_key", None)
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = app_conf.get("dark_mode", False)
if "api_key_auth_failed" not in st.session_state: st.session_state.api_key_auth_failed = False
api_key_is_syntactically_valid = is_api_key_valid(st.session_state.get("openrouter_api_key"))
app_requires_api_key_setup = (
    not st.session_state.get("openrouter_api_key") or
    not api_key_is_syntactically_valid or
    st.session_state.get("api_key_auth_failed", False)
)

# -------------------- Main Application Rendering -------------------
if app_requires_api_key_setup:
    # --- API Key Setup Page (Unchanged) ---
    st.set_page_config(page_title="OpenRouter API Key Setup", layout="centered")
    load_custom_css()
    st.title("🔒 OpenRouter API Key Required")
    st.markdown("---", unsafe_allow_html=True)
    if st.session_state.get("api_key_auth_failed"): st.error("API Key Authentication Failed. Please verify your key on OpenRouter.ai and re-enter.")
    elif not api_key_is_syntactically_valid and st.session_state.get("openrouter_api_key") is not None: st.error("The configured API Key has an invalid format. It must start with `sk-or-`.")
    elif not st.session_state.get("openrouter_api_key"): st.info("Please configure your OpenRouter API Key to use the application.")
    else: st.info("API Key configuration required.")
    st.markdown( "You can get a key from [OpenRouter.ai Keys](https://openrouter.ai/keys). Enter it below to continue." )
    new_key_input_val = st.text_input("Enter OpenRouter API Key", type="password", key="api_key_setup_input", value="", placeholder="sk-or-...")
    if st.button("Save and Validate API Key", key="save_api_key_setup_button", use_container_width=True, type="primary"):
        if is_api_key_valid(new_key_input_val):
            st.session_state.openrouter_api_key = new_key_input_val
            _save_app_config(api_key_value=new_key_input_val)
            st.session_state.api_key_auth_failed = False
            with st.spinner("Validating API Key..."): fetched_credits_data = get_credits()
            if st.session_state.get("api_key_auth_failed"): st.error("Authentication failed with the provided API Key. Please check the key and try again.")
            elif fetched_credits_data == (None, None, None): st.error("Could not validate API Key. Network issue or OpenRouter API problem? Key saved, but functionality may be affected."); time.sleep(1.5); st.rerun()
            else:
                st.success("API Key saved and validated! Initializing application...")
                if "credits" not in st.session_state: st.session_state.credits = {}
                st.session_state.credits["total"], st.session_state.credits["used"], st.session_state.credits["remaining"] = fetched_credits_data
                st.session_state.credits_ts = time.time()
                time.sleep(1.0); st.rerun()
        elif not new_key_input_val: st.warning("API Key field cannot be empty.")
        else: st.error("Invalid API key format. It must start with 'sk-or-'.")

    dark_setup = st.checkbox("Enable dark mode", value=st.session_state.get("dark_mode", False), key="dark_mode_setup")
    if dark_setup != st.session_state.get("dark_mode"):
        st.session_state.dark_mode = dark_setup
        _save_app_config(dark_mode_value=dark_setup)
        st.rerun()

    st.markdown("---", unsafe_allow_html=True); st.caption("Your API key is stored locally in `app_config.json`.")

# --- Main App Logic (Executes only if API key setup is NOT required) ---
else:
    st.set_page_config(page_title="OpenRouter Chat", layout="wide", initial_sidebar_state="expanded")
    load_custom_css()

    # --- Session State Initialization (Unchanged) ---
    if "settings_panel_open" not in st.session_state: st.session_state.settings_panel_open = False
    if "credits" not in st.session_state: st.session_state.credits = {"total": None, "used": None, "remaining": None}; st.session_state.credits_ts = 0
    needs_save_session = False
    if "sid" not in st.session_state: st.session_state.sid = _new_sid(); needs_save_session = True; logging.info(f"Initialized new session ID: {st.session_state.sid}")
    elif st.session_state.sid not in sessions: logging.warning(f"Session ID {st.session_state.sid} not found. Creating new chat."); st.session_state.sid = _new_sid(); needs_save_session = True
    if needs_save_session: _save(SESS_FILE, sessions)

    # --- Credit Refresh Logic (Unchanged) ---
    credits_are_stale = time.time() - st.session_state.get("credits_ts", 0) > 3600
    credits_never_fetched = st.session_state.get("credits_ts", 0) == 0
    credits_are_none = any(st.session_state.credits.get(k) is None for k in ["total", "used", "remaining"])
    if credits_are_stale or credits_never_fetched or credits_are_none:
        logging.info(f"Refreshing credits (Stale: {credits_are_stale}, Never Fetched: {credits_never_fetched}, Are None: {credits_are_none}).")
        credits_data = get_credits()
        if st.session_state.get("api_key_auth_failed"): logging.error("API Key auth failed during scheduled credit refresh. Credits remain unchanged."); st.session_state.credits_ts = time.time() # Update timestamp even on auth failure
        elif credits_data != (None, None, None): st.session_state.credits["total"], st.session_state.credits["used"], st.session_state.credits["remaining"] = credits_data; st.session_state.credits_ts = time.time(); logging.info("Credits refreshed successfully.")
        else: logging.warning("Scheduled credit refresh failed (non-auth). Credits remain unchanged."); st.session_state.credits_ts = time.time() # Update timestamp on other failures
        # Ensure credits dict structure exists even if fetch failed
        if "credits" not in st.session_state or not isinstance(st.session_state.credits, dict): st.session_state.credits = {"total": None, "used": None, "remaining": None}
        for k in ["total", "used", "remaining"]: st.session_state.credits.setdefault(k, None)


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
            elif current_api_key_in_panel: key_display = "Current key: `sk-or-...`"
            else: key_display = "Current key: Not set"
            st.caption(key_display)
            if st.session_state.get("api_key_auth_failed"): st.error("Current API Key failed authentication.")

            new_key_input_sidebar = st.text_input("New OpenRouter API Key (optional)", type="password", key="api_key_sidebar_input", placeholder="sk-or-...")
            if st.button("Save New API Key", key="save_api_key_sidebar_button", use_container_width=True):
                if is_api_key_valid(new_key_input_sidebar):
                    st.session_state.openrouter_api_key = new_key_input_sidebar
                    _save_app_config(api_key_value=new_key_input_sidebar)
                    st.session_state.api_key_auth_failed = False
                    with st.spinner("Validating new API key..."): credits_data = get_credits()
                    if st.session_state.get("api_key_auth_failed"): st.error("New API Key failed authentication.")
                    elif credits_data == (None,None,None): st.warning("Could not validate new API key (network/API issue?). Saved, but functionality may be affected.")
                    else:
                        st.success("New API Key saved and validated!"); st.session_state.credits["total"],st.session_state.credits["used"],st.session_state.credits["remaining"] = credits_data; st.session_state.credits_ts = time.time()
                    time.sleep(0.8); st.rerun()
                elif not new_key_input_sidebar: st.warning("API Key field empty. No changes made.")
                else: st.error("Invalid API key format. Must start with 'sk-or-'.")

            dark_mode_toggle = st.checkbox("Enable dark mode", value=st.session_state.get("dark_mode", False), key="dark_mode_toggle")
            if dark_mode_toggle != st.session_state.get("dark_mode"):
                st.session_state.dark_mode = dark_mode_toggle
                _save_app_config(dark_mode_value=dark_mode_toggle)
                st.rerun()

            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("📊 Detailed Model Quotas")
            _ensure_quota_data_is_current() # Ensure data loaded

            # Iterate through ALL models defined in NEW_PLAN_CONFIG for detailed view
            for m_key_loop in sorted(NEW_PLAN_CONFIG.keys()):
                # Skip if model isn't fully defined for display (needs MAP, EMOJI, DESCRIPTIONS)
                if m_key_loop not in MODEL_MAP or m_key_loop not in EMOJI or m_key_loop not in MODEL_DESCRIPTIONS:
                    logging.warning(f"Skipping detailed quota display for incompletely defined model key: {m_key_loop}")
                    continue

                stats = get_quota_usage_and_limits(m_key_loop)
                if not stats:
                    st.markdown(f"**{EMOJI.get(m_key_loop, '')} {m_key_loop} ({MODEL_MAP.get(m_key_loop, 'N/A').split('/')[-1]})**: Could not retrieve quota details.")
                    continue

                # Extract short name robustly from MODEL_DESCRIPTIONS
                model_desc_full = MODEL_DESCRIPTIONS.get(m_key_loop, "")
                try:
                    model_short_name = model_desc_full.split('(')[1].split(')')[0] if '(' in model_desc_full and ')' in model_desc_full else MODEL_MAP.get(m_key_loop, "Unknown").split('/')[-1]
                except IndexError:
                    model_short_name = MODEL_MAP.get(m_key_loop, "Unknown").split('/')[-1]

                model_name_display = f"{EMOJI.get(m_key_loop, '')} <span class='detailed-quota-modelname'>{m_key_loop} ({model_short_name})</span>"
                st.markdown(f"{model_name_display}", unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                # Helper to format limit display
                def format_limit(limit_val): return format_token_count(limit_val) if limit_val != float('inf') and limit_val > 0 else '∞'

                with col1:
                    st.markdown(f"""
                    <div class="detailed-quota-block">
                    <ul>
                        <li><b>Daily Msgs:</b> {stats['used_daily_msg']}/{format_limit(stats['limit_daily_msg'])}</li>
                        <li><b>Daily In Tok:</b> {format_token_count(stats['used_daily_in_tokens'])}/{format_limit(stats['limit_daily_in_tokens'])}</li>
                        <li><b>Daily Out Tok:</b> {format_token_count(stats['used_daily_out_tokens'])}/{format_limit(stats['limit_daily_out_tokens'])}</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="detailed-quota-block">
                    <ul>
                        <li><b>Monthly Msgs:</b> {stats['used_monthly_msg']}/{format_limit(stats['limit_monthly_msg'])}</li>
                        <li><b>Monthly In Tok:</b> {format_token_count(stats['used_monthly_in_tokens'])}/{format_limit(stats['limit_monthly_in_tokens'])}</li>
                        <li><b>Monthly Out Tok:</b> {format_token_count(stats['used_monthly_out_tokens'])}/{format_limit(stats['limit_monthly_out_tokens'])}</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)

                # Display 3-hour limit details only for Model A if configured and finite
                if m_key_loop == "A" and stats["limit_3hr_msg"] != float('inf'):
                    time_until_next_msg_str = ""
                    active_model_a_calls = sorted(_g_quota_data.get(MODEL_A_3H_CALLS_KEY, []))
                    if len(active_model_a_calls) >= stats['limit_3hr_msg']:
                         if active_model_a_calls:
                             oldest_blocking_call_idx = max(0, len(active_model_a_calls) - int(stats['limit_3hr_msg']))
                             oldest_blocking_call_ts = active_model_a_calls[oldest_blocking_call_idx]
                             # Ensure plan A config exists and has window value before calculating expiry
                             if "A" in NEW_PLAN_CONFIG and len(NEW_PLAN_CONFIG["A"]) > 7 and NEW_PLAN_CONFIG["A"][7] > 0:
                                 expiry_time = oldest_blocking_call_ts + NEW_PLAN_CONFIG["A"][7]
                                 time_remaining_seconds = expiry_time - time.time()
                                 if time_remaining_seconds > 0:
                                    mins, secs = divmod(int(time_remaining_seconds), 60)
                                    hrs, mins_rem = divmod(mins, 60)
                                    if hrs > 0: time_until_next_msg_str = f" (Next in {hrs}h {mins_rem}m)"
                                    elif mins_rem > 0: time_until_next_msg_str = f" (Next in {mins_rem}m {secs}s)"
                                    else: time_until_next_msg_str = f" (Next in {secs}s)"

                    st.markdown(f"""
                    <div class="detailed-quota-block" style="margin-top: -0.5rem; margin-left:0.1rem;">
                    <ul><li><b>3-Hour Msgs:</b> {stats['used_3hr_msg']}/{int(stats['limit_3hr_msg'])}{time_until_next_msg_str}</li></ul>
                    </div>""", unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True) # End settings-panel

        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

        # --- Sidebar Header (Unchanged) ---
        st.markdown("<div class='sidebar-title-container'>", unsafe_allow_html=True)
        logo_title_cols = st.columns([1, 5], gap="small")
        with logo_title_cols[0]: st.image("https://avatars.githubusercontent.com/u/130328222?s=200&v=4", width=40)
        with logo_title_cols[1]: st.title("OpenRouter Chat")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

        # --- Daily Quota Gauges ---
        with st.expander("⚡ DAILY MODEL QUOTAS", expanded=True):
            # Display gauges ONLY for models defined in ALL required dicts (Config, Map, Emoji)
            active_model_keys_for_display = sorted([k for k in NEW_PLAN_CONFIG.keys() if k in MODEL_MAP and k in EMOJI])
            if not active_model_keys_for_display: st.caption("No fully configured quota-tracked models.")
            else:
                _ensure_quota_data_is_current()
                num_models = len(active_model_keys_for_display)
                # Dynamic columns: more models -> more columns, up to a max (e.g., 7)
                num_cols = min(num_models, 7)
                quota_cols = st.columns(num_cols)

                for i, m_key in enumerate(active_model_keys_for_display):
                    with quota_cols[i % num_cols]:
                        stats = get_quota_usage_and_limits(m_key)
                        if not stats: left_d_msgs, lim_d_msgs = 0, 0
                        else:
                            # Use helper function for remaining messages
                            left_d_msgs = get_remaining_daily_messages(m_key)
                            lim_d_msgs = stats["limit_daily_msg"]

                        if lim_d_msgs == float('inf') or lim_d_msgs <= 0: # Handle infinite or zero limits
                            pct_float, fill_width_val, left_display = 1.0, 100, "∞"
                            bar_color = "var(--app-success-color)" # Show infinite as full/green
                        else: # Handle finite limits
                            pct_float = max(0.0, min(1.0, left_d_msgs / lim_d_msgs))
                            fill_width_val = int(pct_float * 100)
                            left_display = str(int(left_d_msgs)) # Show as integer
                            # Color logic
                            if pct_float > 0.5: bar_color = "var(--app-success-color)"
                            elif pct_float > 0.15: bar_color = "var(--app-warning-color)" # Warning threshold maybe lower
                            else: bar_color = "var(--app-danger-color)"

                        emoji_char = EMOJI.get(m_key, "❔")
                        limit_display_tt = '∞' if lim_d_msgs == float('inf') or lim_d_msgs <= 0 else str(int(lim_d_msgs))
                        tooltip_text = f"{left_display} / {limit_display_tt} Daily Msgs Left"
                        st.markdown(f"""<div class="compact-quota-item" title="{tooltip_text}">
                                            <div class="cq-info">{emoji_char} <b>{m_key}</b></div>
                                            <div class="cq-bar-track"><div class="cq-bar-fill" style="width: {fill_width_val}%; background-color: {bar_color};"></div></div>
                                            <div class="cq-value" style="color: {bar_color};">{left_display}</div>
                                        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

        # --- New Chat Button (Unchanged logic) ---
        current_session_is_truly_blank = (st.session_state.sid in sessions and
                                          sessions[st.session_state.sid].get("title") == "New chat" and
                                          not sessions[st.session_state.sid].get("messages"))
        if st.button("➕ New chat", key="new_chat_button_top", use_container_width=True, disabled=current_session_is_truly_blank):
            st.session_state.sid = _new_sid()
            _save(SESS_FILE, sessions)
            st.rerun()

        # --- Chat History List (Unchanged logic) ---
        st.subheader("Chats")
        valid_sids = [s for s in sessions.keys() if isinstance(s, str) and s.isdigit()]
        sorted_sids = sorted(valid_sids, key=lambda s: int(s), reverse=True)
        for sid_key in sorted_sids:
            if sid_key not in sessions: continue
            session_data = sessions[sid_key]; title = session_data.get("title", f"Chat {sid_key}")
            display_title = (title[:30] + "…") if len(title) > 30 else title
            is_active_chat = (st.session_state.sid == sid_key)
            if st.button(display_title, key=f"session_button_{sid_key}", use_container_width=True, disabled=is_active_chat):
                if not is_active_chat:
                    current_session_data = sessions.get(st.session_state.sid, {})
                    current_session_was_blank = (current_session_data.get("title") == "New chat" and not current_session_data.get("messages"))
                    if not current_session_was_blank: _delete_unused_blank_sessions(keep_sid=sid_key)
                    st.session_state.sid = sid_key
                    _save(SESS_FILE, sessions)
                    st.rerun()

        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

        # --- Model Info ---
        st.subheader("Model Info & Costs")
        st.caption(f"Router: {ROUTER_MODEL_ID.split('/')[-1]}")
        # Define the intended cost order explicitly for display
        cost_order_display = ["F", "G", "D", "B", "A", "H", "I", "E", "C"]
        cost_order_title = " < ".join(cost_order_display)
        with st.expander(f"Cost Order: {cost_order_title}", expanded=False):
            # Iterate through models in the defined cost order
            for k_model in cost_order_display:
                 # Check if model is fully defined for display
                 if k_model not in MODEL_MAP or k_model not in MODEL_DESCRIPTIONS or k_model not in EMOJI: continue

                 desc_full = MODEL_DESCRIPTIONS.get(k_model, MODEL_MAP.get(k_model, "N/A"))
                 try:
                     desc_parts = desc_full.split("(")
                     main_desc = desc_parts[0].strip()
                     model_name_in_desc = desc_parts[1].split(")")[0] if len(desc_parts) > 1 and ')' in desc_parts[1] else MODEL_MAP.get(k_model, "N/A").split('/')[-1]
                 except IndexError:
                     main_desc = desc_full
                     model_name_in_desc = MODEL_MAP.get(k_model, "N/A").split('/')[-1]

                 max_tok = MAX_TOKENS.get(k_model, 0)
                 emoji_char = EMOJI.get(k_model, '')
                 st.markdown(f"**{emoji_char} {k_model}**: {main_desc} ({model_name_in_desc}) <br><small style='color:var(--app-text-secondary-color);'>Max Output: {max_tok:,} tokens</small>", unsafe_allow_html=True)

            # Add Fallback model info at the end
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(f"**{FALLBACK_MODEL_KEY}**: {FALLBACK_MODEL_EMOJI} {FALLBACK_MODEL_ID.split('/')[-1]} <br><small style='color:var(--app-text-secondary-color);'>Max Output: {FALLBACK_MODEL_MAX_TOKENS:,} tokens (Free; used when quotas exhausted or routing fails)</small>", unsafe_allow_html=True)

        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

        # --- Account Credits (Unchanged logic) ---
        with st.expander("Account stats (credits)", expanded=False):
            if st.button("Refresh Credits", key="refresh_credits_button_sidebar"):
                 with st.spinner("Refreshing credits..."): credits_data = get_credits()
                 if st.session_state.get("api_key_auth_failed"): st.error("API Key authentication failed. Cannot refresh credits.")
                 elif credits_data != (None,None,None):
                     st.session_state.credits["total"],st.session_state.credits["used"],st.session_state.credits["remaining"] = credits_data; st.session_state.credits_ts = time.time(); st.success("Credits refreshed!"); st.rerun()
                 else: st.warning("Could not refresh credits (network/API issue?).")

            tot, used, rem = st.session_state.credits.get("total"), st.session_state.credits.get("used"), st.session_state.credits.get("remaining")
            if st.session_state.get("api_key_auth_failed"): st.warning("Cannot display credits. API Key failed authentication.")
            elif tot is None or used is None or rem is None: st.warning("Could not fetch/display credits.")
            else:
                 try: rem_f = f"${float(rem):.2f} cr"
                 except (ValueError, TypeError): rem_f = "N/A"
                 try: used_f = f"${float(used):.2f} cr"
                 except (ValueError, TypeError): used_f = "N/A"
                 st.markdown(f"**Remaining:** {rem_f}<br>**Used:** {used_f}", unsafe_allow_html=True)
            ts = st.session_state.get("credits_ts", 0); last_updated_str = datetime.fromtimestamp(ts, TZ).strftime('%-d %b, %H:%M') if ts else "Never"; st.caption(f"Last updated: {last_updated_str}")

    # ---- Main chat area ----
    if st.session_state.sid not in sessions:
        logging.error(f"CRITICAL: Current SID {st.session_state.sid} missing. Resetting."); st.session_state.sid = _new_sid(); _save(SESS_FILE, sessions); st.rerun()
    current_sid = st.session_state.sid
    if "messages" not in sessions[current_sid]: sessions[current_sid]["messages"] = []; logging.warning(f"Initialized missing 'messages' for session {current_sid}."); _save(SESS_FILE, sessions)
    chat_history = sessions[current_sid]["messages"]

    # --- Display Existing Chat Messages ---
    for msg_idx, msg in enumerate(chat_history):
        role = msg.get("role", "assistant")
        avatar_char = None
        if role == "user": avatar_char = "👤"
        elif role == "assistant":
            m_key = msg.get("model") # Get model key used
            if m_key == FALLBACK_MODEL_KEY: avatar_char = FALLBACK_MODEL_EMOJI
            elif m_key in EMOJI: avatar_char = EMOJI[m_key] # Use specific model emoji
            else: avatar_char = "🤖" # Default if model unknown
        else: role="assistant"; avatar_char = "⚙️" # Display others as assistant
        with st.chat_message(role, avatar=avatar_char): st.markdown(msg.get("content", "*empty message*"))

    # --- Chat Input Logic ---
    if prompt := st.chat_input("Ask anything… (use 'search', 'latest', 'investigate' for web access)", key=f"chat_input_{current_sid}"):
        # Append user message and display
        user_message = {"role":"user","content":prompt}
        chat_history.append(user_message)
        with st.chat_message("user", avatar="👤"): st.markdown(prompt)
        _save(SESS_FILE, sessions) # Save user msg

        # Check API key validity before routing
        if st.session_state.get("api_key_auth_failed") or not is_api_key_valid(st.session_state.get("openrouter_api_key")):
            st.error("OpenRouter API Key is invalid or failed authentication. Please fix in ⚙️ Settings.")
            st.stop()

        # --- Model Selection Logic ---
        routing_start_time = time.time()
        with st.spinner("Selecting best model (Crisis? Search? General?)..."):
            _ensure_quota_data_is_current() # Refresh quotas

            logging.info("--- Checking Model Availability Before Routing ---")
            # Get all models from config that are fully defined and check availability
            all_possible_models = [k for k in NEW_PLAN_CONFIG.keys() if k in MODEL_MAP and k in EMOJI]
            allowed_models_for_router = []
            availability_log = []
            for k_map in sorted(all_possible_models): # Check all configured models
                 available = is_model_available(k_map)
                 availability_log.append(f"Model {k_map} ({MODEL_MAP.get(k_map,'?').split('/')[-1]}): Available = {available}")
                 if available: allowed_models_for_router.append(k_map)
            # Always add the free fallback as an "allowed" option for the router's internal logic,
            # even though it won't be passed explicitly to the API call selection prompt unless needed.
            # The route_choice function handles the fallback logic internally.
            # allowed_models_for_router.append(FALLBACK_MODEL_KEY) # Don't add fallback here, router handles it.
            logging.info("\n".join(availability_log))
            logging.info(f"Final allowed models passed to router function: {allowed_models_for_router}")

            use_fallback = False
            chosen_model_key = None
            model_id_to_use = None
            max_tokens_api = FALLBACK_MODEL_MAX_TOKENS
            avatar_resp = "🤖" # Default avatar

            # Call the updated router function
            routed_key_letter = route_choice(prompt, allowed_models_for_router, chat_history)
            routing_end_time = time.time()
            logging.info(f"Routing took {routing_end_time - routing_start_time:.2f} seconds. Router decided: '{routed_key_letter}'")

            # Handle router failure due to auth (route_choice returns None)
            if routed_key_letter is None and st.session_state.get("api_key_auth_failed"):
                 st.error("API Authentication failed during model routing. Please check the key in ⚙️ Settings.")
                 _save(SESS_FILE, sessions) # Save chat history up to this point
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
                # Final check: is the chosen model *still* available? (Quota might update)
                # This check is technically redundant if route_choice already did it, but adds safety.
                if is_model_available(routed_key_letter):
                    chosen_model_key = routed_key_letter
                    model_id_to_use = MODEL_MAP[chosen_model_key]
                    max_tokens_api = MAX_TOKENS.get(chosen_model_key, FALLBACK_MODEL_MAX_TOKENS) # Use specific max_tokens
                    avatar_resp = EMOJI.get(chosen_model_key, "🤖") # Use specific emoji
                    logging.info(f"Using router-selected model: {chosen_model_key} ({model_id_to_use})")
                else:
                    logging.warning(f"Router chose '{routed_key_letter}', but it's no longer available. Using free fallback {FALLBACK_MODEL_ID}.")
                    use_fallback = True
                    chosen_model_key = FALLBACK_MODEL_KEY
                    model_id_to_use = FALLBACK_MODEL_ID
                    max_tokens_api = FALLBACK_MODEL_MAX_TOKENS
                    avatar_resp = FALLBACK_MODEL_EMOJI
            else:
                 # Router returned something unexpected (should have returned FALLBACK_MODEL_KEY in failures)
                 logging.error(f"Router returned unexpected key '{routed_key_letter}'. Using free fallback {FALLBACK_MODEL_ID}.")
                 use_fallback = True
                 chosen_model_key = FALLBACK_MODEL_KEY
                 model_id_to_use = FALLBACK_MODEL_ID
                 max_tokens_api = FALLBACK_MODEL_MAX_TOKENS
                 avatar_resp = FALLBACK_MODEL_EMOJI

            # --- API Call and Response Streaming ---
            if model_id_to_use:
                with st.chat_message("assistant", avatar=avatar_resp):
                    response_placeholder = st.empty()
                    response_placeholder.markdown("Thinking... 💭")
                    full_response = ""
                    api_call_ok = True
                    error_message_from_stream = None
                    stream_start_time = time.time()

                    for chunk_content, err_msg in streamed(model_id_to_use, chat_history, max_tokens_api):
                        if err_msg:
                            logging.error(f"Error during streaming for model {model_id_to_use}: {err_msg}")
                            if st.session_state.get("api_key_auth_failed"): error_message_from_stream = "❗ **API Authentication Error**: Update Key in ⚙️ Settings."
                            elif "rate limit" in err_msg.lower(): error_message_from_stream = f"❗ **API Rate Limit**: Please try again later. ({model_id_to_use})"
                            elif "context_length_exceeded" in err_msg.lower(): error_message_from_stream = f"❗ **Context Too Long** for {model_id_to_use}. Start new chat?"
                            # Handle Perplexity specific errors if they arise
                            elif "perplexity" in model_id_to_use.lower() and "search query failed" in err_msg.lower(): error_message_from_stream = f"❗ **Search Failed**: Perplexity could not complete the search. Try rephrasing. ({model_id_to_use})"
                            else: error_message_from_stream = f"❗ **API Error**: {err_msg}"
                            api_call_ok = False; break
                        if chunk_content is not None: full_response += chunk_content; response_placeholder.markdown(full_response + "▌")

                    stream_end_time = time.time(); logging.info(f"Streaming took {stream_end_time - stream_start_time:.2f} seconds.")

                    if error_message_from_stream: response_placeholder.markdown(error_message_from_stream); full_response = error_message_from_stream
                    elif not full_response and api_call_ok: response_placeholder.markdown("*Assistant returned an empty response.*"); full_response = ""; logging.warning(f"Model {model_id_to_use} returned empty response.")
                    else: response_placeholder.markdown(full_response) # Final display

                # --- Post-Response Processing ---
                last_usage = st.session_state.pop("last_stream_usage", None)
                prompt_tokens_used = 0; completion_tokens_used = 0
                if api_call_ok and last_usage:
                    prompt_tokens_used = last_usage.get("prompt_tokens", 0)
                    completion_tokens_used = last_usage.get("completion_tokens", 0) if full_response and not error_message_from_stream else 0
                    logging.info(f"API call completed for {model_id_to_use}. Usage: P={prompt_tokens_used}, C={completion_tokens_used}")
                elif api_call_ok and not last_usage: logging.warning(f"Token usage info not found in stream response for {model_id_to_use}.")

                assistant_message = {"role": "assistant", "content": full_response, "model": chosen_model_key, "prompt_tokens": prompt_tokens_used, "completion_tokens": completion_tokens_used}
                chat_history.append(assistant_message)

                # Record quota usage ONLY if successful, NOT fallback, and tokens > 0
                if api_call_ok and not use_fallback and chosen_model_key in NEW_PLAN_CONFIG and (prompt_tokens_used > 0 or completion_tokens_used > 0):
                    record_use(chosen_model_key, prompt_tokens_used, completion_tokens_used)
                elif api_call_ok and not use_fallback and chosen_model_key in NEW_PLAN_CONFIG:
                     logging.info(f"Skipping quota recording for {chosen_model_key} due to zero token usage.")

                # Auto-title if new and successful
                if api_call_ok and not error_message_from_stream and full_response and sessions[current_sid]["title"] == "New chat":
                   sessions[current_sid]["title"] = _autoname(prompt)

                _save(SESS_FILE, sessions) # Save updated history, title, usage
                # Use st.experimental_rerun() if available and issues persist, but st.rerun() is standard
                st.rerun()

            else: # Handle case where no model_id_to_use was determined
                 if not st.session_state.get("api_key_auth_failed"):
                    st.error("Unexpected error: Could not determine a model to use.")
                    logging.error("Failed to determine model_id_to_use after selection logic, and no API auth failure detected.")
                 _save(SESS_FILE, sessions); st.stop()
