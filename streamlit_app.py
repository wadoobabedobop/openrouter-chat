#!/usr/bin/env python3
# -*- coding: utf-8 -*- # THIS SHOULD BE LINE 2
"""
OpenRouter Streamlit Chat — Full Edition + Search + Persistence
• Persistent chat sessions & quotas across app restarts
• Daily/weekly/monthly quotas
• Pretty ‘token-jar’ gauges (fixed at top)
• Detailed model-routing panel (Mistral router) + Perplexity Search Models
• Live credit/usage stats (GET /credits)
• Auto-titling of new chats
• Comprehensive logging
• In-app API Key configuration (via Settings panel or initial setup)
"""

# ------------------------- Imports ------------------------- #
import json, logging, os, sys, subprocess, time, requests, atexit, tempfile
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo # Python 3.9+
import streamlit as st

# -------------------------- Configuration --------------------------- #
# API Key now managed via app_config.json and st.session_state
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT     = 120

# Fallback Model (Unchanged)
FALLBACK_MODEL_ID = "deepseek/deepseek-chat-v3-0324:free"
FALLBACK_MODEL_KEY = "_FALLBACK_"
FALLBACK_MODEL_EMOJI = "🆓"
FALLBACK_MODEL_MAX_TOKENS = 8000

# Model Definitions (Includes Search)
MODEL_MAP = {
    "A": "google/gemini-2.5-pro-preview",    "B": "openai/o4-mini",
    "C": "openai/chatgpt-4o-latest",         "D": "deepseek/deepseek-r1",
    "E": "anthropic/claude-3.7-sonnet",      "F": "google/gemini-2.5-flash-preview",
    "G": "perplexity/sonar",                 "H": "perplexity/sonar-reasoning-pro",
    "I": "perplexity/sonar-deep-research",
}
ROUTER_MODEL_ID = "nousresearch/deephermes-3-mistral-24b-preview:free" # Using a capable router
MAX_HISTORY_CHARS_FOR_ROUTER = 1000

# Max Tokens (Unchanged)
MAX_TOKENS = {
    "A": 16_000, "B": 8_000, "C": 16_000, "D": 8_000, "E": 8_000, "F": 8_000,
    "G": 16_000, "H": 16_000, "I": 16_000
}

# Quota Config (Unchanged)
NEW_PLAN_CONFIG = {
    "A": (5, 200, 5000, 100000, 5000, 100000, 3, 3 * 3600), "B": (10, 200, 5000, 100000, 5000, 100000, 0, 0),
    "C": (5, 200, 5000, 100000, 5000, 100000, 0, 0),       "D": (10, 200, 5000, 100000, 5000, 100000, 0, 0),
    "E": (5, 200, 5000, 100000, 5000, 100000, 0, 0),       "F": (150, 3000, 75000, 1500000, 75000, 1500000, 0, 0),
    "G": (50, 1000, 20000, 400000, 20000, 400000, 0, 0),    "H": (10, 200, 5000, 100000, 5000, 100000, 0, 0),
    "I": (5, 100, 5000, 100000, 5000, 100000, 0, 0)
}

# Emojis (Unchanged)
EMOJI = {
    "A": "🌟", "B": "🔷", "C": "🟥", "D": "🟢", "E": "🖋️", "F": "🌀",
    "G": "🔎", "H": "💡", "I": "📚"
}

# Model Descriptions (Unchanged)
MODEL_DESCRIPTIONS = {
    "A": "🌟 (gemini-2.5-pro) – High capability, moderate cost.", "B": "🔷 (o4-mini) – Mid-stakes reasoning, cost-effective.",
    "C": "🟥 (chatgpt-4o) – Polished/empathetic, HIGHEST cost.",    "D": "🟢 (deepseek-r1) – Cheap factual/technical reasoning.",
    "E": "🖋️ (claude-3.7-sonnet) – Novel, creative, high cost.",  "F": "🌀 (gemini-2.5-flash) – Quick, CHEAPEST, simple NON-SEARCH tasks.",
    "G": "🔎 (sonar) – Cheap web search, simple queries + response.", "H": "💡 (sonar-pro) – Adv. search + reasoning, complex search Qs.",
    "I": "📚 (sonar-deep) – Deep multi-step search investigation.",
}

# Router Guidance (Unchanged)
ROUTER_MODEL_GUIDANCE_SENSITIVE = {
    "A": "(Model A: High Capability [Cost Rank 5/9]) Use for complex non-search tasks. **Suitable for sensitive/crisis topics if E is unavailable.** Cheaper than E, C.",
    "B": "(Model B: Mid-Tier [Cost Rank 4/9]) Use for general moderate non-search tasks. Suitable for *mild-to-moderate* sensitivity. **Generally AVOID for severe crisis/self-harm unless E, A, C are all unavailable.** Cheaper than A, H, I, E, C.",
    "C": "(Model C: Polished, HIGHEST COST [Cost Rank 9/9]) Avoid unless extreme polish is essential AND cheaper options inadequate. **Can be a fallback for crisis if E and A are unavailable.**",
    "D": "(Model D: Factual/Technical [Cost Rank 3/9]) Use for factual/code tasks if F/G are insufficient. **NOT suitable for sensitive topics.** Slow.",
    "E": "(Model E: Novel & Nuanced, High Cost [Cost Rank 8/9]) Use for unique creative non-search tasks OR **handling serious sensitive topics/crisis situations.** **Preferred choice for crisis if available.** Cheaper than C.",
    "F": "(Model F: CHEAPEST [Cost Rank 1/9]) Use ONLY for simple, low-stakes, **NON-SEARCH**, non-sensitive tasks. ***DO NOT USE 'F' IF*** query involves: search, complexity, sensitivity (esp. crisis/safety), math, deep analysis, high accuracy needs.",
    "G": "(Model G: Cheap Search [Cost Rank 2/9]) **Use ONLY for queries requiring web search.** Best for *simple* search needs. If query is complex *beyond* the search itself, prefer H. **NOT suitable for sensitive topics.**",
    "H": "(Model H: Adv Search+Reasoning [Cost Rank 6/9]) **Use ONLY for queries requiring web search.** Use for *complex* search queries needing reasoning *over search results*. Prefer over G if query complexity warrants cost. **NOT suitable for sensitive topics.**",
    "I": "(Model I: Deep Search [Cost Rank 7/9]) **Use ONLY for queries requiring extensive web research.** Use for explicit requests for *deep investigation*, multi-step research. High cost. **NOT suitable for sensitive topics.**"
}

# --- File Paths & Timezone ---
TZ = ZoneInfo("Australia/Sydney")
DATA_DIR   = Path(__file__).parent.resolve() # Use resolve for absolute path
SESS_FILE  = DATA_DIR / "chat_sessions.json"
QUOTA_FILE = DATA_DIR / "quotas.json"
CONFIG_FILE = DATA_DIR / "app_config.json"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------- Logging Setup (Early) ----------------------------
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
numeric_level = getattr(logging, log_level, logging.INFO)
# Configure logging only if handlers are not already present
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
        stream=sys.stdout,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info(f"Logging initialized. Level: {log_level}")
    logging.info(f"Data directory: {DATA_DIR}")
    logging.info(f"Session file: {SESS_FILE}")
    logging.info(f"Quota file: {QUOTA_FILE}")
    logging.info(f"Config file: {CONFIG_FILE}")


# ------------------------ Helper Functions (Revised Save) -----------------------

def _load(path: Path, default):
    """Loads JSON data from a file, returning default if not found or invalid."""
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        logging.warning(f"File not found: {path}. Returning default.")
        return default
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from {path}: {e}. Returning default.")
        # Optionally: Backup corrupted file here
        try:
            corrupt_backup_path = path.with_suffix(f".corrupt.{int(time.time())}.json")
            path.rename(corrupt_backup_path)
            logging.info(f"Backed up corrupted file to: {corrupt_backup_path}")
        except Exception as backup_err:
             logging.error(f"Could not back up corrupted file {path}: {backup_err}")
        return default
    except Exception as e:
        logging.error(f"Unexpected error loading {path}: {e}. Returning default.")
        return default

def _save(path: Path, obj):
    """Saves a Python object to a JSON file atomically."""
    try:
        # Create a temporary file in the same directory
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as tmp_file:
            json.dump(obj, tmp_file, indent=2, ensure_ascii=False)
            temp_path = tmp_file.name # Get the temporary file path
        # Atomically replace the original file with the temporary file
        os.replace(temp_path, path) # os.replace is atomic on most modern systems
        # logging.debug(f"Successfully saved data to {path}") # Reduce log noise
    except Exception as e:
        logging.exception(f"Failed to save file atomically to {path}: {e}") # Log full traceback
        # Attempt to remove the temporary file if it still exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try: os.remove(temp_path)
            except OSError: logging.error(f"Could not remove temporary file {temp_path} after failed save.")


# --- Date/Time Helpers ---
def _today():    return datetime.now(TZ).date().isoformat()
def _ymonth():   return datetime.now(TZ).strftime("%Y-%m")

# --- App Config Helpers ---
def _load_app_config():
    return _load(CONFIG_FILE, {})

def _save_app_config(api_key_value: str):
    config_data = _load_app_config()
    config_data["openrouter_api_key"] = api_key_value
    _save(CONFIG_FILE, config_data) # Use robust save
    logging.info("Saved app config (API key).")

# --- Token Formatting ---
def format_token_count(num):
    if num is None: return "N/A"
    try: num = float(num)
    except (ValueError, TypeError): return "N/A"
    if num == float('inf'): return "∞"
    if num < 1000: return str(int(num))
    if num < 1_000_000: return f"{num/1000:.1f}".replace(".0", "") + "k"
    return f"{num/1_000_000:.1f}".replace(".0", "") + "M"

# --------------------- Persistence Setup (Load Early, Save on Exit) ------------------------

# Load data ONCE at the start of the script execution
_initial_sessions = _load(SESS_FILE, {})
_initial_quota_data = _load(QUOTA_FILE, {})
logging.info("Initial session and quota data loaded into memory.")

# Use these global variables to hold the state in memory
# We wrap sessions in a dictionary to make it mutable for atexit if needed,
# though direct modification should also work. Quotas are already a dict.
app_state = {
    "sessions": _initial_sessions,
    "quota_data": _initial_quota_data,
    "quota_data_dirty": False, # Flag to track if quota data needs saving
    "sessions_dirty": False   # Flag to track if session data needs saving
}
_g_quota_data_last_refreshed_stamps = {"d": None, "m": None} # Keep track of reset stamps


def save_app_state():
    """Saves the current in-memory state (quotas, sessions) to files if marked dirty."""
    global app_state
    saved_something = False
    if app_state.get("quota_data_dirty"):
        try:
            _save(QUOTA_FILE, app_state["quota_data"])
            logging.info(f"Persisted quota data to {QUOTA_FILE} on exit/save.")
            app_state["quota_data_dirty"] = False # Reset flag after saving
            saved_something = True
        except Exception as e:
            logging.error(f"Failed to save quota data on exit/save: {e}")

    if app_state.get("sessions_dirty"):
        try:
            _save(SESS_FILE, app_state["sessions"])
            logging.info(f"Persisted session data to {SESS_FILE} on exit/save.")
            app_state["sessions_dirty"] = False # Reset flag after saving
            saved_something = True
        except Exception as e:
            logging.error(f"Failed to save session data on exit/save: {e}")

    # if saved_something:
    #      logging.info("save_app_state finished.")
    # else:
    #      logging.debug("save_app_state called, but no data marked as dirty.")


# Register the save function to be called at Python interpreter exit
# This is a best-effort approach. It might not run if the process is killed forcefully (SIGKILL).
atexit.register(save_app_state)
logging.info("Registered save_app_state function with atexit for persistence.")

# --------------------- Quota Management (Revised for Persistence) ------------------------
USAGE_KEYS_PERIODIC = ["d_u", "m_u", "d_it_u", "m_it_u", "d_ot_u", "m_ot_u"]
MODEL_A_3H_CALLS_KEY = "model_A_3h_calls"

def _reset(block: dict, period_prefix: str, current_stamp: str, model_keys_zeros: dict) -> bool:
    """Resets quota usage for a period if the timestamp differs. Operates on the passed dict."""
    data_changed = False
    period_stamp_key = period_prefix

    if block.get(period_stamp_key) != current_stamp:
        block[period_stamp_key] = current_stamp
        for usage_type_suffix in ["_u", "_it_u", "_ot_u"]:
            usage_dict_key = f"{period_prefix}{usage_type_suffix}"
            # Ensure previous data (if any) is cleared before assigning new zeros
            block[usage_dict_key] = {}
            block[usage_dict_key] = model_keys_zeros.copy()
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
                for model_k_map in model_keys_zeros.keys():
                    if model_k_map not in current_period_usage_dict:
                        current_period_usage_dict[model_k_map] = 0
                        data_changed = True
                        logging.info(f"Added missing model '{model_k_map}' to usage dict '{usage_dict_key}' for stamp '{current_stamp}'.")
    return data_changed

def _ensure_quota_data_is_current():
    """
    Checks and updates the in-memory quota data (_app_state['quota_data'])
    based on current time and configured models.
    Returns True if the data was modified (needs saving), False otherwise.
    """
    global app_state, _g_quota_data_last_refreshed_stamps
    now_d_stamp, now_m_stamp = _today(), _ymonth()
    data_was_modified = False

    # Use the in-memory dictionary
    current_q_data = app_state["quota_data"]

    # Check if period stamps need updating
    needs_daily_reset = current_q_data.get("d") != now_d_stamp
    needs_monthly_reset = current_q_data.get("m") != now_m_stamp

    # Determine active models and cleanup obsolete ones
    active_model_keys = set(NEW_PLAN_CONFIG.keys())
    cleaned_during_load = False
    for usage_key_template in USAGE_KEYS_PERIODIC:
        if usage_key_template in current_q_data:
            current_period_usage_dict = current_q_data[usage_key_template]
            keys_in_usage = list(current_period_usage_dict.keys())
            for model_key_in_usage in keys_in_usage:
                if model_key_in_usage not in active_model_keys:
                    try:
                        del current_period_usage_dict[model_key_in_usage]
                        logging.info(f"[Quota Check] Removed obsolete model key '{model_key_in_usage}' from usage '{usage_key_template}'.")
                        cleaned_during_load = True
                    except KeyError: pass
    if cleaned_during_load: data_was_modified = True

    # Reset periods if needed
    current_model_zeros = {k: 0 for k in active_model_keys}
    reset_occurred_d = _reset(current_q_data, "d", now_d_stamp, current_model_zeros)
    reset_occurred_m = _reset(current_q_data, "m", now_m_stamp, current_model_zeros)
    if reset_occurred_d or reset_occurred_m: data_was_modified = True

    # --- Handle Model A 3-hour call pruning ---
    three_hr_config_valid = ("A" in NEW_PLAN_CONFIG and len(NEW_PLAN_CONFIG["A"]) > 7 and
                             NEW_PLAN_CONFIG["A"][6] > 0 and NEW_PLAN_CONFIG["A"][7] > 0)

    if three_hr_config_valid:
        three_hr_window_seconds = NEW_PLAN_CONFIG["A"][7]
        if MODEL_A_3H_CALLS_KEY not in current_q_data:
            current_q_data[MODEL_A_3H_CALLS_KEY] = []
            data_was_modified = True
            logging.info(f"[Quota Check] Initialized Model A 3-hour call list ({MODEL_A_3H_CALLS_KEY}).")

        # Prune expired timestamps
        current_time = time.time()
        original_len = len(current_q_data.get(MODEL_A_3H_CALLS_KEY, []))
        current_q_data[MODEL_A_3H_CALLS_KEY] = [
            ts for ts in current_q_data.get(MODEL_A_3H_CALLS_KEY, [])
            if current_time - ts < three_hr_window_seconds
        ]
        new_len = len(current_q_data.get(MODEL_A_3H_CALLS_KEY, []))
        if new_len != original_len:
             logging.info(f"[Quota Check] Pruned Model A 3-hour calls. Original: {original_len}, New: {new_len}.")
             data_was_modified = True

    elif MODEL_A_3H_CALLS_KEY in current_q_data:
        # Config removed, clean up the key
        del current_q_data[MODEL_A_3H_CALLS_KEY]
        data_was_modified = True
        logging.info(f"[Quota Check] Removed obsolete Model A 3-hour call list ({MODEL_A_3H_CALLS_KEY}).")

    # Update the last refreshed stamps in memory
    _g_quota_data_last_refreshed_stamps = {"d": now_d_stamp, "m": now_m_stamp}

    # Mark data as dirty if it was modified
    if data_was_modified:
        app_state["quota_data_dirty"] = True
        logging.info("Quota data was modified in memory and marked as dirty for saving.")
        # NO save here - saving is triggered by save_app_state or explicit calls

    return data_was_modified # Return whether data was changed


def get_quota_usage_and_limits(model_key: str):
    """Gets quota usage and limits for a model from the in-memory state."""
    if model_key == FALLBACK_MODEL_KEY:
        # Return infinite limits for the fallback model
        return {"used_daily_msg": 0, "limit_daily_msg": float('inf'),
                "used_monthly_msg": 0, "limit_monthly_msg": float('inf'),
                "used_daily_in_tokens": 0, "limit_daily_in_tokens": float('inf'),
                "used_monthly_in_tokens": 0, "limit_monthly_in_tokens": float('inf'),
                "used_daily_out_tokens": 0, "limit_daily_out_tokens": float('inf'),
                "used_monthly_out_tokens": 0, "limit_monthly_out_tokens": float('inf'),
                "used_3hr_msg": 0, "limit_3hr_msg": float('inf')}

    if model_key not in NEW_PLAN_CONFIG:
        logging.error(f"Model key '{model_key}' not in NEW_PLAN_CONFIG. Cannot get quotas.")
        return {}

    # Ensure the in-memory data is up-to-date before reading
    _ensure_quota_data_is_current() # This function now just checks/updates memory

    current_q_data = app_state["quota_data"]
    plan = NEW_PLAN_CONFIG[model_key]
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
        "used_3hr_msg": 0
    }

    if model_key == "A" and has_3hr_limit:
        # Read directly from the already pruned list in memory
        usage["used_3hr_msg"] = len(current_q_data.get(MODEL_A_3H_CALLS_KEY, []))

    return {**usage, **limits}


def is_model_available(model_key: str) -> bool:
    """Checks if a model is available based on in-memory quota data."""
    if model_key == FALLBACK_MODEL_KEY: return True
    if model_key not in NEW_PLAN_CONFIG: return False

    # get_quota_usage_and_limits ensures data is current before returning stats
    stats = get_quota_usage_and_limits(model_key)
    if not stats: return False

    # Simplified checks using != float('inf')
    if stats["limit_daily_msg"] != float('inf') and stats["used_daily_msg"] >= stats["limit_daily_msg"]: return False
    if stats["limit_monthly_msg"] != float('inf') and stats["used_monthly_msg"] >= stats["limit_monthly_msg"]: return False
    if stats["limit_daily_in_tokens"] != float('inf') and stats["used_daily_in_tokens"] >= stats["limit_daily_in_tokens"]: return False
    if stats["limit_monthly_in_tokens"] != float('inf') and stats["used_monthly_in_tokens"] >= stats["limit_monthly_in_tokens"]: return False
    if stats["limit_daily_out_tokens"] != float('inf') and stats["used_daily_out_tokens"] >= stats["limit_daily_out_tokens"]: return False
    if stats["limit_monthly_out_tokens"] != float('inf') and stats["used_monthly_out_tokens"] >= stats["limit_monthly_out_tokens"]: return False
    if model_key == "A" and stats["limit_3hr_msg"] != float('inf') and stats["used_3hr_msg"] >= stats["limit_3hr_msg"]: return False

    return True


def get_remaining_daily_messages(model_key: str) -> int | float:
    """Gets remaining daily messages from in-memory state."""
    if model_key == FALLBACK_MODEL_KEY: return float('inf')
    if model_key not in NEW_PLAN_CONFIG: return 0

    stats = get_quota_usage_and_limits(model_key) # Ensures data is current
    if not stats: return 0
    if stats["limit_daily_msg"] == float('inf') or stats["limit_daily_msg"] <= 0: return float('inf')
    return max(0, stats["limit_daily_msg"] - stats["used_daily_msg"])


def record_use(model_key: str, prompt_tokens: int, completion_tokens: int):
    """Records model usage in the in-memory quota data and marks it dirty."""
    global app_state
    if model_key == FALLBACK_MODEL_KEY or model_key not in NEW_PLAN_CONFIG:
        # logging.debug(f"Skipping usage recording for non-quota-tracked model: {model_key}")
        return

    # Ensures the dicts/keys exist before incrementing
    # This function also marks data as dirty if it modifies it.
    data_modified_by_check = _ensure_quota_data_is_current()

    current_q_data = app_state["quota_data"]

    # Use setdefault to ensure keys exist and initialize if necessary
    # This prevents KeyErrors if _ensure_quota_data_is_current didn't already add them
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
    if model_key == "A" and "A" in NEW_PLAN_CONFIG and len(NEW_PLAN_CONFIG["A"]) > 7 and NEW_PLAN_CONFIG["A"][6] > 0:
        # Ensure the list exists before appending
        current_q_data.setdefault(MODEL_A_3H_CALLS_KEY, []).append(time.time())

    # Mark quota data as dirty since we modified it
    app_state["quota_data_dirty"] = True

    # Save immediately after recording use for higher reliability
    save_app_state()

    logging.info(f"Recorded usage for '{model_key}': 1 msg, {prompt_tokens}p, {completion_tokens}c. State marked dirty.")


# --------------------- Session Management (Operates on app_state['sessions']) -----------------------

def _delete_unused_blank_sessions(keep_sid: str = None):
    """Deletes blank sessions from the in-memory app_state['sessions']."""
    global app_state
    sessions_dict = app_state["sessions"]
    sids_to_delete = []
    modified = False
    for sid, data in list(sessions_dict.items()): # Iterate over a copy of items
        if sid == keep_sid: continue
        if data.get("title") == "New chat" and not data.get("messages"):
            sids_to_delete.append(sid)
    if sids_to_delete:
        for sid_del in sids_to_delete:
            logging.info(f"Auto-deleting blank session from memory: {sid_del}")
            try:
                del sessions_dict[sid_del]
                modified = True
            except KeyError: logging.warning(f"Session {sid_del} already deleted, skipping.")
        if modified:
            app_state["sessions_dirty"] = True # Mark as dirty
            # Don't save here, let caller decide or rely on atexit/explicit saves
    return modified # Return if deletion occurred


def _new_sid():
    """Creates a new session in app_state['sessions'] and marks state as dirty."""
    global app_state
    sessions_dict = app_state["sessions"]
    sid = str(int(time.time() * 1000))
    sessions_dict[sid] = {"title": "New chat", "messages": []}
    app_state["sessions_dirty"] = True # Mark as dirty
    logging.info(f"Created new session in memory: {sid}")
    # Clean up *other* blank sessions immediately after creating a new one
    _delete_unused_blank_sessions(keep_sid=sid)
    # Save immediately after creating a new session
    save_app_state()
    return sid

def _autoname(seed: str) -> str:
    # Unchanged
    words = seed.strip().split()
    cand = " ".join(words[:4]) or "Chat"
    return (cand[:30] + "…") if len(cand) > 30 else cand

# --- Define 'sessions' alias for convenience ---
# This allows the rest of the code to use 'sessions' while operating on the persistent state
sessions = app_state["sessions"]

# -------------------------- API Calls (Unchanged) --------------------------
# (api_post, streamed functions remain the same)
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

    except ValueError as ve: logging.error(f"ValueError during streamed call setup: {ve}"); yield None, str(ve)
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else 'N/A'; text = e.response.text if e.response else 'No response text'
        logging.error(f"Stream HTTPError {status_code}: {text}")
        if status_code == 401: st.session_state.api_key_auth_failed = True
        yield None, f"HTTP {status_code}: An error occurred with the API provider. Details: {text}"
    except requests.exceptions.RequestException as e: logging.error(f"Stream Network Error: {e}"); yield None, f"Network Error: Failed to connect to API. {e}"
    except Exception as e: logging.exception(f"Unexpected error during streamed API call: {e}"); yield None, f"An unexpected error occurred: {e}"


# ------------------------- Model Routing (Unchanged - V4) -----------------------
# (route_choice function remains the same)
def route_choice(user_msg: str, allowed: list[str], chat_history: list) -> str:
    non_search_allowed = [k for k in allowed if k in "ABCDEF"]
    search_allowed = [k for k in allowed if k in "GHI"]

    if "F" in non_search_allowed: fallback_choice_letter = "F"
    elif "G" in search_allowed: fallback_choice_letter = "G"
    elif non_search_allowed: fallback_choice_letter = non_search_allowed[0]
    elif search_allowed: fallback_choice_letter = search_allowed[0]
    elif "F" in MODEL_MAP: fallback_choice_letter = "F"
    elif "G" in MODEL_MAP: fallback_choice_letter = "G"
    elif MODEL_MAP: fallback_choice_letter = list(MODEL_MAP.keys())[0]
    else: logging.error("Router: No models configured. Using FALLBACK_MODEL_KEY."); return FALLBACK_MODEL_KEY

    if not allowed: logging.warning(f"Router: No models available due to quotas. Using free fallback: {FALLBACK_MODEL_KEY}."); return FALLBACK_MODEL_KEY
    if len(allowed) == 1: logging.info(f"Router: Only one model allowed ('{allowed[0]}'), selecting it."); return allowed[0]

    history_segments = []; current_chars = 0
    relevant_history_for_router = chat_history[:-1] if chat_history else []
    for msg in reversed(relevant_history_for_router):
        role = msg.get("role", "assistant").capitalize(); content = msg.get("content", "")
        if not isinstance(content, str): content = str(content)
        segment = f"{role}: {content}\n";
        if current_chars + len(segment) > MAX_HISTORY_CHARS_FOR_ROUTER: break
        history_segments.append(segment); current_chars += len(segment)
    history_context_str = "".join(reversed(history_segments)).strip() or "No prior conversation history."

    system_prompt_parts = [
        "You are an expert AI model routing assistant. Your task is to select the *single most appropriate and cost-effective* model letter from the 'Available Models' list to handle the 'Latest User Query'.",
        "Core Principles:",
        "1. **Assess Need: Crisis -> Search -> General:** First, check for CRITICAL SENSITIVITY (self-harm, crisis). Second, check if WEB SEARCH is explicitly or implicitly required. Third, assess general capability needed.",
        "2. **Prioritize Safety (CRISIS):** If the query mentions/implies self-harm, suicide, immediate danger, or severe crisis, **YOU MUST CHOOSE** a highly capable, safe model. Order: **E > A > C**. If none available, use B ONLY as a last resort. **IGNORE search/cost** for crisis. Models F, D, G, H, I are **NOT SUITABLE**.",
        "3. **Prioritize Functionality (SEARCH):** If the query requires up-to-date info, current events, or specific web lookups (and is NOT a crisis), **YOU MUST CHOOSE** a Search model (G, H, I). Select based on complexity:",
        f"   - **Deep Research ('I'):** For explicit multi-step investigations, comprehensive reports needing broad search. (Available: {'Yes' if 'I' in allowed else 'No'})",
        f"   - **Advanced Search + Reasoning ('H'):** For complex questions needing search AND reasoning on results. (Available: {'Yes' if 'H' in allowed else 'No'})",
        f"   - **Simple Search ('G'):** For basic lookups (news, weather, definitions). (Available: {'Yes' if 'G' in allowed else 'No'})",
        "   - **Search Fallback:** If the ideal search model (I/H/G) is unavailable, choose the next best *available* search model (I -> H -> G). If NO search models (G, H, I) are available, select the best *general* model.",
        "4. **Maximize Cost-Effectiveness (GENERAL / NON-CRISIS / NON-SEARCH):** If NOT crisis and NOT search, choose the ***absolute cheapest*** available general model (F, D, B, A, E, C) that meets the capability needed. General Cost Order: **F < G < D < B < A ≈ H ≈ I < E < C**.",
        "5. **Consider History:** Use 'Recent Conversation History' for context."
    ]
    system_prompt_parts.append("\nAvailable Models (Cost Order: F < G < D < B < A ≈ H ≈ I < E < C - may be overridden):")
    for k_model_key in sorted(allowed):
        description = ROUTER_MODEL_GUIDANCE_SENSITIVE.get(k_model_key, f"(Model {k_model_key} - Description not found).")
        system_prompt_parts.append(f"- {k_model_key}: {description}")

    system_prompt_parts.append("\nDecision Process for 'Latest User Query':")
    system_prompt_parts.append("1. **CRISIS CHECK:** Crisis mentioned/implied?")
    system_prompt_parts.append("   - IF YES: Choose E (if avail) -> A (if avail) -> C (if avail) -> B (if avail). STOP.")
    system_prompt_parts.append("   - IF NO: Proceed.")
    system_prompt_parts.append("2. **SEARCH CHECK:** Web search/current info needed?")
    system_prompt_parts.append("   - IF YES:")
    system_prompt_parts.append("     - Deep research needed & 'I' avail? CHOOSE 'I'.")
    system_prompt_parts.append("     - Else, complex search+reasoning needed & 'H' avail? CHOOSE 'H'.")
    system_prompt_parts.append("     - Else, simple lookup & 'G' avail? CHOOSE 'G'.")
    system_prompt_parts.append("     - Else (chosen search unavail): Fallback I->H->G among *available* search models.")
    system_prompt_parts.append("     - Else (NO search models avail): Proceed to Step 3 (Standard Routing).")
    system_prompt_parts.append("   - IF NO: Proceed.")
    system_prompt_parts.append("3. **Standard Routing (Non-Crisis, Non-Search):** Select CHEAPEST sufficient model.")
    system_prompt_parts.append(f"   - 'F' sufficient? (Avail: {'Yes' if 'F' in allowed else 'No'}) CHOOSE 'F'.")
    system_prompt_parts.append(f"   - Else, 'D' sufficient? (Avail: {'Yes' if 'D' in allowed else 'No'}) CHOOSE 'D'.")
    system_prompt_parts.append(f"   - Else, 'B' sufficient? (Avail: {'Yes' if 'B' in allowed else 'No'}) CHOOSE 'B'.")
    system_prompt_parts.append(f"   - Else, 'A' sufficient? (Avail: {'Yes' if 'A' in allowed else 'No'}) CHOOSE 'A'.")
    system_prompt_parts.append(f"   - Else, 'E' sufficient? (Avail: {'Yes' if 'E' in allowed else 'No'}) CHOOSE 'E'.")
    system_prompt_parts.append(f"   - Else, 'C' needed? (Avail: {'Yes' if 'C' in allowed else 'No'}) CHOOSE 'C'.")

    system_prompt_parts.append("\nRecent Conversation History (Context):")
    system_prompt_parts.append(history_context_str)
    system_prompt_parts.append(f"\nAvailable Model Letters: {', '.join(sorted(allowed))}")
    system_prompt_parts.append("\nINSTRUCTION: Analyze the 'Latest User Query' using the 3-step process. Respond with ONLY the single capital letter of your chosen model. NO EXPLANATION.")

    final_system_message = "\n".join(system_prompt_parts)
    logging.debug(f"Router System Prompt V4:\n{final_system_message}")
    router_messages = [{"role": "system", "content": final_system_message}, {"role": "user", "content": user_msg}]
    payload_r = {"model": ROUTER_MODEL_ID, "messages": router_messages, "max_tokens": 10, "temperature": 0.1}
    logging.debug(f"Router Payload: {json.dumps(payload_r, indent=2)}")

    try:
        r = api_post(payload_r)
        choice_data = r.json()
        logging.debug(f"Router Full Response JSON: {json.dumps(choice_data, indent=2)}")
        raw_text_response = choice_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()
        logging.info(f"Router raw text response: '{raw_text_response}' for query: '{user_msg[:100]}...'")

        chosen_model_letter = None
        if raw_text_response in allowed: chosen_model_letter = raw_text_response
        else:
             for char_in_response in raw_text_response:
                 if char_in_response in allowed: chosen_model_letter = char_in_response; break

        if chosen_model_letter:
            if not is_model_available(chosen_model_letter): # Final availability check
                 logging.warning(f"Router chose '{chosen_model_letter}' but it became unavailable. Using free fallback.")
                 return FALLBACK_MODEL_KEY
            else: return chosen_model_letter
        else: # Fallback if router response is invalid
            logging.warning(f"Router returned ('{raw_text_response}') invalid/unavailable choice. Using fallback logic.")
            if is_model_available(fallback_choice_letter): return fallback_choice_letter
            else: logging.warning(f"Fallback choice '{fallback_choice_letter}' also unavailable. Using free fallback."); return FALLBACK_MODEL_KEY

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
    logging.warning(f"Router failed. Using fallback logic -> '{fallback_choice_letter}' or free.")
    if is_model_available(fallback_choice_letter): return fallback_choice_letter
    else: logging.warning(f"Fallback choice '{fallback_choice_letter}' unavailable. Using free fallback."); return FALLBACK_MODEL_KEY

# --------------------- Credits Endpoint (Unchanged) -----------------------
# (get_credits function remains the same)
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

# ------------------------- UI Styling (Unchanged) --------------------------
# (load_custom_css function remains the same)
def load_custom_css():
    css = """
    <style>
        :root { /* CSS Variables */
            --app-bg-color: #F8F9FA; --app-secondary-bg-color: #FFFFFF;
            --app-text-color: #212529; --app-text-secondary-color: #6C757D;
            --app-primary-color: #007BFF; --app-primary-hover-color: #0056b3;
            --app-divider-color: #DEE2E6; --app-border-color: #CED4DA;
            --app-success-color: #28A745; --app-warning-color: #FFC107; --app-danger-color: #DC3545;
            --border-radius-sm: 0.2rem; --border-radius-md: 0.25rem; --border-radius-lg: 0.3rem;
            --spacing-xs: 0.25rem; --spacing-sm: 0.5rem; --spacing-md: 1rem; --spacing-lg: 1.5rem;
            --shadow-sm: 0 .125rem .25rem rgba(0,0,0,.075); --shadow-md: 0 .5rem 1rem rgba(0,0,0,.15);
            --app-font: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
        }
        body, .stApp { font-family: var(--app-font) !important; background-color: var(--app-bg-color) !important; color: var(--app-text-color) !important; }
        .main .block-container { background-color: var(--app-bg-color); padding-top: var(--spacing-md); padding-bottom: var(--spacing-lg); max-width: 1200px; }
        [data-testid="stSidebar"] { background-color: var(--app-secondary-bg-color); border-right: 1px solid var(--app-divider-color); padding: var(--spacing-md); width: 300px !important; }
        [data-testid="stSidebar"] .stImage > img { border-radius: var(--border-radius-md); box-shadow: var(--shadow-sm); width: 40px !important; height: 40px !important; margin-right: var(--spacing-sm); }
        [data-testid="stSidebar"] h1 { font-size: 1.4rem !important; color: var(--app-text-color); font-weight: 600; margin-bottom: 0; line-height: 1.2; padding-top: 0.15rem; }
        .sidebar-title-container { display: flex; align-items: center; margin-bottom: var(--spacing-md); }
        [data-testid="stSidebar"] .stButton > button { border-radius: var(--border-radius-md); border: 1px solid var(--app-border-color); padding: 0.5em 0.8em; font-size: 0.9rem; background-color: var(--app-secondary-bg-color); color: var(--app-text-color); transition: background-color 0.2s, border-color 0.2s; width: 100%; margin-bottom: var(--spacing-sm); text-align: left; font-weight: 500; }
        [data-testid="stSidebar"] .stButton > button:hover:not(:disabled) { border-color: var(--app-primary-color); background-color: color-mix(in srgb, var(--app-primary-color) 8%, transparent); }
        [data-testid="stSidebar"] .stButton > button:disabled { opacity: 1.0; cursor: default; background-color: color-mix(in srgb, var(--app-primary-color) 15%, transparent) !important; border-left: 3px solid var(--app-primary-color) !important; border-top-color: var(--app-border-color) !important; border-right-color: var(--app-border-color) !important; border-bottom-color: var(--app-border-color) !important; font-weight: 600; color: var(--app-text-color); }
        [data-testid="stSidebar"] [data-testid*="new_chat_button_top"] > button { background-color: var(--app-primary-color); color: white; border-color: var(--app-primary-color); font-weight: 600; }
        [data-testid="stSidebar"] [data-testid*="new_chat_button_top"] > button:hover { background-color: var(--app-primary-hover-color); border-color: var(--app-primary-hover-color); }
        [data-testid="stSidebar"] [data-testid*="new_chat_button_top"] > button:disabled { border-left-width: 1px !important; background-color: var(--app-secondary-bg-color) !important; border-color: var(--app-border-color) !important; color: var(--app-text-secondary-color) !important; cursor: not-allowed; }
        [data-testid="stSidebar"] h3, [data-testid="stSidebar"] .stSubheader { font-size: 0.75rem !important; text-transform: uppercase; font-weight: 600; color: var(--app-text-secondary-color); margin-top: var(--spacing-md); margin-bottom: var(--spacing-sm); letter-spacing: 0.03em; }
        [data-testid="stSidebar"] [data-testid="stExpander"] { border: 1px solid var(--app-divider-color); border-radius: var(--border-radius-md); background-color: var(--app-secondary-bg-color); margin-bottom: var(--spacing-sm); }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary { padding: 0.5rem var(--spacing-sm) !important; font-size: 0.8rem !important; font-weight: 500 !important; color: var(--app-text-color) !important; border-bottom: 1px solid var(--app-divider-color); border-top-left-radius: var(--border-radius-md); border-top-right-radius: var(--border-radius-md); }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover { background-color: color-mix(in srgb, var(--app-text-color) 4%, transparent); }
        [data-testid="stSidebar"] [data-testid="stExpander"] div[data-testid="stExpanderDetails"] { padding: var(--spacing-sm) !important; background-color: color-mix(in srgb, var(--app-bg-color) 50%, var(--app-secondary-bg-color) 50%); border-bottom-left-radius: var(--border-radius-md); border-bottom-right-radius: var(--border-radius-md); }
        [data-testid="stSidebar"] [data-testid="stExpander"][aria-label^="⚡ DAILY MODEL QUOTAS"] div[data-testid="stExpanderDetails"] { padding: 0.4rem var(--spacing-xs) 0.1rem var(--spacing-xs) !important; }
        [data-testid="stSidebar"] [data-testid="stExpander"][aria-label^="⚡ DAILY MODEL QUOTAS"] div[data-testid="stHorizontalBlock"] { gap: 0.15rem !important; }
        .compact-quota-item { display: flex; flex-direction: column; align-items: center; text-align: center; padding: var(--spacing-xs); background-color: color-mix(in srgb, var(--app-text-color) 2%, transparent); border-radius: var(--border-radius-sm); min-width: 30px; flex-grow: 1; }
        .cq-info { font-size: 0.65rem; margin-bottom: 2px; line-height: 1; white-space: nowrap; color: var(--app-text-color); }
        .cq-bar-track { width: 100%; height: 6px; background-color: color-mix(in srgb, var(--app-text-color) 10%, transparent); border: 1px solid var(--app-divider-color); border-radius: 3px; overflow: hidden; margin-bottom: 3px; }
        .cq-bar-fill { height: 100%; border-radius: 3px; transition: width 0.3s ease-in-out, background-color 0.3s ease-in-out; }
        .cq-value { font-size: 0.65rem; font-weight: 600; line-height: 1; }
        .settings-panel { border: 1px solid var(--app-divider-color); border-radius: var(--border-radius-md); padding: var(--spacing-sm); margin-top: var(--spacing-xs); margin-bottom: var(--spacing-md); background-color: var(--app-bg-color); }
        .settings-panel .stTextInput input { border-color: var(--app-border-color) !important; background-color: var(--app-secondary-bg-color) !important; color: var(--app-text-color) !important; font-size: 0.85rem; }
        .settings-panel .stSubheader { color: var(--app-text-color) !important; font-weight: 600 !important; font-size: 0.9rem !important; margin-bottom: var(--spacing-xs) !important; }
        .settings-panel hr { border-top: 1px solid var(--app-divider-color); margin: var(--spacing-sm) 0; }
        .detailed-quota-modelname { font-weight: 600; font-size: 0.95em; margin-bottom: 0.2rem; display:block; color: var(--app-primary-color); }
        .detailed-quota-block { font-size: 0.8rem; line-height: 1.5; }
        .detailed-quota-block ul { list-style-type: none; padding-left: 0; margin-bottom: 0.3rem;}
        .detailed-quota-block li { margin-bottom: 0.1rem; }
        [data-testid="stChatInputContainer"] { background-color: var(--app-secondary-bg-color); border-top: 1px solid var(--app-divider-color); padding: var(--spacing-sm) var(--spacing-md); box-shadow: 0 -2px 5px rgba(0,0,0,0.03); }
        [data-testid="stChatInput"] textarea { border: 1px solid var(--app-border-color) !important; border-radius: var(--border-radius-md) !important; background-color: var(--app-secondary-bg-color) !important; color: var(--app-text-color) !important; box-shadow: var(--shadow-sm) inset; }
        [data-testid="stChatInput"] textarea:focus { border-color: var(--app-primary-color) !important; box-shadow: 0 0 0 0.2rem color-mix(in srgb, var(--app-primary-color) 25%, transparent) !important; }
        [data-testid="stChatMessage"] { border-radius: var(--border-radius-lg); padding: 0.8rem 1rem; margin-bottom: var(--spacing-sm); box-shadow: var(--shadow-sm); border: 1px solid transparent; max-width: 85%; line-height: 1.5; }
        [data-testid="stChatMessage"] p { margin-bottom: 0.5em; } [data-testid="stChatMessage"] p:last-child { margin-bottom: 0; }
        [data-testid="stChatMessage"][data-testid^="stChatMessageUser"] { background-color: var(--app-primary-color); color: white; margin-left: auto; border-bottom-right-radius: var(--border-radius-sm); }
        [data-testid="stChatMessage"][data-testid^="stChatMessageAssistant"] { background-color: var(--app-secondary-bg-color); color: var(--app-text-color); margin-right: auto; border: 1px solid var(--app-divider-color); border-bottom-left-radius: var(--border-radius-sm); }
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { padding-top: 0.1rem; padding-bottom: 0.1rem; }
        [data-testid="stChatMessage"] code { font-size: 0.85em; padding: 0.2em 0.4em; margin: 0; background-color: color-mix(in srgb, var(--app-text-color) 5%, transparent); border-radius: var(--border-radius-sm); }
        [data-testid="stChatMessage"] pre > code { background-color: initial; padding: 0; border-radius: 0; }
        .sidebar-divider { margin: var(--spacing-md) 0; border: 0; border-top: 1px solid var(--app-divider-color); }
        /* #GithubIcon { display: none; } */
        .main .stButton > button:not([data-testid*="new_chat_button_top"]):not([data-testid*="toggle_settings_button_sidebar"]):not([data-testid*="session_button_"]) { border-radius: var(--border-radius-md); border: 1px solid var(--app-primary-color); background-color: var(--app-primary-color); color: white; padding: 0.5em 1em; font-weight: 500; }
        .main .stButton > button:not([data-testid*="new_chat_button_top"]):not([data-testid*="toggle_settings_button_sidebar"]):not([data-testid*="session_button_"]):hover { background-color: var(--app-primary-hover-color); border-color: var(--app-primary-hover-color); }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ----------------- API Key State Initialization (Checks Config File) -------------------
if "openrouter_api_key" not in st.session_state:
    app_conf = _load_app_config()
    st.session_state.openrouter_api_key = app_conf.get("openrouter_api_key", None)
    logging.info("Initialized API key from config file into session state.")
if "api_key_auth_failed" not in st.session_state: st.session_state.api_key_auth_failed = False

api_key_is_syntactically_valid = is_api_key_valid(st.session_state.get("openrouter_api_key"))
app_requires_api_key_setup = (
    not st.session_state.get("openrouter_api_key") or
    not api_key_is_syntactically_valid or
    st.session_state.get("api_key_auth_failed", False)
)

# -------------------- Main Application Rendering -------------------

if app_requires_api_key_setup:
    # --- API Key Setup Page (Unchanged Functionally) ---
    st.set_page_config(page_title="OpenRouter API Key Setup", layout="centered")
    load_custom_css()
    st.title("🔒 OpenRouter API Key Required")
    st.markdown("---", unsafe_allow_html=True)
    # Display error/info messages (unchanged)
    if st.session_state.get("api_key_auth_failed"): st.error("API Key Authentication Failed. Please verify your key on OpenRouter.ai and re-enter.")
    elif not api_key_is_syntactically_valid and st.session_state.get("openrouter_api_key") is not None: st.error("The configured API Key has an invalid format. It must start with `sk-or-`.")
    elif not st.session_state.get("openrouter_api_key"): st.info("Please configure your OpenRouter API Key to use the application.")
    else: st.info("API Key configuration required.")

    st.markdown( "You can get a key from [OpenRouter.ai Keys](https://openrouter.ai/keys). Enter it below to continue." )
    new_key_input_val = st.text_input("Enter OpenRouter API Key", type="password", key="api_key_setup_input", value="", placeholder="sk-or-...")
    if st.button("Save and Validate API Key", key="save_api_key_setup_button", use_container_width=True, type="primary"):
        if is_api_key_valid(new_key_input_val):
            st.session_state.openrouter_api_key = new_key_input_val
            _save_app_config(new_key_input_val) # Saves to file
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
    st.markdown("---", unsafe_allow_html=True); st.caption(f"Your API key is stored locally in `{CONFIG_FILE.name}`.")

# --- Main App Logic (Uses Persistent State) ---
else:
    st.set_page_config(page_title="OpenRouter Chat", layout="wide", initial_sidebar_state="expanded")
    load_custom_css()

    # --- Initialize session state variables ---
    if "settings_panel_open" not in st.session_state: st.session_state.settings_panel_open = False
    if "credits" not in st.session_state: st.session_state.credits = {"total": None, "used": None, "remaining": None}; st.session_state.credits_ts = 0

    # --- Session ID Management (Using app_state['sessions']) ---
    # Ensure a valid session ID exists in st.session_state
    if "sid" not in st.session_state or st.session_state.sid not in sessions:
        # If no SID or invalid SID, try to find the latest valid session
        valid_sids = [s for s in sessions.keys() if isinstance(s, str) and s.isdigit()]
        if valid_sids:
            latest_sid = max(valid_sids, key=int)
            st.session_state.sid = latest_sid
            logging.info(f"Initialized session state with latest found SID: {latest_sid}")
        else:
            # If no valid sessions exist at all, create a new one
            st.session_state.sid = _new_sid() # This creates, marks dirty, and saves
            logging.info(f"No valid sessions found. Created new session: {st.session_state.sid}")
            st.rerun() # Rerun immediately after creating the very first session

    current_sid = st.session_state.sid # Use the validated/assigned SID

    # --- Credit Refresh Logic (Unchanged) ---
    credits_are_stale = time.time() - st.session_state.get("credits_ts", 0) > 3600
    credits_never_fetched = st.session_state.get("credits_ts", 0) == 0
    credits_are_none = any(st.session_state.credits.get(k) is None for k in ["total", "used", "remaining"])
    if credits_are_stale or credits_never_fetched or credits_are_none:
        logging.info(f"Refreshing credits (Stale: {credits_are_stale}, Never Fetched: {credits_never_fetched}, Are None: {credits_are_none}).")
        credits_data = get_credits()
        if st.session_state.get("api_key_auth_failed"): logging.error("API Key auth failed during scheduled credit refresh."); st.session_state.credits_ts = time.time()
        elif credits_data != (None, None, None): st.session_state.credits["total"], st.session_state.credits["used"], st.session_state.credits["remaining"] = credits_data; st.session_state.credits_ts = time.time(); logging.info("Credits refreshed successfully.")
        else: logging.warning("Scheduled credit refresh failed (non-auth)."); st.session_state.credits_ts = time.time()
        if "credits" not in st.session_state or not isinstance(st.session_state.credits, dict): st.session_state.credits = {"total": None, "used": None, "remaining": None}
        for k in ["total", "used", "remaining"]: st.session_state.credits.setdefault(k, None)


    # --- Sidebar Rendering (Reads from persistent state via helpers) ---
    with st.sidebar:
        settings_button_label = "⚙️ Close Settings" if st.session_state.settings_panel_open else "⚙️ Settings"
        if st.button(settings_button_label, key="toggle_settings_button_sidebar", use_container_width=True):
            st.session_state.settings_panel_open = not st.session_state.settings_panel_open; st.rerun()

        # --- Settings Panel (Unchanged Functionally) ---
        if st.session_state.get("settings_panel_open"):
            st.markdown("<div class='settings-panel'>", unsafe_allow_html=True)
            st.subheader("🔑 API Key Configuration")
            # API Key Display/Update (unchanged)
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
                    _save_app_config(new_key_input_sidebar) # Saves to file
                    st.session_state.api_key_auth_failed = False
                    with st.spinner("Validating new API key..."): credits_data = get_credits()
                    if st.session_state.get("api_key_auth_failed"): st.error("New API Key failed authentication.")
                    elif credits_data == (None,None,None): st.warning("Could not validate new API key (network/API issue?). Saved.")
                    else: st.success("New API Key saved and validated!"); st.session_state.credits["total"],st.session_state.credits["used"],st.session_state.credits["remaining"] = credits_data; st.session_state.credits_ts = time.time()
                    time.sleep(0.8); st.rerun()
                elif not new_key_input_sidebar: st.warning("API Key field empty.")
                else: st.error("Invalid API key format.")

            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("📊 Detailed Model Quotas")
            # Detailed Quota Display (reads from current state via helpers)
            # _ensure_quota_data_is_current() # Called implicitly by get_quota_usage_and_limits
            for m_key_loop in sorted(NEW_PLAN_CONFIG.keys()):
                if m_key_loop not in MODEL_MAP or m_key_loop not in EMOJI or m_key_loop not in MODEL_DESCRIPTIONS: continue
                stats = get_quota_usage_and_limits(m_key_loop)
                if not stats: st.markdown(f"**{EMOJI.get(m_key_loop, '')} {m_key_loop}**: Could not retrieve quota details."); continue
                model_desc_full = MODEL_DESCRIPTIONS.get(m_key_loop, "")
                try: model_short_name = model_desc_full.split('(')[1].split(')')[0] if '(' in model_desc_full and ')' in model_desc_full else MODEL_MAP.get(m_key_loop, "Unk").split('/')[-1]
                except IndexError: model_short_name = MODEL_MAP.get(m_key_loop, "Unk").split('/')[-1]
                model_name_display = f"{EMOJI.get(m_key_loop, '')} <span class='detailed-quota-modelname'>{m_key_loop} ({model_short_name})</span>"
                st.markdown(f"{model_name_display}", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                def format_limit_display(limit_val): return format_token_count(limit_val) if limit_val != float('inf') else '∞'
                with col1:
                    st.markdown(f"""<div class="detailed-quota-block"><ul>
                        <li><b>Daily Msgs:</b> {stats['used_daily_msg']}/{format_limit_display(stats['limit_daily_msg'])}</li>
                        <li><b>Daily In Tok:</b> {format_token_count(stats['used_daily_in_tokens'])}/{format_limit_display(stats['limit_daily_in_tokens'])}</li>
                        <li><b>Daily Out Tok:</b> {format_token_count(stats['used_daily_out_tokens'])}/{format_limit_display(stats['limit_daily_out_tokens'])}</li>
                    </ul></div>""", unsafe_allow_html=True)
                with col2:
                     st.markdown(f"""<div class="detailed-quota-block"><ul>
                        <li><b>Monthly Msgs:</b> {stats['used_monthly_msg']}/{format_limit_display(stats['limit_monthly_msg'])}</li>
                        <li><b>Monthly In Tok:</b> {format_token_count(stats['used_monthly_in_tokens'])}/{format_limit_display(stats['limit_monthly_in_tokens'])}</li>
                        <li><b>Monthly Out Tok:</b> {format_token_count(stats['used_monthly_out_tokens'])}/{format_limit_display(stats['limit_monthly_out_tokens'])}</li>
                    </ul></div>""", unsafe_allow_html=True)
                if m_key_loop == "A" and stats["limit_3hr_msg"] != float('inf'):
                    time_until_next_msg_str = ""
                    # Use _ensure_quota_data_is_current() to prune and read from memory
                    _ensure_quota_data_is_current() # Ensure list is pruned
                    active_model_a_calls = sorted(app_state["quota_data"].get(MODEL_A_3H_CALLS_KEY, []))
                    limit_3hr_int = int(stats['limit_3hr_msg']) # Convert inf to int for comparison if needed
                    if len(active_model_a_calls) >= limit_3hr_int:
                         if active_model_a_calls:
                             oldest_blocking_call_idx = max(0, len(active_model_a_calls) - limit_3hr_int)
                             oldest_blocking_call_ts = active_model_a_calls[oldest_blocking_call_idx]
                             if "A" in NEW_PLAN_CONFIG and len(NEW_PLAN_CONFIG["A"]) > 7 and NEW_PLAN_CONFIG["A"][7] > 0:
                                 expiry_time = oldest_blocking_call_ts + NEW_PLAN_CONFIG["A"][7]
                                 time_remaining_seconds = expiry_time - time.time()
                                 if time_remaining_seconds > 0:
                                    mins, secs = divmod(int(time_remaining_seconds), 60); hrs, mins_rem = divmod(mins, 60)
                                    if hrs > 0: time_until_next_msg_str = f" (Next in {hrs}h {mins_rem}m)"
                                    elif mins_rem > 0: time_until_next_msg_str = f" (Next in {mins_rem}m {secs}s)"
                                    else: time_until_next_msg_str = f" (Next in {secs}s)"
                    st.markdown(f"""<div class="detailed-quota-block" style="margin-top: -0.5rem; margin-left:0.1rem;"><ul>
                        <li><b>3-Hour Msgs:</b> {stats['used_3hr_msg']}/{limit_3hr_int}{time_until_next_msg_str}</li></ul>
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

        # --- Daily Quota Gauges (Reads from current state) ---
        with st.expander("⚡ DAILY MODEL QUOTAS", expanded=True):
            active_model_keys_for_display = sorted([k for k in NEW_PLAN_CONFIG.keys() if k in MODEL_MAP and k in EMOJI])
            if not active_model_keys_for_display: st.caption("No fully configured models.")
            else:
                # _ensure_quota_data_is_current() # Called by get_remaining_daily_messages
                num_models = len(active_model_keys_for_display); num_cols = min(num_models, 7)
                quota_cols = st.columns(num_cols)
                for i, m_key in enumerate(active_model_keys_for_display):
                    with quota_cols[i % num_cols]:
                        stats = get_quota_usage_and_limits(m_key) # Ensures data is current
                        left_d_msgs = get_remaining_daily_messages(m_key)
                        lim_d_msgs = stats.get("limit_daily_msg", 0) # Get limit from stats

                        if lim_d_msgs == float('inf'): pct_float, fill_width_val, left_display, bar_color = 1.0, 100, "∞", "var(--app-success-color)"
                        elif lim_d_msgs <= 0: pct_float, fill_width_val, left_display, bar_color = 1.0, 100, "0", "var(--app-danger-color)" # Show 0 limit as red/empty
                        else:
                            pct_float = max(0.0, min(1.0, left_d_msgs / lim_d_msgs))
                            fill_width_val = int(pct_float * 100)
                            left_display = str(int(left_d_msgs))
                            if pct_float > 0.5: bar_color = "var(--app-success-color)"
                            elif pct_float > 0.15: bar_color = "var(--app-warning-color)"
                            else: bar_color = "var(--app-danger-color)"

                        emoji_char = EMOJI.get(m_key, "❔")
                        limit_display_tt = '∞' if lim_d_msgs == float('inf') else str(int(lim_d_msgs)) if lim_d_msgs > 0 else '0'
                        tooltip_text = f"{left_display} / {limit_display_tt} Daily Msgs Left"
                        st.markdown(f"""<div class="compact-quota-item" title="{tooltip_text}">
                                            <div class="cq-info">{emoji_char} <b>{m_key}</b></div>
                                            <div class="cq-bar-track"><div class="cq-bar-fill" style="width: {fill_width_val}%; background-color: {bar_color};"></div></div>
                                            <div class="cq-value" style="color: {bar_color};">{left_display}</div>
                                        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

        # --- New Chat Button (Uses persistent state logic) ---
        current_session_data = sessions.get(current_sid, {})
        current_session_is_truly_blank = (current_session_data.get("title") == "New chat" and
                                          not current_session_data.get("messages"))
        if st.button("➕ New chat", key="new_chat_button_top", use_container_width=True, disabled=current_session_is_truly_blank):
            new_session_id = _new_sid() # Creates, marks dirty, saves state
            st.session_state.sid = new_session_id # Update session_state variable
            st.rerun()

        # --- Chat History List (Reads from persistent state) ---
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
                    current_session_data_before_switch = sessions.get(st.session_state.sid, {})
                    current_session_was_blank = (current_session_data_before_switch.get("title") == "New chat" and not current_session_data_before_switch.get("messages"))
                    if not current_session_was_blank:
                         if _delete_unused_blank_sessions(keep_sid=sid_key): # Check if deletion happened
                              save_app_state() # Save if sessions were modified
                    st.session_state.sid = sid_key
                    st.rerun()

        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

        # --- Model Info (Unchanged) ---
        st.subheader("Model Info & Costs")
        st.caption(f"Router: {ROUTER_MODEL_ID.split('/')[-1]}")
        cost_order_display = ["F", "G", "D", "B", "A", "H", "I", "E", "C"]
        cost_order_title = " < ".join(cost_order_display)
        with st.expander(f"Cost Order: {cost_order_title}", expanded=False):
            for k_model in cost_order_display:
                 if k_model not in MODEL_MAP or k_model not in MODEL_DESCRIPTIONS or k_model not in EMOJI: continue
                 desc_full = MODEL_DESCRIPTIONS.get(k_model, MODEL_MAP.get(k_model, "N/A"))
                 try:
                     desc_parts = desc_full.split("(")
                     main_desc = desc_parts[0].strip()
                     model_name_in_desc = desc_parts[1].split(")")[0] if len(desc_parts) > 1 and ')' in desc_parts[1] else MODEL_MAP.get(k_model, "N/A").split('/')[-1]
                 except IndexError: main_desc = desc_full; model_name_in_desc = MODEL_MAP.get(k_model, "N/A").split('/')[-1]
                 max_tok = MAX_TOKENS.get(k_model, 0); emoji_char = EMOJI.get(k_model, '')
                 st.markdown(f"**{emoji_char} {k_model}**: {main_desc} ({model_name_in_desc}) <br><small style='color:var(--app-text-secondary-color);'>Max Output: {format_token_count(max_tok)} tokens</small>", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(f"**{FALLBACK_MODEL_KEY}**: {FALLBACK_MODEL_EMOJI} {FALLBACK_MODEL_ID.split('/')[-1]} <br><small style='color:var(--app-text-secondary-color);'>Max Output: {format_token_count(FALLBACK_MODEL_MAX_TOKENS)} tokens (Free)</small>", unsafe_allow_html=True)

        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

        # --- Account Credits (Unchanged Functionally) ---
        with st.expander("Account stats (credits)", expanded=False):
            if st.button("Refresh Credits", key="refresh_credits_button_sidebar"):
                 with st.spinner("Refreshing credits..."): credits_data = get_credits()
                 if st.session_state.get("api_key_auth_failed"): st.error("API Key authentication failed.")
                 elif credits_data != (None,None,None):
                     st.session_state.credits["total"],st.session_state.credits["used"],st.session_state.credits["remaining"] = credits_data; st.session_state.credits_ts = time.time(); st.success("Credits refreshed!"); st.rerun()
                 else: st.warning("Could not refresh credits (network/API issue?).")
            tot, used, rem = st.session_state.credits.get("total"), st.session_state.credits.get("used"), st.session_state.credits.get("remaining")
            if st.session_state.get("api_key_auth_failed"): st.warning("Cannot display credits. API Key failed.")
            elif tot is None or used is None or rem is None: st.warning("Could not fetch/display credits.")
            else:
                 try: rem_f = f"${float(rem):.2f} cr"
                 except (ValueError, TypeError): rem_f = "N/A"
                 try: used_f = f"${float(used):.2f} cr"
                 except (ValueError, TypeError): used_f = "N/A"
                 st.markdown(f"**Remaining:** {rem_f}<br>**Used:** {used_f}", unsafe_allow_html=True)
            ts = st.session_state.get("credits_ts", 0); last_updated_str = datetime.fromtimestamp(ts, TZ).strftime('%-d %b, %H:%M') if ts else "Never"; st.caption(f"Last updated: {last_updated_str}")

    # ---- Main chat area (Uses persistent state) ----
    if current_sid not in sessions:
        logging.error(f"CRITICAL: Current SID {current_sid} missing from in-memory sessions. Resetting.")
        st.session_state.sid = _new_sid() # Create new, save state
        st.rerun()

    # Ensure messages list exists for the current session in memory
    if "messages" not in sessions[current_sid] or not isinstance(sessions[current_sid]["messages"], list):
         sessions[current_sid]["messages"] = []
         app_state["sessions_dirty"] = True # Mark dirty
         logging.warning(f"Initialized missing/invalid 'messages' list for session {current_sid}.")
         save_app_state() # Save the fix immediately

    chat_history = sessions[current_sid]["messages"]

    # --- Display Existing Chat Messages ---
    for msg_idx, msg in enumerate(chat_history):
        role = msg.get("role", "assistant"); avatar_char = None
        if role == "user": avatar_char = "👤"
        elif role == "assistant":
            m_key = msg.get("model")
            if m_key == FALLBACK_MODEL_KEY: avatar_char = FALLBACK_MODEL_EMOJI
            elif m_key in EMOJI: avatar_char = EMOJI[m_key]
            else: avatar_char = "🤖"
        else: role="assistant"; avatar_char = "⚙️"
        with st.chat_message(role, avatar=avatar_char): st.markdown(msg.get("content", "*empty message*"))

    # --- Chat Input Logic (Saves state after processing) ---
    if prompt := st.chat_input("Ask anything… (use 'search', 'latest', 'investigate' for web access)", key=f"chat_input_{current_sid}"):
        # Append user message to in-memory history and mark dirty
        user_message = {"role":"user","content":prompt}
        chat_history.append(user_message)
        app_state["sessions_dirty"] = True
        with st.chat_message("user", avatar="👤"): st.markdown(prompt)
        # Don't save yet, save after assistant response

        # API Key Check (unchanged)
        if st.session_state.get("api_key_auth_failed") or not is_api_key_valid(st.session_state.get("openrouter_api_key")):
            st.error("OpenRouter API Key is invalid or failed authentication. Please fix in ⚙️ Settings.")
            save_app_state() # Save the user message before stopping
            st.stop()

        # --- Model Selection Logic ---
        routing_start_time = time.time()
        with st.spinner("Selecting best model..."):
            # _ensure_quota_data_is_current() # Called by is_model_available/route_choice
            logging.info("--- Checking Model Availability Before Routing ---")
            all_possible_models = [k for k in NEW_PLAN_CONFIG.keys() if k in MODEL_MAP and k in EMOJI]
            allowed_models_for_router = []
            availability_log = []
            for k_map in sorted(all_possible_models):
                 available = is_model_available(k_map) # Checks against in-memory state
                 availability_log.append(f"Model {k_map} ({MODEL_MAP.get(k_map,'?').split('/')[-1]}): Available = {available}")
                 if available: allowed_models_for_router.append(k_map)
            logging.info("\n".join(availability_log))
            logging.info(f"Final allowed models passed to router function: {allowed_models_for_router}")

            # Routing call (unchanged)
            routed_key_letter = route_choice(prompt, allowed_models_for_router, chat_history)
            routing_end_time = time.time(); logging.info(f"Routing took {routing_end_time - routing_start_time:.2f}s. Router decided: '{routed_key_letter}'")

            # Handle router failure due to auth (unchanged)
            if routed_key_letter is None and st.session_state.get("api_key_auth_failed"):
                 st.error("API Authentication failed during model routing. Please check the key in ⚙️ Settings.")
                 save_app_state() # Save user msg
                 st.stop()

            # Process router decision (unchanged logic, determine model_id_to_use etc.)
            use_fallback = False; chosen_model_key = None; model_id_to_use = None
            max_tokens_api = FALLBACK_MODEL_MAX_TOKENS; avatar_resp = "🤖"
            if routed_key_letter == FALLBACK_MODEL_KEY:
                use_fallback=True; chosen_model_key=FALLBACK_MODEL_KEY; model_id_to_use=FALLBACK_MODEL_ID; avatar_resp=FALLBACK_MODEL_EMOJI
                logging.info(f"Using free fallback: {FALLBACK_MODEL_ID}.")
            elif routed_key_letter in MODEL_MAP:
                if is_model_available(routed_key_letter): # Check availability again (quick check)
                    chosen_model_key=routed_key_letter; model_id_to_use=MODEL_MAP[chosen_model_key]; max_tokens_api=MAX_TOKENS.get(chosen_model_key, FALLBACK_MODEL_MAX_TOKENS); avatar_resp=EMOJI.get(chosen_model_key, "🤖")
                    logging.info(f"Using router-selected model: {chosen_model_key} ({model_id_to_use})")
                else:
                    use_fallback=True; chosen_model_key=FALLBACK_MODEL_KEY; model_id_to_use=FALLBACK_MODEL_ID; avatar_resp=FALLBACK_MODEL_EMOJI
                    logging.warning(f"Router chose '{routed_key_letter}', but unavailable. Using free fallback.")
            else: # Unexpected router response
                 use_fallback=True; chosen_model_key=FALLBACK_MODEL_KEY; model_id_to_use=FALLBACK_MODEL_ID; avatar_resp=FALLBACK_MODEL_EMOJI
                 logging.error(f"Router returned unexpected key '{routed_key_letter}'. Using free fallback.")


            # --- API Call and Response Streaming ---
            if model_id_to_use:
                with st.chat_message("assistant", avatar=avatar_resp):
                    response_placeholder = st.empty(); response_placeholder.markdown("Thinking... 💭")
                    full_response = ""; api_call_ok = True; error_message_from_stream = None
                    stream_start_time = time.time()
                    # Streaming (unchanged)
                    for chunk_content, err_msg in streamed(model_id_to_use, chat_history, max_tokens_api):
                        if err_msg:
                            logging.error(f"Stream error ({model_id_to_use}): {err_msg}")
                            if st.session_state.get("api_key_auth_failed"): error_message_from_stream = "❗ **API Auth Error**"
                            elif "rate limit" in err_msg.lower(): error_message_from_stream = f"❗ **Rate Limit** ({model_id_to_use})"
                            elif "context_length_exceeded" in err_msg.lower(): error_message_from_stream = f"❗ **Context Too Long** ({model_id_to_use})"
                            elif "perplexity" in model_id_to_use.lower() and "search query failed" in err_msg.lower(): error_message_from_stream = f"❗ **Search Failed** ({model_id_to_use})"
                            else: error_message_from_stream = f"❗ **API Error**: {err_msg}"
                            api_call_ok = False; break
                        if chunk_content is not None: full_response += chunk_content; response_placeholder.markdown(full_response + "▌")
                    stream_end_time = time.time(); logging.info(f"Streaming took {stream_end_time - stream_start_time:.2f}s.")
                    # Final display (unchanged)
                    if error_message_from_stream: response_placeholder.markdown(error_message_from_stream); full_response = error_message_from_stream
                    elif not full_response and api_call_ok: response_placeholder.markdown("*Assistant returned empty response.*"); full_response = ""; logging.warning(f"{model_id_to_use} returned empty.")
                    else: response_placeholder.markdown(full_response)

                # --- Post-Response Processing (Saves state) ---
                last_usage = st.session_state.pop("last_stream_usage", None)
                prompt_tokens_used = 0; completion_tokens_used = 0
                if api_call_ok and last_usage:
                    prompt_tokens_used = last_usage.get("prompt_tokens", 0)
                    completion_tokens_used = last_usage.get("completion_tokens", 0) if full_response and not error_message_from_stream else 0
                    logging.info(f"API call OK ({model_id_to_use}). Usage: P={prompt_tokens_used}, C={completion_tokens_used}")
                elif api_call_ok: logging.warning(f"Token usage info missing for {model_id_to_use}.")

                # Append assistant message to in-memory history
                assistant_message = {"role": "assistant", "content": full_response, "model": chosen_model_key, "prompt_tokens": prompt_tokens_used, "completion_tokens": completion_tokens_used}
                chat_history.append(assistant_message)
                app_state["sessions_dirty"] = True # Mark session dirty

                # Record quota usage (this marks quota dirty and saves state)
                if api_call_ok and not use_fallback and chosen_model_key in NEW_PLAN_CONFIG and (prompt_tokens_used > 0 or completion_tokens_used > 0):
                    record_use(chosen_model_key, prompt_tokens_used, completion_tokens_used) # This saves state
                elif api_call_ok and not use_fallback: logging.info(f"Skipping quota recording for {chosen_model_key} (zero tokens).")

                # Auto-title if new and successful
                if api_call_ok and not error_message_from_stream and full_response and sessions[current_sid]["title"] == "New chat":
                   sessions[current_sid]["title"] = _autoname(prompt)
                   app_state["sessions_dirty"] = True # Mark dirty if title changed

                # Ensure state is saved *after* all updates for this interaction
                save_app_state()
                st.rerun() # Rerun to refresh UI

            else: # Handle case where no model_id_to_use was determined
                 if not st.session_state.get("api_key_auth_failed"):
                    st.error("Unexpected error: Could not determine model.")
                    logging.error("Failed to determine model_id_to_use, no API auth failure.")
                 save_app_state() # Save history up to this point
                 st.stop()
