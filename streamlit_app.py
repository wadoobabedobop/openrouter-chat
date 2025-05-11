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
from datetime import datetime, date
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

    # Determine a sensible fallback choice early.
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

    # --- Detailed guidance for the router model (internal) ---
    # These are more prescriptive than the user-facing MODEL_DESCRIPTIONS
    ROUTER_MODEL_GUIDANCE = {
        "A": "gemini-2.5-pro-preview (Model A - Highest Cost/Quality): Use for EXTREMELY complex, multi-step reasoning; highly advanced creative generation (e.g., novel excerpts, sophisticated poetry); tasks demanding cutting-edge knowledge and deep nuanced understanding. CHOOSE ONLY if query explicitly demands top-tier, 'genius-level' output AND cheaper models are CLEARLY insufficient. Avoid for anything less.",
        "B": "o4-mini (Model B - Mid-Tier Cost/Quality): Use for general purpose chat; moderate complexity reasoning; summarization; drafting emails/content; brainstorming; standard instruction following. A good balance of capability and cost. CHOOSE if 'F' or 'D' are too basic, AND 'A' or 'C' are overkill/not strictly necessary for the task's core requirements.",
        "C": "chatgpt-4o-latest (Model C - High Cost/Quality, Polished): Use for tasks requiring highly polished, empathetic, or very human-like conversational interactions; complex multi-turn instruction adherence where its specific stylistic strengths are key; creative content generation with a defined sophisticated tone. CHOOSE ONLY if query *specifically* benefits from its unique interaction style or demands exceptional refinement AND 'B' is clearly inadequate. More expensive than B.",
        "D": "deepseek-r1 (Model D - Low Cost, Factual/Technical): Use for factual Q&A; code generation/explanation/debugging; data extraction; straightforward logical reasoning; technical or scientific queries. Very cost-effective. CHOOSE for tasks that are well-defined, benefit from specialized reasoning, and do not require broad world knowledge, deep creativity, or nuanced conversation. Prefer over B for these specific tasks if cost is a factor.",
        "F": "gemini-2.5-flash-preview (Model F - Lowest Cost, Quick/Simple): Use for very quick, simple Q&A; fast summarization of short texts; basic classification; brief translations; or when speed is paramount and task complexity is very low. Most cost-effective general model. Default starting point for most simple requests."
    }

    system_prompt_parts = [
        "You are an expert AI model routing assistant. Your task is to select the *single most appropriate and cost-effective* model letter from the 'Available Models' list to handle the given 'User Query'.",
        "Strictly adhere to these decision-making principles in order of importance:",
        "1. Maximize Cost-Effectiveness: This is your PRIMARY GOAL. Always prefer a cheaper model (F > D > B > C > A in general cost order) if it can adequately perform the task. Do NOT select expensive models (A, C) unless explicitly justified by the query's extreme complexity and specific requirements that cheaper models demonstrably cannot meet.",
        "2. Analyze User Query Intent: Deeply understand what the user is trying to achieve, the complexity involved (simple, moderate, high, extreme), the desired output style (factual, creative, conversational), and any implicit needs.",
        "3. Match to Model Strengths and Weaknesses as described below."
    ]

    system_prompt_parts.append("\nAvailable Models (select one letter):")
    for k_model_key in allowed:
        # Use the detailed ROUTER_MODEL_GUIDANCE. Fallback to a generic message if a model isn't in our detailed guide.
        description = ROUTER_MODEL_GUIDANCE.get(k_model_key, f"({MODEL_MAP.get(k_model_key, 'Unknown Model')}) - General purpose. Evaluate based on query and cost guidance.")
        system_prompt_parts.append(f"- {k_model_key}: {description}")

    system_prompt_parts.append("\nSpecific Selection Guidance (apply rigorously):")
    if "F" in allowed:
        system_prompt_parts.append("  - If 'F' is available AND the query is simple (e.g., basic factual question, quick definition, short summary of <200 words, simple classification), CHOOSE 'F'. This is your first consideration for low-complexity tasks.")
    if "D" in allowed:
        system_prompt_parts.append("  - If 'D' is available AND the query is primarily factual, technical, code-related (generation, debugging, explanation), or requires straightforward logical deduction, AND 'F' (if available) is too basic for the detail required, STRONGLY PREFER 'D'. 'D' is very cost-effective for these domains.")
    if "B" in allowed:
        system_prompt_parts.append("  - If 'B' is available, AND 'F'/'D' (if available) are insufficient for the task's general reasoning, drafting, or moderate creative needs (e.g., writing a standard email, brainstorming ideas, moderately complex Q&A not fitting D's specialty), 'B' is a good general-purpose choice. Prefer 'B' over 'A'/'C' if peak quality/style isn't explicitly demanded or implied by extreme complexity.")

    system_prompt_parts.append("\n  - Guidance for Expensive Models (A, C) - Use Sparingly:")
    if "C" in allowed:
        system_prompt_parts.append("    - CHOOSE 'C' (4o) ONLY if the query *explicitly requires or strongly implies a need for* a highly polished, empathetic, human-like conversational tone, or involves nuanced, multi-turn creative collaboration where its specific stylistic strengths are indispensable (e.g., role-playing as a character, writing a very personal letter), AND 'B' (if available) is clearly inadequate for this specific stylistic requirement. Consider if the query mentions 'tone', 'style', 'empathy', or 'conversation partner'.")
    if "A" in allowed:
        system_prompt_parts.append("    - CHOOSE 'A' (Pro) ONLY if the query involves *exceptionally* complex, multi-layered reasoning (e.g., detailed scientific analysis, philosophical debate, complex strategic planning), requires generation of extensive, high-stakes creative content (like writing a detailed story plot, a research paper outline, or sophisticated poetry), or tasks demanding the absolute frontier of AI capability that *no other available model can credibly handle*. The query should signal 'deep', 'comprehensive', 'expert-level', 'highly creative', or 'groundbreaking'.")

    system_prompt_parts.append("\nExample Query Analysis (Illustrative - your judgment is key):")
    system_prompt_parts.append("  - User Query: 'What's the weather in London?' -> If F available, choose F. Else if D available, choose D.")
    system_prompt_parts.append("  - User Query: 'Write a python function to calculate factorial.' -> If D available, choose D. Else if B available, B.")
    system_prompt_parts.append("  - User Query: 'Summarize this article [long article text]' -> If F too simple for length, B is good. Avoid A/C.")
    system_prompt_parts.append("  - User Query: 'I'm feeling down, can you cheer me up?' -> If C available, C is a strong candidate due to empathy. If not, B. Avoid A/D.")
    system_prompt_parts.append("  - User Query: 'Generate a 1000-word short story about a space explorer finding a new planet, focus on detailed world-building.' -> This is more complex. B might handle it. If 'highly detailed and award-winning quality' is implied, A could be considered if B is known to be weak here. C less likely unless a specific narrative style is requested.")
    system_prompt_parts.append("  - User Query: 'Devise a comprehensive marketing strategy for launching a new eco-friendly toothbrush, including target audience, channels, and budget allocation for 3 months.' -> This is complex and strategic. If A is available, A is a strong candidate. C might be if a very persuasive/polished document is needed. B is a less optimal fallback.")

    system_prompt_parts.append("\nUser Query will be provided next.")
    system_prompt_parts.append("Respond with ONLY the single capital letter of your chosen model (e.g., A, B, C, D, or F). NO EXPLANATION, NO EXTRA TEXT, JUST THE LETTER.")

    final_system_message = "\n".join(system_prompt_parts)

    router_messages = [
        {"role": "system", "content": final_system_message},
        {"role": "user", "content": f"User Query: \"{user_msg}\""} # Clearly label the user query
    ]
    # Low temperature for more deterministic routing
    payload_r = {"model": ROUTER_MODEL_ID, "messages": router_messages, "max_tokens": 10, "temperature": 0.1}

    try:
        r = api_post(payload_r) # api_post handles 401 by setting flag and re-raising
        choice_data = r.json()
        # Extract the message content; be robust to potential variations
        raw_text_response = choice_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()

        logging.info(f"Router raw response: '{raw_text_response}' for query: '{user_msg}'")
        # For debugging, you might want to log the full prompt sent to the router:
        # logging.debug(f"Router system prompt for query '{user_msg}':\n{final_system_message}")

        # Iterate through the response to find the first valid model letter
        # This handles cases where the model might output "Model A" or "A."
        chosen_model_letter = None
        for char_in_response in raw_text_response:
            if char_in_response in allowed:
                chosen_model_letter = char_in_response
                break

        if chosen_model_letter:
            logging.info(f"Router selected model: '{chosen_model_letter}'")
            return chosen_model_letter
        else:
            # If no valid character found in the (potentially longer) response
            logging.warning(f"Router returned a response ('{raw_text_response}') that did not contain any allowed model letters. Falling back to: {fallback_choice}")
            return fallback_choice

    except ValueError as ve: # Catch API key not found/invalid from api_post
        logging.error(f"ValueError during router call (likely API key issue): {ve}")
    except requests.exceptions.HTTPError as e:
         # Flag st.session_state.api_key_auth_failed is set in api_post if 401
        logging.error(f"Router call HTTPError {e.response.status_code}: {e.response.text}")
    except (KeyError, IndexError, AttributeError, json.JSONDecodeError) as je: # Catch issues with JSON structure or decoding
        response_text_for_log = "N/A"
        if 'r' in locals() and hasattr(r, 'text'):
            response_text_for_log = r.text
        logging.error(f"Router call JSON parsing/structure error: {je}. Raw Response: {response_text_for_log}")
    except Exception as e:
        logging.error(f"Router call unexpected error: {e}")

    logging.warning(f"Router failed or returned invalid. Falling back to model: {fallback_choice}")
    return fallback_choice

# --------------------- Credits Endpoint -----------------------
def get_credits():
    """Returns (total, used, remaining) or (None, None, None) on failure."""
    active_api_key = st.session_state.get("openrouter_api_key")
    if not is_api_key_valid(active_api_key):
        logging.warning("get_credits: API Key is not syntactically valid or not set.")
        return None, None, None
    try:
        r = requests.get(
            f"{OPENROUTER_API_BASE}/credits",
            headers={"Authorization": f"Bearer {active_api_key}"},
            timeout=10
        )
        r.raise_for_status()
        d = r.json()["data"]
        st.session_state.api_key_auth_failed = False
        return d["total_credits"], d["total_usage"], d["total_credits"] - d["total_usage"]
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        err_text = e.response.text
        if status_code == 401:
            st.session_state.api_key_auth_failed = True
            logging.warning(f"Could not fetch /credits: HTTP {status_code} Unauthorized. {err_text}")
        else:
             logging.warning(f"Could not fetch /credits: HTTP {status_code}. {err_text}")
        return None, None, None
    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
        logging.warning(f"Could not fetch /credits due to network/parsing error: {e}")
        return None, None, None

# ------------------------- UI Styling --------------------------
def load_custom_css():
    css = """
    <style>
        /* --- General & Variables --- */
        :root {
            --border-radius-sm: 4px;
            --border-radius-md: 8px;
            --border-radius-lg: 12px;
            --spacing-sm: 0.5rem;
            --spacing-md: 1rem;
            --spacing-lg: 1.5rem;
            --shadow-light: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
            /* --divider-color will be inherited from Streamlit's theme or defaults */
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
        }

        /* --- Sidebar Styling --- */
        [data-testid="stSidebar"] {
            background-color: var(--secondaryBackgroundColor); /* Use theme variable */
            padding: var(--spacing-lg) var(--spacing-md);
            border-right: 1px solid var(--divider-color, #262730); /* Add a subtle border */
        }
        [data-testid="stSidebar"] .stImage > img { /* Sidebar Logo */
            border-radius: 50%;
            box-shadow: var(--shadow-light);
            width: 48px !important; height: 48px !important;
            margin-right: var(--spacing-sm);
        }
        [data-testid="stSidebar"] h1 { /* Sidebar Title */
            font-size: 1.5rem !important; color: var(--primaryColor); /* Theme variable */
            font-weight: 600; margin-bottom: 0;
            padding-top: 0.2rem;
        }
        [data-testid="stSidebar"] .stButton > button { /* General Sidebar Buttons */
            border-radius: var(--border-radius-md);
            border: 1px solid var(--divider-color, #333);
            padding: 0.6em 1em; font-size: 0.9em;
            background-color: transparent;
            color: var(--textColor);
            transition: background-color 0.2s, border-color 0.2s;
            width: 100%; margin-bottom: var(--spacing-sm); text-align: left;
            font-weight: 500;
        }
        [data-testid="stSidebar"] .stButton > button:hover:not(:disabled) { /* Don't apply hover to disabled (active) */
            border-color: var(--primaryColor);
            background-color: color-mix(in srgb, var(--primaryColor) 15%, transparent);
        }
        /* Style for active (disabled) chat buttons */
        [data-testid="stSidebar"] .stButton > button:disabled {
            opacity: 1.0; /* Ensure it's fully visible */
            cursor: default; /* Default cursor for active item */
            background-color: color-mix(in srgb, var(--primaryColor) 25%, transparent) !important;
            border-left: 3px solid var(--primaryColor) !important;
            border-top-color: var(--divider-color, #333) !important; /* Keep other borders consistent */
            border-right-color: var(--divider-color, #333) !important;
            border-bottom-color: var(--divider-color, #333) !important;
            font-weight: 600;
            color: var(--textColor); /* Ensure text color is not dimmed */
        }

        /* Specific "New Chat" button */
        [data-testid="stSidebar"] [data-testid="stButton-new_chat_button_top"] > button {
            background-color: var(--primaryColor); color: white;
            border-color: var(--primaryColor);
            font-weight: 600;
        }
        [data-testid="stSidebar"] [data-testid="stButton-new_chat_button_top"] > button:hover {
            background-color: color-mix(in srgb, var(--primaryColor) 85%, black);
            border-color: color-mix(in srgb, var(--primaryColor) 85%, black);
        }
        /* Disabled state for "New Chat" specifically if it's truly blank */
        [data-testid="stSidebar"] [data-testid="stButton-new_chat_button_top"] > button:disabled {
            background-color: var(--primaryColor) !important; /* Keep its primary color */
            color: white !important;
            border-color: var(--primaryColor) !important;
            opacity: 0.6 !important; /* Dim it slightly like other disabled buttons */
            cursor: not-allowed !important;
            border-left: 1px solid var(--primaryColor) !important; /* Reset active chat border */
        }


        /* Sidebar Subheaders (e.g., "CHATS", "MODEL-ROUTING MAP") */
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] .stSubheader {
            font-size: 0.8rem !important; text-transform: uppercase; font-weight: 700;
            color: var(--text-color-secondary, #A0A0A0);
            margin-top: var(--spacing-lg); margin-bottom: var(--spacing-sm);
            letter-spacing: 0.05em;
        }

        /* --- Sidebar Expanders --- */
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            border: 1px solid var(--divider-color, #262730);
            border-radius: var(--border-radius-md);
            background-color: transparent; /* Make it blend better, less boxy */
            margin-bottom: var(--spacing-md);
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary {
            padding: 0.6rem var(--spacing-md) !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            color: var(--textColor) !important;
            border-bottom: 1px solid var(--divider-color, #262730);
            border-top-left-radius: var(--border-radius-md); /* Match expander border */
            border-top-right-radius: var(--border-radius-md);
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
            background-color: color-mix(in srgb, var(--textColor) 5%, transparent);
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
            padding: var(--spacing-sm) var(--spacing-md) !important;
            background-color: var(--secondaryBackgroundColor); /* Content area slightly different */
            border-bottom-left-radius: var(--border-radius-md);
            border-bottom-right-radius: var(--border-radius-md);
        }
        /* Quota items specific padding */
        [data-testid="stSidebar"] [data-testid="stExpander"][aria-label^="⚡ DAILY MODEL QUOTAS"] div[data-testid="stExpanderDetails"] {
            padding-top: 0.6rem !important;
            padding-bottom: 0.2rem !important;
            padding-left: 0.1rem !important;
            padding-right: 0.1rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"][aria-label^="⚡ DAILY MODEL QUOTAS"] div[data-testid="stHorizontalBlock"] {
            gap: 0.25rem !important;
        }

        /* Compact Quota Bar Styling */
        .compact-quota-item {
            display: flex; flex-direction: column; align-items: center;
            text-align: center; padding: 0px 4px;
        }
        .cq-info {
            font-size: 0.7rem; margin-bottom: 3px; line-height: 1.1;
            white-space: nowrap; color: var(--textColor);
        }
        .cq-bar-track {
            width: 100%; height: 8px;
            background-color: color-mix(in srgb, var(--textColor) 10%, transparent);
            border: 1px solid var(--divider-color, #333);
            border-radius: var(--border-radius-sm); overflow: hidden; margin-bottom: 5px; /* Increased space below bar */
        }
        .cq-bar-fill {
            height: 100%; border-radius: var(--border-radius-sm);
            transition: width 0.3s ease-in-out, background-color 0.3s ease-in-out;
        }
        .cq-value { font-size: 0.7rem; font-weight: bold; line-height: 1; }

        /* --- Settings Panel (in Sidebar) --- */
        .settings-panel { /* This is the div wrapper in Python, not an expander */
            border: 1px solid var(--divider-color, #333);
            border-radius: var(--border-radius-md);
            padding: var(--spacing-md);
            margin-top: var(--spacing-sm); margin-bottom: var(--spacing-md);
            background-color: color-mix(in srgb, var(--backgroundColor) 50%, var(--secondaryBackgroundColor));
        }
        .settings-panel .stTextInput input {
            border-color: var(--divider-color, #444) !important;
        }

        /* --- Main Chat Area Styling --- */
        [data-testid="stChatInputContainer"] { /* Target the container of the chat input */
            background-color: var(--secondaryBackgroundColor);
            border-top: 1px solid var(--divider-color, #262730);
            padding: var(--spacing-sm) var(--spacing-md); /* Add some padding around the input itself */
        }
        [data-testid="stChatInput"] textarea { /* Actual input field (is a textarea) */
            border-color: var(--divider-color, #444) !important;
            border-radius: var(--border-radius-md) !important;
            background-color: var(--backgroundColor) !important; /* Match app background or slightly lighter */
            color: var(--textColor) !important;
        }
        [data-testid="stChatMessage"] { /* Chat Message Bubbles */
            border-radius: var(--border-radius-lg);
            padding: var(--spacing-md) 1.25rem;
            margin-bottom: var(--spacing-md);
            box-shadow: var(--shadow-light);
            border: 1px solid transparent;
            max-width: 85%; /* Prevent bubbles from taking full width */
        }
        /* User Message */
        [data-testid="stChatMessage"][data-testid^="stChatMessageUser"] {
            background-color: var(--primaryColor);
            color: white;
            margin-left: auto; /* Align user messages to the right */
            border-top-right-radius: var(--border-radius-sm);
        }
        /* Assistant Message */
        [data-testid="stChatMessage"][data-testid^="stChatMessageAssistant"] {
            background-color: var(--secondaryBackgroundColor);
            color: var(--textColor);
            margin-right: auto; /* Align assistant messages to the left */
            border-top-left-radius: var(--border-radius-sm);
        }

        /* Horizontal Rule / Divider */
        hr {
            margin-top: var(--spacing-md); margin-bottom: var(--spacing-md); border: 0;
            border-top: 1px solid var(--divider-color, #262730);
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ----------------- API Key State Initialization -------------------
if "openrouter_api_key" not in st.session_state:
    app_conf = _load_app_config()
    st.session_state.openrouter_api_key = app_conf.get("openrouter_api_key", None)

if "api_key_auth_failed" not in st.session_state:
    st.session_state.api_key_auth_failed = False

api_key_is_syntactically_valid = is_api_key_valid(st.session_state.get("openrouter_api_key"))
app_requires_api_key_setup = not api_key_is_syntactically_valid or st.session_state.api_key_auth_failed


# -------------------- Main Application Rendering -------------------

if app_requires_api_key_setup:
    st.set_page_config(page_title="OpenRouter API Key Setup", layout="centered")
    load_custom_css()

    st.title("🔒 OpenRouter API Key Required")
    st.markdown("---")

    if st.session_state.api_key_auth_failed:
         st.error("API Key Authentication Failed. The key may be incorrect, revoked, disabled, or lack credits. Please verify your key on OpenRouter.ai and re-enter.")
    elif not api_key_is_syntactically_valid and st.session_state.get("openrouter_api_key") is not None:
        st.error("The previously configured API Key has an invalid format. It must start with `sk-or-`.")
    else:
        st.info("Please configure your OpenRouter API Key to use the application.")

    st.markdown(
        "You can get a key from [OpenRouter.ai Keys](https://openrouter.ai/keys). "
         "Enter it below to continue."
     )

    new_key_input_val = st.text_input(
        "Enter OpenRouter API Key", type="password", key="api_key_setup_input",
        value="",
        placeholder="sk-or-..."
    )

    if st.button("Save and Validate API Key", key="save_api_key_setup_button", use_container_width=True, type="primary"):
        if is_api_key_valid(new_key_input_val):
            st.session_state.openrouter_api_key = new_key_input_val
            _save_app_config(new_key_input_val)

            st.session_state.api_key_auth_failed = False

            with st.spinner("Validating API Key..."):
                fetched_credits_data = get_credits()

            if st.session_state.api_key_auth_failed:
                 st.error("Authentication failed with the provided API Key. Please check the key and try again.")
                 time.sleep(0.5)
                 st.rerun()
            elif fetched_credits_data == (None, None, None):
                st.error("Could not validate API Key. There might be a network issue or an unexpected problem with the API provider. Please try again.")
            else:
                st.success("API Key saved and validated! Initializing application...")
                if "credits" not in st.session_state: st.session_state.credits = {}
                st.session_state.credits["total"], st.session_state.credits["used"], st.session_state.credits["remaining"] = fetched_credits_data
                st.session_state.credits_ts = time.time()
                time.sleep(1.0)
                st.rerun()
        elif not new_key_input_val:
            st.warning("API Key field cannot be empty.")
        else:
            st.error("Invalid API key format. It must start with 'sk-or-'.")

    st.markdown("---")
    st.caption("Your API key is stored locally in `app_config.json` and used only to communicate with the OpenRouter API.")

else:
    st.set_page_config(
        page_title="OpenRouter Chat",
        layout="wide", # Changed to wide, chat apps usually benefit from this
        initial_sidebar_state="expanded"
    )
    load_custom_css()

    if "settings_panel_open" not in st.session_state:
        st.session_state.settings_panel_open = False

    needs_save_session = False
    if "sid" not in st.session_state:
        st.session_state.sid = _new_sid()
        needs_save_session = True
    elif st.session_state.sid not in sessions:
        logging.warning(f"Session ID {st.session_state.sid} from state not found in loaded sessions. Creating a new chat.")
        st.session_state.sid = _new_sid()
        needs_save_session = True

    if _delete_unused_blank_sessions(keep_sid=st.session_state.sid):
       needs_save_session = True

    if needs_save_session:
       _save(SESS_FILE, sessions)
       st.rerun()

    if "credits" not in st.session_state:
         st.session_state.credits = {"total": 0.0, "used": 0.0, "remaining": 0.0}
         st.session_state.credits_ts = 0

    credits_are_stale = time.time() - st.session_state.get("credits_ts", 0) > 3600
    credits_are_default = st.session_state.credits.get("total") == 0.0 and \
                          st.session_state.credits.get("used") == 0.0 and \
                          st.session_state.credits.get("remaining") == 0.0 and \
                          st.session_state.credits_ts != 0

    if credits_are_stale or credits_are_default:
        logging.info("Refreshing credits (stale or default values).")
        credits_data = get_credits()

        if st.session_state.get("api_key_auth_failed"):
            # This error will be shown in the main panel if chat interaction triggers it.
            # For now, just log it here, the main panel will handle user-facing error.
            logging.error("API Key authentication failed during credit refresh.")
            # No st.rerun() or st.stop() here, let the main flow decide.

        if credits_data != (None, None, None):
            st.session_state.credits["total"], st.session_state.credits["used"], st.session_state.credits["remaining"] = credits_data
            st.session_state.credits_ts = time.time()
        else:
            st.session_state.credits_ts = time.time() # Update timestamp even on failure to prevent rapid retries
            if not all(isinstance(st.session_state.credits.get(k), (int,float)) for k in ["total", "used", "remaining"]):
                 st.session_state.credits = {"total": 0.0, "used": 0.0, "remaining": 0.0}


    # ------------------------- Sidebar -----------------------------
    with st.sidebar:
        # Settings Toggle Button
        settings_button_label = "⚙️ Close Settings" if st.session_state.settings_panel_open else "⚙️ Settings"
        if st.button(settings_button_label, key="toggle_settings_button_sidebar", use_container_width=True):
            st.session_state.settings_panel_open = not st.session_state.settings_panel_open
            st.rerun() # Rerun to reflect button label change and panel visibility

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
                    st.session_state.api_key_auth_failed = False

                    with st.spinner("Validating new API key..."):
                        credits_data = get_credits()

                    if st.session_state.api_key_auth_failed:
                        st.error("New API Key failed authentication. Further actions may require re-setup.")
                    elif credits_data == (None,None,None):
                        st.warning("Could not validate the new API key. Key is saved, but functionality may be affected.")
                    else:
                        st.success("New API Key saved and validated!")
                        st.session_state.credits["total"],st.session_state.credits["used"],st.session_state.credits["remaining"] = credits_data
                        st.session_state.credits_ts = time.time()

                    st.session_state.settings_panel_open = False # Close panel on save
                    time.sleep(0.8)
                    st.rerun()
                elif not new_key_input_sidebar:
                    st.warning("API Key field is empty. No changes made.")
                else:
                    st.error("Invalid API key format. It must start with 'sk-or-'.")
            st.markdown("</div>", unsafe_allow_html=True)
        st.divider()


        logo_title_cols = st.columns([1, 4], gap="small")
        with logo_title_cols[0]: st.image("https://avatars.githubusercontent.com/u/130328222?s=200&v=4", width=48) # Adjusted width
        with logo_title_cols[1]: st.title("OpenRouter Chat")
        st.divider()

        # New Quota Display
        with st.expander("⚡ DAILY MODEL QUOTAS", expanded=True):
            active_model_keys = sorted(MODEL_MAP.keys())

            if not active_model_keys:
                st.caption("No models configured for quota tracking.")
            else:
                quota_cols = st.columns(len(active_model_keys))

                for i, m_key in enumerate(active_model_keys):
                    with quota_cols[i]:
                        left, _, _ = remaining(m_key)
                        lim, _, _  = PLAN[m_key]

                        is_unlimited = lim > 900_000

                        if is_unlimited:
                            pct_float = 1.0
                            fill_width_val = 100
                            left_display = "∞"
                        elif lim > 0:
                            pct_float = max(0.0, min(1.0, left / lim))
                            fill_width_val = int(pct_float * 100)
                            left_display = str(left)
                        else:
                            pct_float = 0.0
                            fill_width_val = 0
                            left_display = "0"

                        if pct_float > 0.5:
                            bar_color = "#4caf50" # Green
                        elif pct_float > 0.25:
                            bar_color = "#ffc107" # Amber
                        else:
                            bar_color = "#f44336" # Red

                        if is_unlimited:
                            bar_color = "var(--primaryColor)" # Use theme primary for unlimited

                        emoji_char = EMOJI.get(m_key, "❔")

                        st.markdown(f"""
                            <div class="compact-quota-item">
                                <div class="cq-info">{emoji_char} <b>{m_key}</b></div>
                                <div class="cq-bar-track">
                                    <div class="cq-bar-fill" style="width: {fill_width_val}%; background-color: {bar_color};"></div>
                                </div>
                                <div class="cq-value" style="color: {bar_color};">{left_display}</div>
                            </div>
                        """, unsafe_allow_html=True)
        st.divider()


        current_session_is_truly_blank = (st.session_state.sid in sessions and
                                          sessions[st.session_state.sid].get("title") == "New chat" and
                                          not sessions[st.session_state.sid].get("messages"))

        if st.button("➕ New chat", key="new_chat_button_top", use_container_width=True, disabled=current_session_is_truly_blank):
            old_sid = st.session_state.sid
            st.session_state.sid = _new_sid()
            _delete_unused_blank_sessions(keep_sid=st.session_state.sid)
            _save(SESS_FILE, sessions)
            st.rerun()

        st.subheader("Chats")
        valid_sids = [s for s in sessions.keys() if isinstance(s, str) and s.isdigit()]
        sorted_sids = sorted(valid_sids, key=lambda s: int(s), reverse=True)

        for sid_key in sorted_sids:
            if sid_key not in sessions: continue
            title = sessions[sid_key].get("title", "Untitled")
            display_title = title[:25] + ("…" if len(title) > 25 else "")
            is_active_chat = st.session_state.sid == sid_key

            # The CSS will style disabled buttons to look "active"
            if st.button(display_title, key=f"session_button_{sid_key}", use_container_width=True, disabled=is_active_chat):
                if not is_active_chat: # This condition is technically redundant due to disabled state, but good for clarity
                    _delete_unused_blank_sessions(keep_sid=sid_key)
                    st.session_state.sid = sid_key
                    _save(SESS_FILE, sessions)
                    st.rerun()
        st.divider()

        st.subheader("Model-Routing Map")
        st.caption(f"Router: {ROUTER_MODEL_ID}")
        with st.expander("Letters → Models", expanded=False):
            for k_model in sorted(MODEL_MAP.keys()):
                desc = MODEL_DESCRIPTIONS.get(k_model, MODEL_MAP.get(k_model, "N/A"))
                max_tok = MAX_TOKENS.get(k_model, 0)
                st.markdown(f"**{k_model}**: {desc} (max_out={max_tok:,})")
        st.divider()

        with st.expander("Account stats (credits)", expanded=False):
            if st.button("Refresh Credits", key="refresh_credits_button_sidebar"): # Changed key to avoid conflict
                 with st.spinner("Refreshing credits..."):
                    credits_data = get_credits()
                 if not st.session_state.get("api_key_auth_failed"):
                    if credits_data != (None,None,None):
                        st.session_state.credits["total"],st.session_state.credits["used"],st.session_state.credits["remaining"] = credits_data
                        st.session_state.credits_ts = time.time()
                        st.success("Credits refreshed!")
                    else:
                        st.warning("Could not refresh credits (network or API issue).")
                 else:
                     st.error("API Key authentication failed. Cannot refresh credits.")
                 st.rerun()

            tot = st.session_state.credits.get("total")
            used = st.session_state.credits.get("used")
            rem = st.session_state.credits.get("remaining")

            if tot is None or used is None or rem is None :
                 st.warning("Could not fetch/display credits. Check network or API key (in Settings).")
            else:
                st.markdown(f"**Remaining:** ${float(rem):.2f} cr")
                st.markdown(f"**Used:** ${float(used):.2f} cr")

            ts = st.session_state.get("credits_ts", 0)
            last_updated_str = datetime.fromtimestamp(ts, TZ).strftime('%-d %b, %H:%M:%S') if ts else "N/A"
            st.caption(f"Last updated: {last_updated_str}")


    # ------------------------- Main Chat Panel ---------------------
    if st.session_state.sid not in sessions:
        logging.error(f"Current session ID {st.session_state.sid} missing from sessions. Resetting to new chat.")
        st.session_state.sid = _new_sid()
        _save(SESS_FILE, sessions)
        st.rerun()
        st.stop()

    current_sid = st.session_state.sid
    chat_history = sessions[current_sid]["messages"]

    for msg in chat_history:
        role = msg.get("role", "assistant")
        avatar_char = "👤" if role == "user" else None # Avatar for user

        if role == "assistant":
            m_key = msg.get("model")
            if m_key == FALLBACK_MODEL_KEY:
                avatar_char = FALLBACK_MODEL_EMOJI
            elif m_key in EMOJI:
                avatar_char = EMOJI[m_key]
            else:
                avatar_char = "🤖" # Default assistant avatar

        with st.chat_message(role, avatar=avatar_char):
             st.markdown(msg.get("content", "*empty message*"))

    if prompt := st.chat_input("Ask anything…", key=f"chat_input_{current_sid}"):
        chat_history.append({"role":"user","content":prompt})
        with st.chat_message("user", avatar="👤"): st.markdown(prompt) # Ensure user avatar here too

        if not is_api_key_valid(st.session_state.get("openrouter_api_key")) or st.session_state.get("api_key_auth_failed"):
            st.error("API Key is not configured or has failed. Please set it up in ⚙️ Settings.")
            # No st.rerun() here, let the error message persist until user fixes it.
            # st.stop() might be too abrupt; the error message should be enough warning.
        else:
            allowed_standard_models = [k for k in MODEL_MAP if remaining(k)[0] > 0]
            use_fallback, chosen_model_key, model_id_to_use, max_tokens_api, avatar_resp = (
                False, None, None, None, "🤖"
            )

            if not allowed_standard_models:
                logging.info(f"Using fallback (all quotas used): {FALLBACK_MODEL_ID}")
                st.info(f"{FALLBACK_MODEL_EMOJI} Daily quotas for standard models exhausted. Using free fallback.")
                use_fallback, chosen_model_key, model_id_to_use, max_tokens_api, avatar_resp = (True, FALLBACK_MODEL_KEY, FALLBACK_MODEL_ID, FALLBACK_MODEL_MAX_TOKENS, FALLBACK_MODEL_EMOJI)
            else:
                routed_key = route_choice(prompt, allowed_standard_models)
                if st.session_state.get("api_key_auth_failed"): # Check after routing attempt
                     st.error("API Authentication failed during model routing. Please check your API Key in Settings.")
                     # No rerun, let error show.
                elif routed_key not in MODEL_MAP or routed_key not in allowed_standard_models:
                    logging.warning(f"Router chose '{routed_key}' (invalid or no quota). Using fallback {FALLBACK_MODEL_ID}.")
                    st.warning(f"{FALLBACK_MODEL_EMOJI} Model routing issue or chosen model '{routed_key}' has no quota. Using free fallback.")
                    use_fallback, chosen_model_key, model_id_to_use, max_tokens_api, avatar_resp = (True, FALLBACK_MODEL_KEY, FALLBACK_MODEL_ID, FALLBACK_MODEL_MAX_TOKENS, FALLBACK_MODEL_EMOJI)
                else:
                    chosen_model_key = routed_key
                    model_id_to_use = MODEL_MAP[chosen_model_key]
                    max_tokens_api = MAX_TOKENS[chosen_model_key]
                    avatar_resp = EMOJI.get(chosen_model_key, "🤖")

            with st.chat_message("assistant", avatar=avatar_resp):
                response_placeholder, full_response = st.empty(), ""
                api_call_ok = True

                for chunk, err_msg in streamed(model_id_to_use, chat_history, max_tokens_api):
                    if st.session_state.get("api_key_auth_failed"): # Check during streaming
                        full_response = "❗ **API Authentication Error**: Your API Key failed. Please update it in ⚙️ Settings."
                        api_call_ok = False; break
                    if err_msg:
                        full_response = f"❗ **API Error**: {err_msg}"
                        api_call_ok = False; break
                    if chunk:
                       full_response += chunk
                       response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

            chat_history.append({"role":"assistant","content":full_response,"model": chosen_model_key if api_call_ok else FALLBACK_MODEL_KEY})

            if api_call_ok:
                if not use_fallback and chosen_model_key: # Ensure chosen_model_key is not None
                   record_use(chosen_model_key)
                if sessions[current_sid]["title"] == "New chat" and prompt:
                   sessions[current_sid]["title"] = _autoname(prompt)
                   _delete_unused_blank_sessions(keep_sid=current_sid)

            _save(SESS_FILE, sessions)
            st.rerun() # Rerun to update sidebar (quota, possibly new chat title)
