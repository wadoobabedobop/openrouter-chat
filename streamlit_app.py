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
    # "E": "x-ai/grok-3-beta", # REMOVED E
    "F": "google/gemini-2.5-flash-preview"
}
# Router uses Mistral 7B Instruct
ROUTER_MODEL_ID = "mistralai/mistral-7b-instruct:free"

# Token limits for outputs
MAX_TOKENS = {
    "A": 16_000, "B": 8_000, "C": 16_000,
    "D": 8_000,  # "E": 4_000, # REMOVED E
    "F": 8_000
}

# Quota plan: (daily, weekly, monthly) messages
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
    """
    Deletes all sessions that are blank (title "New chat" and no messages),
    except for the session with sid `keep_sid` if provided.
    Modifies the global `sessions` object.
    Returns True if any sessions were deleted.
    """
    sids_to_delete = []
    for sid, data in sessions.items(): # Iterate over a copy if modifying dict during iteration
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

sessions = _load(SESS_FILE, {}) # Initialize global sessions

def _new_sid():
    """
    Clears ALL old blank/unused sessions, then creates a new session ID 
    and adds a basic session structure to the global `sessions` dict.
    """
    _delete_unused_blank_sessions(keep_sid=None) # Clear ALL old blank sessions first

    sid = str(int(time.time() * 1000))
    sessions[sid] = {"title": "New chat", "messages": []}
    # The caller of _new_sid is responsible for _save(SESS_FILE, sessions)
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

# ───────────────────────── Streamlit UI ────────────────────────
st.set_page_config(
    page_title="OpenRouter Chat",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initial SID Management and Cleanup
needs_save_and_rerun_on_startup = False
if "sid" not in st.session_state:
    st.session_state.sid = _new_sid() # _new_sid cleans, creates, adds to global 'sessions'
    needs_save_and_rerun_on_startup = True
elif st.session_state.sid not in sessions:
    logging.warning(f"Session ID {st.session_state.sid} from state not found in loaded sessions. Creating a new chat.")
    st.session_state.sid = _new_sid() # _new_sid cleans, creates, adds to global 'sessions'
    needs_save_and_rerun_on_startup = True
else:
    # SID is valid and exists in sessions. Clean up OTHER blank sessions, keeping the current one.
    if _delete_unused_blank_sessions(keep_sid=st.session_state.sid):
        needs_save_and_rerun_on_startup = True

if needs_save_and_rerun_on_startup:
    _save(SESS_FILE, sessions)
    st.rerun()

# Initialize credits if not already in session state
if "credits" not in st.session_state:
    st.session_state.credits = dict(zip(
        ("total", "used", "remaining"),
        get_credits()
    ))
    st.session_state.credits_ts = time.time()


# ───────────────────────── Sidebar ─────────────────────────────
with st.sidebar:
    st.image("https://avatars.githubusercontent.com/u/130328222?s=200&v=4", width=50)
    st.title("OpenRouter Chat")

    st.subheader("Daily Jars (Msgs Left)")
    cols = st.columns(len(MODEL_MAP))
    active_model_keys = sorted(MODEL_MAP.keys())
    for i, m_key in enumerate(active_model_keys):
        left, _, _ = remaining(m_key)
        lim, _, _  = PLAN[m_key]
        pct = 1.0 if lim > 900_000 else max(0.0, left / lim if lim > 0 else 0.0)
        fill = int(pct * 100)
        color = "#4caf50" if pct > .5 else "#ff9800" if pct > .25 else "#f44336"
        cols[i].markdown(f"""
            <div style="width:44px; margin:auto; text-align:center;">
              <div style="height:60px; border:1px solid #ccc; border-radius:7px; background:#f5f5f5; position:relative; overflow:hidden; box-shadow: inset 0 1px 2px rgba(0,0,0,0.07),0 1px 1px rgba(0,0,0,0.05);"><div style="position:absolute; bottom:0; width:100%; height:{fill}%; background:{color}; box-shadow: inset 0 2px 2px rgba(255,255,255,0.3); box-sizing: border-box;"></div><div style="position:absolute; top:2px; width:100%; font-size:18px; line-height:1;">{EMOJI[m_key]}</div><div style="position:absolute; bottom:2px; width:100%; font-size:11px; font-weight:bold; color:#555; line-height:1;">{m_key}</div></div>
              <span style="display:block; margin-top:4px; font-size:11px; font-weight:600; color:#333; line-height:1;">{'∞' if lim > 900_000 else left}</span>
            </div>""", unsafe_allow_html=True)
    st.markdown("---")

    # New Chat button
    current_session_is_truly_blank = False
    if st.session_state.sid in sessions: # Ensure current_sid is valid before checking its content
        current_session_data = sessions.get(st.session_state.sid)
        if current_session_data and \
           current_session_data.get("title") == "New chat" and \
           not current_session_data.get("messages"):
            current_session_is_truly_blank = True
    
    if st.button("➕ New chat", use_container_width=True, disabled=current_session_is_truly_blank):
        new_session_id = _new_sid() # _new_sid calls _delete_unused_blank_sessions internally
        st.session_state.sid = new_session_id
        _save(SESS_FILE, sessions) # Save all changes (new session, potentially deleted old blanks)
        st.rerun()
    elif current_session_is_truly_blank:
        st.caption("Current chat is empty. Add a message or switch.")

    # Chat session list
    st.subheader("Chats")
    sorted_sids = sorted(sessions.keys(), key=lambda s: int(s), reverse=True)
    for sid_key in sorted_sids:
        title = sessions[sid_key].get("title", "Untitled")[:25] + ("…" if len(sessions[sid_key].get("title", "")) > 25 else "")
        if st.button(title, key=f"session_button_{sid_key}", use_container_width=True):
            if st.session_state.sid != sid_key:
                st.session_state.sid = sid_key
                # Switched to sid_key. Clean up other blank sessions, keeping the newly selected one.
                if _delete_unused_blank_sessions(keep_sid=sid_key):
                    _save(SESS_FILE, sessions)
                st.rerun()
    st.markdown("---")

    st.subheader("Model-Routing Map")
    st.caption(f"Router engine: {ROUTER_MODEL_ID}")
    with st.expander("Letters → Models"):
        for k_model in sorted(MODEL_MAP.keys()):
            st.markdown(f"**{k_model}**: {MODEL_DESCRIPTIONS[k_model]} (max_output={MAX_TOKENS[k_model]:,})")
    st.markdown("---")

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
            st.markdown(f"**Purchased:** {tot:.2f} cr")
            st.markdown(f"**Used:** {used:.2f} cr")
            st.markdown(f"**Remaining:** {rem:.2f} cr")
            try:
                last_updated_str = datetime.fromtimestamp(st.session_state.credits_ts).strftime('%Y-%m-%d %H:%M:%S')
                st.caption(f"Last updated: {last_updated_str}")
            except TypeError: st.caption("Last updated: N/A")


# ────────────────────────── Main Chat Panel ─────────────────────
current_sid = st.session_state.sid
if current_sid not in sessions:
    st.error("Selected chat session not found. Creating a new one.")
    current_sid = _new_sid() # _new_sid cleans, then creates new session
    st.session_state.sid = current_sid
    _save(SESS_FILE, sessions) # Save because _new_sid modified global sessions
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

if prompt := st.chat_input("Ask anything…"):
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
            routed_key = route_choice(prompt, allowed_standard_models)
            logging.info(f"Router selected model: '{routed_key}'.")
            chosen_model_key_for_api = routed_key
        model_id_to_use_for_api = MODEL_MAP[chosen_model_key_for_api]
        max_tokens_for_api = MAX_TOKENS[chosen_model_key_for_api]
        avatar_for_response = EMOJI[chosen_model_key_for_api]

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

    chat_history.append({"role":"assistant","content":full_response_content,"model": chosen_model_key_for_api})

    if api_call_ok:
        if not use_fallback_model: record_use(chosen_model_key_for_api)
        
        # Check if current chat was "New chat" and now has messages
        if sessions[current_sid]["title"] == "New chat" and sessions[current_sid]["messages"]: # messages list now includes user + assistant
            sessions[current_sid]["title"] = _autoname(prompt)
            # Current chat is now named. Clean up any OTHER blank sessions.
            _delete_unused_blank_sessions(keep_sid=current_sid) 
            # No specific save/rerun here for _delete, main ones below cover it

    _save(SESS_FILE, sessions)
    st.rerun()

# ───────────────────────── Self-Relaunch ──────────────────────
if __name__ == "__main__" and os.getenv("_IS_STRL") != "1":
    os.environ["_IS_STRL"] = "1"
    port = os.getenv("PORT", "8501")
    cmd = [sys.executable, "-m", "streamlit", "run", __file__, "--server.port", port, "--server.address", "0.0.0.0"]
    logging.info(f"Relaunching with Streamlit: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)
