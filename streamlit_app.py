#!/usr/bin/env python3
"""
OpenRouter Streamlit Chat — Full Edition (Redesigned UI - Polished)
"""

# ───────────────────────── Imports ─────────────────────────
import json, logging, os, sys, subprocess, time, requests
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo
import streamlit as st

# ────────────────────────── Configuration ───────────────────
OPENROUTER_API_KEY  = "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" # Replace
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT     = 120

FALLBACK_MODEL_ID = "deepseek/deepseek-chat-v3-0324:free"
FALLBACK_MODEL_KEY = "_FALLBACK_"
FALLBACK_MODEL_EMOJI = "🆓"
FALLBACK_MODEL_MAX_TOKENS = 8000

MODEL_MAP = {
    "A": "google/gemini-2.5-pro-preview",
    "B": "openai/o4-mini",
    "C": "openai/chatgpt-4o-latest",
    "D": "deepseek/deepseek-r1",
    "F": "google/gemini-2.5-flash-preview"
}
ROUTER_MODEL_ID = "mistralai/mistral-7b-instruct:free"

MAX_TOKENS = {"A": 16_000, "B": 8_000, "C": 16_000, "D": 8_000, "F": 8_000}
PLAN = {
    "A": (10, 70, 300), "B": (5, 35, 150), "C": (1, 7, 30),
    "D": (4, 28, 120), "F": (180, 500, 2000)
}
EMOJI = {"A": "🌟", "B": "🔷", "C": "🟥", "D": "🟢", "F": "🌀"}
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
    try: return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError): return default
def _save(path: Path, obj): path.write_text(json.dumps(obj, indent=2))
def _today(): return date.today().isoformat()
def _yweek(): return datetime.now(TZ).strftime("%G-%V")
def _ymonth(): return datetime.now(TZ).strftime("%Y-%m")

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
    _reset(q, "d", _today(), zeros); _reset(q, "w", _yweek(), zeros); _reset(q, "m", _ymonth(), zeros)
    _save(QUOTA_FILE, q); return q
quota = _load_quota()

def remaining(key: str):
    ud = quota.get("d_u", {}).get(key, 0); uw = quota.get("w_u", {}).get(key, 0); um = quota.get("m_u", {}).get(key, 0)
    if key not in PLAN: logging.error(f"Unknown key for remaining: {key}"); return 0,0,0
    ld, lw, lm = PLAN[key]; return ld - ud, lw - uw, lm - um

def record_use(key: str):
    if key not in MODEL_MAP: logging.warning(f"Unknown model key for record_use: {key}"); return
    for blk_key in ("d_u", "w_u", "m_u"):
        if blk_key not in quota: quota[blk_key] = {k: 0 for k in MODEL_MAP}
        quota[blk_key][key] = quota[blk_key].get(key, 0) + 1
    _save(QUOTA_FILE, quota)

# ───────────────────── Session Management ───────────────────────
def _delete_unused_blank_sessions(keep_sid: str = None):
    sids_to_delete = [sid for sid, data in sessions.items() if sid != keep_sid and data.get("title") == "New chat" and not data.get("messages")]
    if sids_to_delete:
        for sid_del in sids_to_delete: logging.info(f"Auto-deleting blank session: {sid_del}"); del sessions[sid_del]
        return True
    return False
sessions = _load(SESS_FILE, {})
def _new_sid():
    _delete_unused_blank_sessions(keep_sid=None)
    sid = str(int(time.time() * 1000))
    sessions[sid] = {"title": "New chat", "messages": []}; return sid
def _autoname(seed: str) -> str:
    words = seed.strip().split(); cand = " ".join(words[:3]) or "Chat"
    return (cand[:25] + "…") if len(cand) > 25 else cand

# ─────────────────────────── Logging ────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)

# ────────────────────────── API Calls ──────────────────────────
def api_post(payload: dict, *, stream: bool=False, timeout: int=DEFAULT_TIMEOUT):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type":  "application/json"}
    logging.info(f"POST /chat/completions → model={payload.get('model')}, stream={stream}, max_tokens={payload.get('max_tokens')}")
    return requests.post(f"{OPENROUTER_API_BASE}/chat/completions", headers=headers, json=payload, stream=stream, timeout=timeout)

def streamed(model: str, messages: list, max_tokens_out: int):
    payload = {"model": model, "messages": messages, "stream": True, "max_tokens": max_tokens_out}
    with api_post(payload, stream=True) as r:
        try: r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            text = r.text; logging.error(f"Stream HTTPError {e.response.status_code}: {text}"); yield None, f"HTTP {e.response.status_code}: {text}"; return
        for line in r.iter_lines():
            if not line: continue
            line_str = line.decode("utf-8")
            if line_str.startswith(": OPENROUTER PROCESSING"): logging.info(f"OpenRouter PING: {line_str.strip()}"); continue
            if not line_str.startswith("data: "): logging.warning(f"Unexpected non-event-stream line: {line}"); continue
            data = line_str[6:].strip()
            if data == "[DONE]": break
            try: chunk = json.loads(data)
            except json.JSONDecodeError: logging.error(f"Bad JSON chunk: {data}"); yield None, "Error decoding response chunk"; return
            if "error" in chunk: msg = chunk["error"].get("message", "Unknown API error"); logging.error(f"API chunk error: {msg}"); yield None, msg; return
            delta = chunk["choices"][0]["delta"].get("content")
            if delta is not None: yield delta, None

# ───────────────────────── Model Routing ───────────────────────
def route_choice(user_msg: str, allowed: list[str]) -> str:
    if not allowed: return "F" if "F" in MODEL_MAP else (list(MODEL_MAP.keys())[0] if MODEL_MAP else "F")
    if len(allowed) == 1: return allowed[0]
    system_lines = ["You are an intelligent model-routing assistant.", "Select ONLY one letter from the following available models:"]
    for k in allowed:
        if k in MODEL_DESCRIPTIONS: system_lines.append(f"- {k}: {MODEL_DESCRIPTIONS[k]}")
    system_lines.extend(["Based on the user's query, choose the letter that best balances quality, speed, and cost-sensitivity.", "Respond with ONLY the single capital letter. No extra text."])
    router_messages = [{"role": "system", "content": "\n".join(system_lines)}, {"role": "user", "content": user_msg}]
    payload_r = {"model": ROUTER_MODEL_ID, "messages": router_messages, "max_tokens": 10}
    try:
        r = api_post(payload_r); r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip().upper()
        logging.info(f"Router raw response: {text}")
        for ch in text:
            if ch in allowed: return ch
    except Exception as e: logging.error(f"Router call error: {e}")
    fallback_choice = "F" if "F" in allowed else allowed[0]
    logging.warning(f"Router fallback to model: {fallback_choice}"); return fallback_choice

# ───────────────────── Credits Endpoint ───────────────────────
def get_credits():
    try:
        r = requests.get(f"{OPENROUTER_API_BASE}/credits", headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}, timeout=10)
        r.raise_for_status(); d = r.json()["data"]; return d["total_credits"], d["total_usage"], d["total_credits"] - d["total_usage"]
    except Exception as e: logging.warning(f"Could not fetch /credits: {e}"); return None, None, None

# ───────────────────────── UI Styling ──────────────────────────
def load_custom_css():
    css = """
    <style>
        /* General Styles */
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
        }
        /* Ensure main content area uses Streamlit's theme background */
        [data-testid="stAppViewContainer"] > .main {
            background-color: var(--background-color);
        }
        [data-testid="stAppViewContainer"] > .main > .block-container {
            padding-top: 2.5rem; 
            padding-bottom: 2.5rem;
            max-width: 900px; /* Optional: Constrain main content width for readability */
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #1F2937; /* Consistent dark sidebar */
            padding: 1.5rem 1rem;
            border-right: 1px solid #374151; 
        }
        html[data-theme="light"] [data-testid="stSidebar"] {
            background-color: #F3F4F6; /* Lighter grey for light theme sidebar */
            border-right: 1px solid #E5E7EB;
        }

        /* Sidebar Header (Logo + Title) */
        [data-testid="stSidebar"] > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) { 
            display: flex !important; align-items: center !important; gap: 12px; 
            margin-bottom: 2rem !important; padding-bottom: 1.25rem;
            border-bottom: 1px solid #4B5563; /* Slightly more visible border in dark sidebar */
        }
        html[data-theme="light"] [data-testid="stSidebar"] > div:nth-child(1) > div:nth-child(1) > div:nth-child(1) {
            border-bottom: 1px solid #D1D5DB;
        }
        [data-testid="stSidebar"] .stImage > img {
            border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            width: 42px !important; height: 42px !important;
        }
        [data-testid="stSidebar"] h1 { /* App Title */
            font-size: 1.5rem !important; color: #E5E7EB; /* Lighter text for dark bg */
            font-weight: 600; margin-bottom: 0;
        }
        html[data-theme="light"] [data-testid="stSidebar"] h1 { color: var(--text-color); }


        /* Sidebar Subheaders */
        [data-testid="stSidebar"] h3 {
            font-size: 0.75rem !important; text-transform: uppercase;
            font-weight: 500; /* Less bold */
            color: #9CA3AF; /* Muted grey for dark theme subheaders */
            margin-top: 2rem; margin-bottom: 0.8rem; letter-spacing: 0.025em;
        }
        html[data-theme="light"] [data-testid="stSidebar"] h3 { color: #6B7280; }

        /* Sidebar Buttons (Session list, New Chat) */
        [data-testid="stSidebar"] .stButton > button {
            border-radius: 6px; border: none; 
            padding: 0.65em 0.8em; font-size: 0.9rem; font-weight: 500;
            background-color: transparent; 
            color: #D1D5DB; /* Default button text color in dark sidebar */
            transition: background-color 0.2s, color 0.2s;
            width: 100%; margin-bottom: 0.25rem; text-align: left;
            display: flex; align-items: center; gap: 8px; /* Gap for icon and text */
        }
        html[data-theme="light"] [data-testid="stSidebar"] .stButton > button { color: #4B5563; }

        [data-testid="stSidebar"] .stButton > button:hover {
            background-color: #374151; /* Darker hover for dark sidebar */
            color: #F9FAFB; /* Brighter text on hover */
        }
        html[data-theme="light"] [data-testid="stSidebar"] .stButton > button:hover {
            background-color: #E5E7EB; color: var(--primary-color);
        }
        
        /* Specifically target the "New Chat" button if it needs to be different,
           but for screenshot match, it's similar to other buttons */
        [data-testid="stSidebar"] [data-testid="stButton-new_chat_button_top"] > button {
            /* Inherits general button style, looks like screenshot */
            font-weight: 500; /* Matches other buttons */
        }
        [data-testid="stSidebar"] [data-testid="stButton-new_chat_button_top"] > button:hover {
             background-color: var(--primary-color-darker, #4a5568); /* Slightly different hover for New Chat if desired */
             color: white;
        }
         html[data-theme="light"] [data-testid="stSidebar"] [data-testid="stButton-new_chat_button_top"] > button:hover {
             background-color: var(--primary-color);
             color: white;
        }


        /* Active session button (with 🔹) */
        [data-testid="stSidebar"] .stButton > button:has(span:contains("🔹")) { 
             color: var(--primary-color); /* Use Streamlit's primary color for active */
             background-color: color-mix(in srgb, var(--primary-color) 15%, transparent);
             font-weight: 500;
        }
         /* Alternative for non :has browsers, if you wrap in span class="active-chat-item" */
        [data-testid="stSidebar"] .stButton > button .active-chat-item {
            color: var(--primary-color); font-weight: 500;
        }


        /* Sidebar Caption ("Current chat is empty...") */
        [data-testid="stSidebar"] .stCaption {
            color: #9CA3AF; font-size: 0.8rem; text-align: left;
            padding: 0.2rem 0.1rem 1rem 0.1rem; line-height: 1.4;
        }
        html[data-theme="light"] [data-testid="stSidebar"] .stCaption { color: #6B7280; }


        /* Custom Token Jar Styling */
        .token-jar-container { max-width: 52px; margin: 0 auto 0.4rem auto; text-align: center; }
        .token-jar {
            height: 55px; border: 1px solid #4B5563; border-radius: 7px;
            background: #2d3748; /* Slightly different dark shade for jar */
            position: relative; overflow: hidden; margin-bottom: 4px;
        }
        html[data-theme="light"] .token-jar { border-color: #D1D5DB; background: #E5E7EB; }
        .token-jar-fill { position: absolute; bottom: 0; width: 100%; transition: height 0.3s ease-in-out; }
        .token-jar-emoji { position: absolute; top: 6px; width:100%; font-size:18px; line-height:1; }
        .token-jar-key {
            position: absolute; bottom: 5px; width:100%; font-size:10px;
            font-weight: 500; color: #A0AEC0; line-height:1;
        }
        html[data-theme="light"] .token-jar-key { color: #4A5568; }
        .token-jar-remaining {
            display: block; margin-top: 2px; font-size: 11px;
            font-weight: 600; color: #CBD5E0; line-height:1;
        }
        html[data-theme="light"] .token-jar-remaining { color: #2D3748; }

        /* Expander Styling */
        .stExpander {
            border: 1px solid #4B5563; border-radius: 8px; margin-bottom: 1rem;
            background-color: transparent; 
        }
        html[data-theme="light"] .stExpander { border-color: #D1D5DB; }
        .stExpander header {
            font-weight: 500; font-size: 0.85rem; padding: 0.6rem 1rem !important;
            background-color: #2D3748; /* Header distinct from content */
            border-bottom: 1px solid #4B5563;
            border-top-left-radius: 7px; border-top-right-radius: 7px;
            color: #CBD5E0;
        }
        html[data-theme="light"] .stExpander header { background-color: #E5E7EB; border-color: #D1D5DB; color: #4A5568; }
        .stExpander header:hover { background-color: #374151; }
        html[data-theme="light"] .stExpander header:hover { background-color: #D1D5DB; }
        .stExpander div[data-testid="stExpanderDetails"] { padding: 0.75rem 1rem; background-color: transparent; }

        /* Chat Message Styling */
        [data-testid="stChatMessage"] {
            border-radius: 12px; padding: 12px 18px; margin-bottom: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: none; max-width: 80%; line-height: 1.6;
        }
        [data-testid^="stChatMessageUser"] { 
            background-color: var(--primary-color); color: white; margin-left: auto; border-bottom-right-radius: 4px; 
        }
        [data-testid^="stChatMessageUser"] .stMarkdown p, [data-testid^="stChatMessageUser"] .stMarkdown li { color: white !important; }
        [data-testid^="stChatMessageAssistant"] { 
            background-color: #2D3748; /* Darker assistant bubble for dark theme */
            color: #E2E8F0; border-bottom-left-radius: 4px; margin-right: auto;
        }
        html[data-theme="light"] [data-testid^="stChatMessageAssistant"] { background-color: #F1F5F9; color: var(--text-color); }
        [data-testid^="stChatMessageAssistant"] .stMarkdown p, [data-testid^="stChatMessageAssistant"] .stMarkdown li { color: inherit !important; }
        
        /* Chat Input */
        [data-testid="stChatInput"] {
            background-color: var(--background-color); /* Match main app background */
            border-top: 1px solid #374151; padding: 0.75rem 0.5rem; /* Adjust padding */
        }
        html[data-theme="light"] [data-testid="stChatInput"] { border-top-color: #E5E7EB; }

        [data-testid="stChatInput"] textarea { /* Target textarea specifically */
            border-radius: 20px !important; /* More pronounced rounding */
            border: 1px solid #4A5568 !important;
            background-color: #2D3748 !important; /* Darker input field */
            color: #E2E8F0 !important;
            padding: 10px 15px !important;
            line-height: 1.5 !important;
        }
        html[data-theme="light"] [data-testid="stChatInput"] textarea {
            border-color: #D1D5DB !important; background-color: #FFFFFF !important; color: var(--text-color) !important;
        }
        [data-testid="stChatInput"] textarea:focus {
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 1px var(--primary-color) !important;
        }
        /* Send button icon color */
        [data-testid="stChatInput"] button svg {
            fill: #9CA3AF; /* Muted send icon */
        }
        [data-testid="stChatInput"] button:hover svg {
            fill: var(--primary-color);
        }


        /* Centered Welcome Message for Empty Chat */
        .empty-chat-container {
            display: flex; flex-direction: column; align-items: center; justify-content: center;
            min-height: 65vh; text-align: center; padding: 2rem;
        }
        .empty-chat-container img.logo-main {
            width: 80px; height: 80px; border-radius: 16px; margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        }
        .empty-chat-container h2 {
            font-size: 1.9rem; font-weight: 600; margin-bottom: 0.8rem; color: var(--text-color);
        }
        .empty-chat-container p {
            font-size: 1.05rem; color: var(--text-color-secondary); max-width: 480px; line-height: 1.65;
        }

        hr { margin: 1.8rem 0; border: 0; border-top: 1px solid #374151; }
        html[data-theme="light"] hr { border-top-color: #E5E7EB; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ───────────────────────── Streamlit UI ────────────────────────
st.set_page_config(page_title="OpenRouter Chat", layout="wide", initial_sidebar_state="expanded")
load_custom_css()

# Initial SID Management
needs_save_and_rerun_on_startup = False
if "sid" not in st.session_state: st.session_state.sid = _new_sid(); needs_save_and_rerun_on_startup = True
elif st.session_state.sid not in sessions:
    logging.warning(f"SID {st.session_state.sid} not found. New chat."); st.session_state.sid = _new_sid(); needs_save_and_rerun_on_startup = True
else:
    if _delete_unused_blank_sessions(keep_sid=st.session_state.sid): needs_save_and_rerun_on_startup = True
if needs_save_and_rerun_on_startup: _save(SESS_FILE, sessions); st.rerun()

if "credits" not in st.session_state:
    st.session_state.credits = dict(zip(("total", "used", "remaining"), get_credits()))
    st.session_state.credits_ts = time.time()

# ───────────────────────── Sidebar ─────────────────────────────
with st.sidebar:
    st.image("https://avatars.githubusercontent.com/u/130328222?s=200&v=4", width=50) 
    st.title("OpenRouter Chat")

    st.subheader("Daily Jars (Msgs Left)")
    active_model_keys = sorted(MODEL_MAP.keys())
    cols = st.columns(len(active_model_keys))
    for i, m_key in enumerate(active_model_keys):
        left, _, _ = remaining(m_key); lim, _, _  = PLAN[m_key]
        pct = 1.0 if lim > 900_000 else max(0.0, left / lim if lim > 0 else 0.0)
        fill_color = "#4caf50" if pct > .5 else ("#ffc107" if pct > .25 else "#f44336")
        cols[i].markdown(f"""<div class="token-jar-container">
              <div class="token-jar"><div class="token-jar-fill" style="height:{int(pct*100)}%; background-color:{fill_color};"></div>
                <div class="token-jar-emoji">{EMOJI[m_key]}</div><div class="token-jar-key">{m_key}</div></div>
              <span class="token-jar-remaining">{'∞' if lim > 900_000 else left}</span></div>""", unsafe_allow_html=True)
    st.divider() 

    current_session_is_truly_blank = (st.session_state.sid in sessions and
                                      sessions[st.session_state.sid].get("title") == "New chat" and
                                      not sessions[st.session_state.sid].get("messages"))
    
    if st.button("➕ New chat", key="new_chat_button_top", use_container_width=True, disabled=current_session_is_truly_blank):
        st.session_state.sid = _new_sid(); _save(SESS_FILE, sessions); st.rerun()
    
    if current_session_is_truly_blank and not st.session_state.get("new_chat_button_top_clicked"): # Avoid showing caption if button was just clicked
         st.caption("Current chat is empty. Add a message or switch.")


    st.subheader("Chats")
    sorted_sids = sorted(sessions.keys(), key=lambda s: int(s), reverse=True)
    for sid_key in sorted_sids:
        title = sessions[sid_key].get("title", "Untitled")
        display_title = title[:28] + ("…" if len(title) > 28 else "")
        if st.session_state.sid == sid_key:
            # Wrap active title in a span with a class for CSS targeting (optional, :has is better)
            # display_title_html = f"<span class='active-chat-item'>🔹 {display_title}</span>"
            # For simplicity with current CSS using :has or just text content:
            display_title = f"🔹 {display_title}"

        if st.button(display_title, key=f"session_button_{sid_key}", use_container_width=True): # unsafe_allow_html=(st.session_state.sid == sid_key)
            if st.session_state.sid != sid_key:
                st.session_state.sid = sid_key
                if _delete_unused_blank_sessions(keep_sid=sid_key): _save(SESS_FILE, sessions)
                st.rerun()
    st.divider()

    st.subheader("Model-Routing Map")
    st.caption(f"Router engine: {ROUTER_MODEL_ID}")
    with st.expander("Letters → Models", expanded=False):
        for k_model in sorted(MODEL_MAP.keys()):
            st.markdown(f"**{k_model}**: {MODEL_DESCRIPTIONS[k_model]} (max_output={MAX_TOKENS[k_model]:,})")
    st.divider()

    tot, used, rem = (st.session_state.credits.get(k) for k in ("total","used","remaining"))
    with st.expander("Account stats (credits)", expanded=False):
        if st.button("Refresh Credits", key="refresh_credits_button"):
            st.session_state.credits = dict(zip(("total","used","remaining"), get_credits()))
            st.session_state.credits_ts = time.time(); st.rerun()
        if tot is None: st.warning("Could not fetch credits.")
        else:
            st.markdown(f"**Purchased:** ${tot:.2f} cr\n\n**Used:** ${used:.2f} cr\n\n**Remaining:** ${rem:.2f} cr")
            try: st.caption(f"Last updated: {datetime.fromtimestamp(st.session_state.credits_ts, TZ).strftime('%-d %b %Y, %H:%M:%S')}")
            except TypeError: st.caption("Last updated: N/A")

# ────────────────────────── Main Chat Panel ─────────────────────
current_sid = st.session_state.sid
if current_sid not in sessions: # Should be caught by startup logic, but as a safeguard
    st.error("Chat session error. Creating new."); current_sid = _new_sid(); st.session_state.sid = current_sid
    _save(SESS_FILE, sessions); st.rerun()

chat_history = sessions[current_sid]["messages"]
is_new_empty_chat = not chat_history and sessions[current_sid]["title"] == "New chat"

if is_new_empty_chat:
    st.markdown(f"""<div class="empty-chat-container">
        <img src="https://avatars.githubusercontent.com/u/130328222?s=200&v=4" class="logo-main">
        <h2>How can I help you today, Asher?</h2>
        <p>I can help you choose the best model for your task based on its capabilities and your remaining quotas. Just type your query below!</p>
    </div>""", unsafe_allow_html=True)
else:
    for msg_idx, msg in enumerate(chat_history):
        role = msg["role"]; avatar = "👤"
        if role == "assistant":
            model_key = msg.get("model")
            avatar = FALLBACK_MODEL_EMOJI if model_key == FALLBACK_MODEL_KEY else EMOJI.get(model_key, EMOJI.get("F", "🤖"))
        with st.chat_message(role, avatar=avatar): st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything…", key=f"chat_input_{current_sid}"):
    chat_history.append({"role":"user","content":prompt})
    if not is_new_empty_chat: # Display user message immediately only if chat wasn't empty
        with st.chat_message("user", avatar="👤"): st.markdown(prompt)

    allowed_standard_models = [k for k in MODEL_MAP if remaining(k)[0] > 0]
    chosen_model_key, model_id_to_use, max_tokens_api, avatar_resp = FALLBACK_MODEL_KEY, FALLBACK_MODEL_ID, FALLBACK_MODEL_MAX_TOKENS, FALLBACK_MODEL_EMOJI
    use_fallback = not allowed_standard_models

    if not use_fallback:
        chosen_model_key = allowed_standard_models[0] if len(allowed_standard_models) == 1 else route_choice(prompt, allowed_standard_models)
        logging.info(f"Chosen model key: '{chosen_model_key}' (Fallback={use_fallback})")
        model_id_to_use = MODEL_MAP[chosen_model_key]
        max_tokens_api = MAX_TOKENS[chosen_model_key]
        avatar_resp = EMOJI[chosen_model_key]
    else:
        st.info(f"{FALLBACK_MODEL_EMOJI} Standard quotas used. Using fallback: {FALLBACK_MODEL_ID}")
        logging.info(f"All standard quotas used. Using fallback model: {FALLBACK_MODEL_ID}")
    
    response_content, api_ok = "", True
    # Stream display logic adjusted for empty chat state
    if not is_new_empty_chat:
        with st.chat_message("assistant", avatar=avatar_resp):
            placeholder = st.empty()
            for chunk, err_msg in streamed(model_id_to_use, chat_history, max_tokens_api):
                if err_msg: response_content = f"❗ **API Error**: {err_msg}"; placeholder.error(response_content); api_ok=False; break
                if chunk: response_content += chunk; placeholder.markdown(response_content + "▌")
            if api_ok: placeholder.markdown(response_content)
    else: # For new empty chats, generate response but don't stream live, will show on rerun
        for chunk, err_msg in streamed(model_id_to_use, chat_history, max_tokens_api):
            if err_msg: response_content = f"❗ **API Error**: {err_msg}"; api_ok=False; break
            if chunk: response_content += chunk
            
    chat_history.append({"role":"assistant","content":response_content,"model": chosen_model_key})
    if api_ok and not use_fallback: record_use(chosen_model_key)
    if sessions[current_sid]["title"] == "New chat" and len(chat_history) >=2 : # Auto-title after first exchange
        sessions[current_sid]["title"] = _autoname(prompt)
        _delete_unused_blank_sessions(keep_sid=current_sid) # Clean up other potential blanks

    _save(SESS_FILE, sessions)
    st.rerun()

# ───────────────────────── Self-Relaunch ──────────────────────
if __name__ == "__main__" and os.getenv("_IS_STRL") != "1":
    os.environ["_IS_STRL"] = "1"; port = os.getenv("PORT", "8501")
    cmd = [sys.executable, "-m", "streamlit", "run", __file__, "--server.port", port, "--server.address", "0.0.0.0"]
    logging.info(f"Relaunching with Streamlit: {' '.join(cmd)}"); subprocess.run(cmd, check=False)
