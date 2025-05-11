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
• In-app API Key configuration (via Settings panel)
"""

# ───────────────────────── Imports ─────────────────────────
import json, logging, os, sys, subprocess, time, requests
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo
import streamlit as st

# ────────────────────────── Configuration ───────────────────
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
PLAN = {"A": (10,70,300), "B": (5,35,150), "C": (1,7,30), "D": (4,28,120), "F": (180,500,2000)}
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
CONFIG_FILE = DATA_DIR / "app_config.json"


# ──────────────────────── Helper Functions ───────────────────────
def _load(path: Path, default):
    try: return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError): return default

def _save(path: Path, obj): path.write_text(json.dumps(obj, indent=2))
def _today(): return date.today().isoformat()
def _yweek(): return datetime.now(TZ).strftime("%G-%V")
def _ymonth(): return datetime.now(TZ).strftime("%Y-%m")

def _load_app_config(): return _load(CONFIG_FILE, {})
def _save_app_config(api_key_value: str):
    config_data = _load_app_config()
    config_data["openrouter_api_key"] = api_key_value
    _save(CONFIG_FILE, config_data)

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
    _save(QUOTA_FILE, q)
    return q
quota = _load_quota()

def remaining(key: str):
    ud, uw, um = quota.get("d_u", {}).get(key,0), quota.get("w_u", {}).get(key,0), quota.get("m_u", {}).get(key,0)
    if key not in PLAN: logging.error(f"Unknown key for remaining quota: {key}"); return 0,0,0
    ld,lw,lm = PLAN[key]
    return ld-ud, lw-uw, lm-um

def record_use(key: str):
    if key not in MODEL_MAP: logging.warning(f"Unknown key for record_use: {key}"); return
    for blk_key in ("d_u", "w_u", "m_u"):
        if blk_key not in quota: quota[blk_key] = {k:0 for k in MODEL_MAP}
        quota[blk_key][key] = quota[blk_key].get(key,0)+1
    _save(QUOTA_FILE, quota)

# ───────────────────── Session Management ───────────────────────
def _delete_unused_blank_sessions(keep_sid:str=None):
    sids_to_delete = [sid for sid,data in sessions.items() if sid != keep_sid and data.get("title")=="New chat" and not data.get("messages")]
    if sids_to_delete:
        for sid_del in sids_to_delete: logging.info(f"Auto-deleting blank session: {sid_del}"); del sessions[sid_del]
        return True
    return False
sessions = _load(SESS_FILE, {})
def _new_sid():
    _delete_unused_blank_sessions(); sid = str(int(time.time()*1000))
    sessions[sid] = {"title":"New chat", "messages":[]}
    return sid
def _autoname(seed:str) -> str:
    words = seed.strip().split(); cand = " ".join(words[:3]) or "Chat"
    return (cand[:25]+"…") if len(cand)>25 else cand

# ─────────────────────────── Logging ────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)

# ────────────────────────── API Calls ──────────────────────────
def api_post(payload: dict, *, stream: bool=False, timeout: int=DEFAULT_TIMEOUT):
    active_api_key = st.session_state.get("openrouter_api_key")
    if not (active_api_key and active_api_key.startswith("sk-or-")): # Check validity here too
        raise ValueError("OpenRouter API Key is not set or invalid in session state for api_post.")
    headers = {"Authorization":f"Bearer {active_api_key}", "Content-Type":"application/json"}
    logging.info(f"POST /chat/completions → model={payload.get('model')}, stream={stream}, max_tokens={payload.get('max_tokens')}")
    return requests.post(f"{OPENROUTER_API_BASE}/chat/completions", headers=headers, json=payload, stream=stream, timeout=timeout)

def streamed(model: str, messages: list, max_tokens_out: int):
    payload = {"model":model, "messages":messages, "stream":True, "max_tokens":max_tokens_out}
    try:
        with api_post(payload, stream=True) as r:
            try: r.raise_for_status()
            except requests.exceptions.HTTPError as e:
                text, scode = r.text, e.response.status_code; logging.error(f"Stream HTTPError {scode}: {text}")
                yield None, f"HTTP {scode}: {'Unauthorized. Check API Key.' if scode==401 else ''} {text}"; return
            for line in r.iter_lines():
                if not line: continue
                line_str = line.decode("utf-8")
                if line_str.startswith(": OPENROUTER PROCESSING"): logging.info(f"OR PING: {line_str.strip()}"); continue
                if not line_str.startswith("data: "): logging.warning(f"Unexpected line: {line}"); continue
                data = line_str[6:].strip()
                if data == "[DONE]": break
                try: chunk = json.loads(data)
                except json.JSONDecodeError: logging.error(f"Bad JSON: {data}"); yield None,"Error decoding chunk"; return
                if "error" in chunk: msg=chunk["error"].get("message","Unknown API err"); logging.error(f"API chunk err: {msg}"); yield None,msg; return
                delta = chunk["choices"][0]["delta"].get("content")
                if delta is not None: yield delta, None
    except ValueError as ve: logging.error(f"ValueError in streamed: {ve}"); yield None, str(ve)
    except Exception as e: logging.error(f"Streamed API call failed: {e}"); yield None, f"Request failed: {e}"

# ───────────────────────── Model Routing ───────────────────────
def route_choice(user_msg: str, allowed: list[str]) -> str:
    active_api_key = st.session_state.get("openrouter_api_key")
    fallback_choice = "F" if "F" in allowed else (allowed[0] if allowed else "F")
    if not (active_api_key and active_api_key.startswith("sk-or-")):
        logging.warning(f"Router: API Key invalid/missing. Fallback: {fallback_choice}"); return fallback_choice
    if not allowed: logging.warning("route_choice empty allowed. Default 'F'"); return "F"
    if len(allowed)==1: logging.info(f"Router: 1 allowed {allowed[0]}"); return allowed[0]
    sys_lines = ["Select ONLY one letter:"] + [f"- {k}: {MODEL_DESCRIPTIONS[k]}" for k in allowed if k in MODEL_DESCRIPTIONS] + ["Choose best letter for user query. ONLY single capital letter."]
    router_msgs = [{"role":"system","content":"\n".join(sys_lines)}, {"role":"user","content":user_msg}]
    payload_r = {"model":ROUTER_MODEL_ID, "messages":router_msgs, "max_tokens":10}
    try:
        r = api_post(payload_r); r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip().upper()
        logging.info(f"Router raw: {text}")
        for ch in text:
            if ch in allowed: return ch
    except ValueError as ve: logging.error(f"ValueError in router: {ve}") # Key error from api_post
    except requests.exceptions.HTTPError as e: logging.error(f"Router HTTPError {e.response.status_code}: {e.response.text}")
    except Exception as e: logging.error(f"Router error: {e}")
    logging.warning(f"Router fallback: {fallback_choice}"); return fallback_choice

# ───────────────────── Credits Endpoint ───────────────────────
def get_credits():
    active_api_key = st.session_state.get("openrouter_api_key")
    if not (active_api_key and active_api_key.startswith("sk-or-")):
        logging.warning("Credits: API Key invalid/missing."); return None,None,None
    try:
        r = requests.get(f"{OPENROUTER_API_BASE}/credits", headers={"Authorization":f"Bearer {active_api_key}"}, timeout=10)
        r.raise_for_status(); d = r.json()["data"]
        return d["total_credits"], d["total_usage"], d["total_credits"]-d["total_usage"]
    except requests.exceptions.HTTPError as e: logging.warning(f"Credits HTTPError {e.response.status_code}: {e.response.text}"); return None,None,None
    except Exception as e: logging.warning(f"Credits error: {e}"); return None,None,None

# ───────────────────────── UI Styling ──────────────────────────
def load_custom_css():
    st.markdown("""<style>...</style>""", unsafe_allow_html=True) # Keep CSS as is for brevity

# ───────────────────────── Streamlit UI ────────────────────────
st.set_page_config(page_title="OpenRouter Chat", layout="wide", initial_sidebar_state="expanded")

# Initialize API key in session state from config file
if "openrouter_api_key" not in st.session_state:
    app_conf = _load_app_config()
    st.session_state.openrouter_api_key = app_conf.get("openrouter_api_key", None)

load_custom_css() # Assuming your CSS is complete and correct

# Initial SID Management
needs_rerun_startup = False
if "sid" not in st.session_state or st.session_state.sid not in sessions:
    st.session_state.sid = _new_sid(); needs_rerun_startup = True
elif _delete_unused_blank_sessions(keep_sid=st.session_state.sid): needs_rerun_startup = True
if needs_rerun_startup: _save(SESS_FILE, sessions); st.rerun()

if "credits" not in st.session_state:
    st.session_state.credits = dict(zip(("total","used","remaining"), get_credits()))
    st.session_state.credits_ts = time.time()

# ───────────────────────── Sidebar ─────────────────────────────
with st.sidebar:
    # --- Settings Panel Logic ---
    # Initialize settings_panel_open state (MUST be done before the button that might change it)
    if "settings_panel_open" not in st.session_state:
        current_api_key_init = st.session_state.get("openrouter_api_key")
        st.session_state.settings_panel_open = not (current_api_key_init and isinstance(current_api_key_init, str) and current_api_key_init.startswith("sk-or-"))

    if st.button("⚙️ Settings", key="toggle_settings_button", use_container_width=True):
        st.session_state.settings_panel_open = not st.session_state.get("settings_panel_open", False)
        # Implicit rerun on button click will show/hide the panel

    if st.session_state.get("settings_panel_open"):
        with st.container(): # Visually group settings
            st.markdown("---")
            st.subheader("API Key Configuration")
            
            current_api_key_in_panel = st.session_state.get("openrouter_api_key")
            key_display = "Not set"
            is_current_key_valid_format = False
            if current_api_key_in_panel and isinstance(current_api_key_in_panel, str):
                key_display = f"••••••••{current_api_key_in_panel[-4:]}" if len(current_api_key_in_panel) > 8 else "••••"
                if current_api_key_in_panel.startswith("sk-or-"):
                    is_current_key_valid_format = True
            st.caption(f"Current key: {key_display}")

            if current_api_key_in_panel and not is_current_key_valid_format:
                 st.warning("The saved API key appears invalid. Please enter a valid key (starts with 'sk-or-').")

            new_key_input = st.text_input(
                "Enter new OpenRouter API Key", type="password", key="api_key_settings_input", placeholder="sk-or-..."
            )
            if st.button("Save API Key", key="save_api_key_settings_button"):
                if new_key_input and new_key_input.startswith("sk-or-"):
                    st.session_state.openrouter_api_key = new_key_input
                    _save_app_config(new_key_input)
                    st.success("API Key saved!")
                    st.session_state.credits = dict(zip(("total","used","remaining"), get_credits()))
                    st.session_state.credits_ts = time.time()
                    st.session_state.settings_panel_open = False # Close panel on success
                    time.sleep(0.5)
                    st.rerun()
                elif not new_key_input: st.warning("Please enter an API key.")
                else: st.error("Invalid API key format. It should start with 'sk-or-'.")
            
            if st.button("Close Settings", key="close_settings_panel", use_container_width=True):
                st.session_state.settings_panel_open = False
                st.rerun() # Rerun to hide panel
            st.markdown("---")
    # --- End Settings Panel ---

    st.image("https://avatars.githubusercontent.com/u/130328222?s=200&v=4", width=50)
    st.title("OpenRouter Chat")
    st.divider()

    st.subheader("Daily Jars (Msgs Left)")
    active_keys = sorted(MODEL_MAP.keys())
    cols = st.columns(len(active_keys))
    for i, mk in enumerate(active_keys):
        left,_,_ = remaining(mk); lim,_,_ = PLAN[mk]
        pct = 1.0 if lim > 900_000 else max(0.0, left/lim if lim > 0 else 0.0)
        fill = int(pct*100); color = "#4caf50" if pct>.5 else ("#ffc107" if pct>.25 else "#f44336")
        cols[i].markdown(f"""<div class="token-jar-container"><div class="token-jar"><div class="token-jar-fill" style="height:{fill}%; background-color:{color};"></div><div class="token-jar-emoji">{EMOJI[mk]}</div><div class="token-jar-key">{mk}</div></div><span class="token-jar-remaining">{'∞' if lim>900_000 else left}</span></div>""", unsafe_allow_html=True)
    st.divider()

    blank_current = sessions[st.session_state.sid]["title"]=="New chat" and not sessions[st.session_state.sid]["messages"]
    if st.button("➕ New chat", key="new_chat_top", use_container_width=True, disabled=blank_current):
        st.session_state.sid = _new_sid(); _save(SESS_FILE, sessions); st.rerun()
    elif blank_current: st.caption("Current chat empty. Add message or switch.")

    st.subheader("Chats")
    for sid_k in sorted(sessions.keys(), key=lambda s: int(s), reverse=True):
        title = sessions[sid_k].get("title","Untitled"); disp_title = title[:25]+("…"if len(title)>25 else "")
        if st.session_state.sid == sid_k: disp_title = f"🔹 {disp_title}"
        if st.button(disp_title, key=f"sess_btn_{sid_k}", use_container_width=True):
            if st.session_state.sid != sid_k:
                st.session_state.sid = sid_k
                if _delete_unused_blank_sessions(keep_sid=sid_k): _save(SESS_FILE, sessions)
                st.rerun()
    st.divider()

    st.subheader("Model-Routing Map")
    st.caption(f"Router: {ROUTER_MODEL_ID}")
    with st.expander("Letters → Models", expanded=False):
        for k_m in sorted(MODEL_MAP.keys()): st.markdown(f"**{k_m}**: {MODEL_DESCRIPTIONS[k_m]} (max_out={MAX_TOKENS[k_m]:,})")
    st.divider()

    tot,usd,rem = st.session_state.credits.get("total"), st.session_state.credits.get("used"), st.session_state.credits.get("remaining")
    with st.expander("Account stats (credits)", expanded=False):
        if st.button("Refresh Credits", key="refresh_credits_btn"):
            st.session_state.credits = dict(zip(("total","used","remaining"), get_credits()))
            st.session_state.credits_ts = time.time(); st.rerun()
        
        active_api_key_for_credits = st.session_state.get("openrouter_api_key")
        valid_key_for_credits = active_api_key_for_credits and isinstance(active_api_key_for_credits, str) and active_api_key_for_credits.startswith("sk-or-")

        if tot is None:
            if not valid_key_for_credits: st.warning("Set valid API Key for credits.")
            else: st.warning("Could not fetch. Check key/network.")
        else:
            st.markdown(f"**Purchased:** ${tot:.2f} cr\n**Used:** ${usd:.2f} cr\n**Remaining:** ${rem:.2f} cr")
            try: last_upd = datetime.fromtimestamp(st.session_state.credits_ts, TZ).strftime('%-d %b %Y, %H:%M:%S'); st.caption(f"Updated: {last_upd}")
            except TypeError: st.caption("Updated: N/A")

# ────────────────────────── Main Chat Panel ─────────────────────
current_sid = st.session_state.sid # Already validated
chat_history = sessions[current_sid]["messages"]
for msg in chat_history:
    role = msg["role"]; avatar = "👤"
    if role == "assistant":
        mkey = msg.get("model")
        avatar = FALLBACK_MODEL_EMOJI if mkey == FALLBACK_MODEL_KEY else EMOJI.get(mkey, EMOJI.get("F", "🤖"))
    with st.chat_message(role, avatar=avatar): st.markdown(msg["content"])

active_api_key_main = st.session_state.get("openrouter_api_key")
is_api_key_valid_for_chat = active_api_key_main and isinstance(active_api_key_main, str) and active_api_key_main.startswith("sk-or-")

if not is_api_key_valid_for_chat:
    st.warning("👋 Please set your OpenRouter API Key via '⚙️ Settings' in the sidebar to start chatting.")

if prompt := st.chat_input("Ask anything…", key=f"chat_input_{current_sid}", disabled=not is_api_key_valid_for_chat):
    chat_history.append({"role":"user","content":prompt})
    with st.chat_message("user", avatar="👤"): st.markdown(prompt)

    allowed_std_models = [k for k in MODEL_MAP if remaining(k)[0]>0]
    use_fallback, chosen_key, model_id, max_tkns, avatar_resp = False, None, None, None, "🤖"

    if not allowed_std_models:
        st.info(f"{FALLBACK_MODEL_EMOJI} Quotas used. Using fallback."); use_fallback=True
        chosen_key, model_id, max_tkns, avatar_resp = FALLBACK_MODEL_KEY, FALLBACK_MODEL_ID, FALLBACK_MODEL_MAX_TOKENS, FALLBACK_MODEL_EMOJI
    else:
        chosen_key = allowed_std_models[0] if len(allowed_std_models)==1 else route_choice(prompt, allowed_std_models)
        if chosen_key not in MODEL_MAP: # Router might return fallback 'F' or invalid if error
            logging.warning(f"Chosen key '{chosen_key}' not in MODEL_MAP. Using fallback.")
            st.info(f"{FALLBACK_MODEL_EMOJI} Model selection issue. Using fallback.")
            chosen_key,model_id,max_tkns,avatar_resp = FALLBACK_MODEL_KEY,FALLBACK_MODEL_ID,FALLBACK_MODEL_MAX_TOKENS,FALLBACK_MODEL_EMOJI; use_fallback=True
        else:
            model_id,max_tkns,avatar_resp = MODEL_MAP[chosen_key],MAX_TOKENS[chosen_key],EMOJI[chosen_key]

    with st.chat_message("assistant", avatar=avatar_resp):
        placeholder, full_resp, ok = st.empty(), "", True
        for chunk, err_msg in streamed(model_id, chat_history, max_tkns):
            if err_msg: full_resp=f"❗ **API Error**: {err_msg}"; placeholder.error(full_resp); ok=False; break
            if chunk: full_resp+=chunk; placeholder.markdown(full_resp+"▌")
        placeholder.markdown(full_resp)

    chat_history.append({"role":"assistant","content":full_resp,"model":chosen_key})
    if ok:
        if not use_fallback and chosen_key in MODEL_MAP: record_use(chosen_key)
        if sessions[current_sid]["title"]=="New chat" and chat_history:
            sessions[current_sid]["title"] = _autoname(prompt)
            _delete_unused_blank_sessions(keep_sid=current_sid)
    _save(SESS_FILE, sessions); st.rerun()

# ───────────────────────── Self-Relaunch ──────────────────────
if __name__=="__main__" and os.getenv("_IS_STRL")!="1":
    os.environ["_IS_STRL"]="1"; port=os.getenv("PORT","8501")
    cmd = [sys.executable,"-m","streamlit","run",__file__,"--server.port",port,"--server.address","0.0.0.0"]
    logging.info(f"Relaunching: {' '.join(cmd)}"); subprocess.run(cmd, check=False)
