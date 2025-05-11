#!/usr/bin/env python3
"""
OpenRouter Streamlit Chat — Full Edition
• Persistent chat sessions
• Daily/weekly/monthly quotas (“6-2-1 / 3-1 / Unlimited”)
• Pretty ‘token-jar’ gauges (fixed at top)
• Detailed model-routing panel (Mistral router)
• Live credit/usage stats (GET /credits)
• Auto-titling of new chats
• Comprehensive logging
• Self-relaunch under `python main.py`
"""

# ───────────────────────── Imports ─────────────────────────
import json, logging, os, sys, subprocess, time, requests
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo
import streamlit as st

# ────────────────────────── Configuration ───────────────────
OPENROUTER_API_KEY  = "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"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT     = 120

# Model definitions
MODEL_MAP = {
    "A": "google/gemini-2.5-pro-preview",
    "B": "openai/o4-mini",
    "C": "openai/chatgpt-4o-latest",
    "D": "deepseek/deepseek-r1",
    "E": "x-ai/grok-3-beta",
    "F": "google/gemini-2.5-flash-preview"
}
# Router uses Mistral 7B Instruct
ROUTER_MODEL_ID = "mistralai/mistral-7b-instruct:free"

# Token limits for outputs
MAX_TOKENS = {
    "A": 16_000, "B": 8_000, "C": 16_000,
    "D": 8_000,  "E": 4_000, "F": 8_000
}

# Quota plan: (daily, weekly, monthly)
PLAN = {
    "A": (6, 45, 180),
    "B": (2, 15, 60),
    "C": (1, 8, 30),
    "D": (3, 25, 100),
    "E": (1, 10, 40),
    "F": (999_999, 50, 190)
}

# Emojis for jars
EMOJI = {
    "A": "🌟",
    "B": "🔷",
    "C": "🟥",
    "D": "🟢",
    "E": "🟡",
    "F": "🌀"
}

# Descriptions shown to the router
MODEL_DESCRIPTIONS = {
    "A": "🌟 (gemini-2.5-pro-preview) – top-quality, creative, expensive.",
    "B": "🔷 (o4-mini) – mid-stakes reasoning, cost-effective.",
    "C": "🟥 (chatgpt-4o-latest) – polished/empathetic, pricier.",
    "D": "🟢 (deepseek-r1) – cheap factual reasoning.",
    "E": "🟡 (grok-3-beta) – edgy style, second opinion.",
    "F": "🌀 (gemini-2.5-flash-preview) – quick, free-tier."
}

# Timezone for weekly/monthly resets
TZ = ZoneInfo("Australia/Sydney")

# Paths for persistence
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
    if block.get(key) != stamp:
        block[key] = stamp
        block[f"{key}_u"] = zeros.copy()

def _load_quota():
    zeros = {k: 0 for k in MODEL_MAP}
    q = _load(QUOTA_FILE, {})
    _reset(q, "d", _today(), zeros)
    _reset(q, "w", _yweek(), zeros)
    _reset(q, "m", _ymonth(), zeros)
    _save(QUOTA_FILE, q)
    return q

quota = _load_quota()

def remaining(key: str):
    ud = quota["d_u"].get(key, 0)
    uw = quota["w_u"].get(key, 0)
    um = quota["m_u"].get(key, 0)
    ld, lw, lm = PLAN[key]
    return ld - ud, lw - uw, lm - um

def record_use(key: str):
    for blk in ("d_u", "w_u", "m_u"):
        quota[blk][key] = quota[blk].get(key, 0) + 1
    _save(QUOTA_FILE, quota)


# ───────────────────── Session Management ───────────────────────

sessions = _load(SESS_FILE, {})

def _new_sid():
    sid = str(int(time.time() * 1000))
    sessions[sid] = {"title": "New chat", "messages": []}
    _save(SESS_FILE, sessions)
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
            if not line or not line.startswith(b"data: "):
                continue
            data = line[6:].decode("utf-8").strip()
            if data == "[DONE]":
                break
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
            if delta is not None:
                yield delta, None


# ───────────────────────── Model Routing ───────────────────────

def route_choice(user_msg: str, allowed: list[str]) -> str:
    if not allowed:
        logging.warning("route_choice called with empty allowed list.")
        return "F" 

    if len(allowed) == 1:
        logging.info(f"Router: Only one model allowed {allowed[0]}, selecting it directly.")
        return allowed[0]

    system_lines = [
        "You are an intelligent model-routing assistant.",
        "Select ONLY one letter from the following available models:",
    ]
    for k in allowed:
        system_lines.append(f"- {k}: {MODEL_DESCRIPTIONS[k]}")
    system_lines.append(
        "Based on the user's query, choose the letter that best balances quality, speed, and cost-sensitivity."
    )
    system_lines.append("Respond with ONLY the single capital letter. No extra text.")

    router_messages = [
        {"role": "system", "content": "\n".join(system_lines)},
        {"role": "user",   "content": user_msg}
    ]
    payload_r = {"model": ROUTER_MODEL_ID, "messages": router_messages, "max_tokens": 5}
    try:
        r = api_post(payload_r)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip().upper()
        logging.info(f"Router raw response: {text}")
        for ch in text:
            if ch in allowed:
                return ch
    except Exception as e:
        logging.error(f"Router call error: {e}")

    fallback_choice = allowed[0]
    logging.warning(f"Router fallback to first allowed model: {fallback_choice}")
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

if "sid" not in st.session_state:
    st.session_state.sid = _new_sid()
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

    # Token-Jar gauges pinned at the top
    st.subheader("Daily Token-Jars")
    cols = st.columns(len(MODEL_MAP))
    for i, m in enumerate(sorted(MODEL_MAP)):
        left, _, _ = remaining(m)
        lim, _, _  = PLAN[m]
        pct = 1.0 if lim > 900_000 else max(0.0, left / lim if lim > 0 else 0.0)
        fill = int(pct * 100)
        color = "#4caf50" if pct > .5 else "#ff9800" if pct > .25 else "#f44336"
        
        # Jar UI with CSS comments removed
        cols[i].markdown(f"""
            <div style="width:44px; margin:auto; text-align:center;">
              <div style="
                height:60px; 
                border:1px solid #ccc; 
                border-radius:7px;    
                background:#f5f5f5; 
                position:relative;
                overflow:hidden; 
                box-shadow: inset 0 1px 2px rgba(0,0,0,0.07), 
                            0 1px 1px rgba(0,0,0,0.05); 
              ">
                <div style=" 
                  position:absolute;
                  bottom:0;
                  width:100%;
                  height:{fill}%;
                  background:{color}; 
                  box-shadow: inset 0 2px 2px rgba(255,255,255,0.3); 
                  box-sizing: border-box;
                "></div>
                <div style=" 
                  position:absolute;
                  top:2px; 
                  width:100%;
                  font-size:18px; 
                  line-height:1; 
                ">{EMOJI[m]}</div>  
                <div style=" 
                  position:absolute;
                  bottom:2px; 
                  width:100%;
                  font-size:11px; 
                  font-weight:bold;
                  color:#555; 
                  line-height:1;
                ">{m}</div>
              </div>
              <span style=" 
                display:block; 
                margin-top:4px;
                font-size:11px;
                font-weight:600; 
                color:#333; 
                line-height:1;
              ">
                {'∞' if lim > 900_000 else left}
              </span>
            </div>""", unsafe_allow_html=True)
    st.markdown("---")

    # New Chat button
    if st.button("➕ New chat", use_container_width=True):
        st.session_state.sid = _new_sid()
        st.experimental_rerun() 

    # Chat session list
    st.subheader("Chats")
    sorted_sids = sorted(sessions.keys(), key=lambda s: int(s), reverse=True)
    for sid_key in sorted_sids:
        title = sessions[sid_key]["title"][:25] + ("…" if len(sessions[sid_key]["title"]) > 25 else "") or "Untitled"
        if st.button(title, key=f"session_button_{sid_key}", use_container_width=True):
            if st.session_state.sid != sid_key: 
                st.session_state.sid = sid_key
                st.experimental_rerun()

    st.markdown("---")

    # Model-routing info
    st.subheader("Model-Routing Map")
    st.caption(f"Router engine: `{ROUTER_MODEL_ID}`")
    with st.expander("Letters → Models"):
        for k_model in sorted(MODEL_MAP.keys()): 
            st.markdown(f"**{k_model}**: {MODEL_DESCRIPTIONS[k_model]} (max_output={MAX_TOKENS[k_model]:,})")

    st.markdown("---")

    # Live credit stats
    tot, used, rem = (
        st.session_state.credits["total"],
        st.session_state.credits["used"],
        st.session_state.credits["remaining"],
    )
    with st.expander("Account stats (credits)", expanded=False): 
        if st.button("Refresh Credits", key="refresh_credits_button"): 
            st.session_state.credits = dict(zip(
                ("total","used","remaining"),
                get_credits()
            ))
            st.session_state.credits_ts = time.time()
            tot, used, rem = (
                st.session_state.credits["total"],
                st.session_state.credits["used"],
                st.session_state.credits["remaining"],
            )
            st.experimental_rerun() 

        if tot is None:
            st.warning("Could not fetch credits.")
        else:
            st.markdown(f"**Purchased:** {tot:.2f} cr")
            st.markdown(f"**Used:** {used:.2f} cr")
            st.markdown(f"**Remaining:** {rem:.2f} cr")
            try:
                last_updated_str = datetime.fromtimestamp(st.session_state.credits_ts).strftime('%Y-%m-%d %H:%M:%S')
                st.caption(f"Last updated: {last_updated_str}")
            except TypeError: 
                st.caption("Last updated: N/A")


# ────────────────────────── Main Chat Panel ─────────────────────

current_sid = st.session_state.sid 
if current_sid not in sessions: 
    st.error("Selected chat session not found. Creating a new one.")
    current_sid = _new_sid()
    st.session_state.sid = current_sid
    st.experimental_rerun()


chat_history = sessions[current_sid]["messages"] 

# Display existing messages
for msg in chat_history: 
    avatar_key = msg.get("model", "F") if msg["role"] == "assistant" else "user" 
    avatar_map = {"user": "👤", **EMOJI} 
    avatar = avatar_map.get(avatar_key, "🤖") # This is approx line 316
    with st.chat_message(msg["role"], avatar=avatar):                     
        st.markdown(msg["content"])                                       

# Input box
if prompt := st.chat_input("Ask anything…"):
    chat_history.append({"role":"user","content":prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    allowed_models = [k for k in MODEL_MAP if remaining(k)[0] > 0]
    if not allowed_models:
        st.error("❗ All daily quotas exhausted. Try again tomorrow.")
        st.stop()

    chosen_model_key = route_choice(prompt, allowed_models)
    model_id_to_use = MODEL_MAP[chosen_model_key]
    max_tokens_for_model  = MAX_TOKENS[chosen_model_key]

    with st.chat_message("assistant", avatar=EMOJI[chosen_model_key]):
        response_placeholder, full_response_content = st.empty(), ""
        api_call_ok = True
        for chunk, error_message in streamed(model_id_to_use, chat_history, max_tokens_for_model):
            if error_message:
                full_response_content = f"❗ **API Error**: {error_message}"
                response_placeholder.error(full_response_content)
                api_call_ok = False
                break
            if chunk:
                full_response_content += chunk
                response_placeholder.markdown(full_response_content + "▌")
        response_placeholder.markdown(full_response_content)

    chat_history.append({"role":"assistant","content":full_response_content,"model":chosen_model_key})

    if api_call_ok: 
        record_use(chosen_model_key)
        if sessions[current_sid]["title"] == "New chat":
            sessions[current_sid]["title"] = _autoname(prompt)
    
    _save(SESS_FILE, sessions) 
    st.experimental_rerun() 


# ───────────────────────── Self-Relaunch ──────────────────────

if __name__ == "__main__" and os.getenv("_IS_STRL") != "1":
    os.environ["_IS_STRL"] = "1"
    port = os.getenv("PORT", "8501")
    cmd = [
        sys.executable, "-m", "streamlit", "run", __file__,
        "--server.port", port, 
        "--server.address", "0.0.0.0",
        "--server.runOnSave", "false", 
        "--client.toolbarMode", "minimal" 
    ]
    logging.info(f"Relaunching with Streamlit: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)
