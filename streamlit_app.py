#!/usr/bin/env python3
"""
OpenRouter Streamlit Chat — Full Edition
• Persistent chat sessions
• Daily/weekly/monthly quotas (“6-2-1 / 3-1 / Unlimited”)
• Pretty ‘token-jar’ gauges (fixed at top + live refresh)
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
    """Lenient JSON loader."""
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default

def _save(path: Path, obj):
    """Pretty-print and save JSON."""
    path.write_text(json.dumps(obj, indent=2))

def _today():    return date.today().isoformat()
def _yweek():    return datetime.now(TZ).strftime("%G-%V")  # ISO week-year
def _ymonth():   return datetime.now(TZ).strftime("%Y-%m")


# ───────────────────── Quota Management ────────────────────────

def _reset(block: dict, key: str, stamp: str, zeros: dict):
    """If the timestamp in block[key] != stamp, reset its usage."""
    if block.get(key) != stamp:
        block[key] = stamp
        block[f"{key}_u"] = zeros.copy()

def _load_quota():
    """Initialize or reset daily/weekly/monthly usage blocks."""
    zeros = {k: 0 for k in MODEL_MAP}
    q = _load(QUOTA_FILE, {})
    _reset(q, "d", _today(), zeros)
    _reset(q, "w", _yweek(), zeros)
    _reset(q, "m", _ymonth(), zeros)
    _save(QUOTA_FILE, q)
    return q

quota = _load_quota()

def remaining(key: str):
    """Return (daily_left, weekly_left, monthly_left) for model key."""
    ud = quota["d_u"].get(key, 0)
    uw = quota["w_u"].get(key, 0)
    um = quota["m_u"].get(key, 0)
    ld, lw, lm = PLAN[key]
    return ld - ud, lw - uw, lm - um

def record_use(key: str):
    """Increment usage counters after a successful call."""
    for blk in ("d_u", "w_u", "m_u"):
        quota[blk][key] = quota[blk].get(key, 0) + 1
    _save(QUOTA_FILE, quota)


# ───────────────────── Session Management ───────────────────────

sessions = _load(SESS_FILE, {})

def _new_sid():
    """Create a new chat session ID and persist."""
    sid = str(int(time.time() * 1000))
    sessions[sid] = {"title": "New chat", "messages": []}
    _save(SESS_FILE, sessions)
    return sid

def _autoname(seed: str) -> str:
    """Generate a short title (<25 chars) from the first user message."""
    words = seed.strip().split()
    cand = " ".join(words[:3]) or "Chat"
    return (cand[:25] + "…") if len(cand) > 25 else cand


# ─────────────────────────── Logging ────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout
)


# ──────────────────────────── API Calls ──────────────────────────

def api_post(payload: dict, *, stream: bool=False, timeout: int=DEFAULT_TIMEOUT):
    """Low-level POST to /chat/completions."""
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
    """
    Yield (chunk_text, error_msg) pairs.
    chunk_text is str or None; error_msg is str or None.
    """
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


# ────────────────────────── Model Routing ───────────────────────

def route_choice(user_msg: str, allowed: list[str]) -> str:
    """Call the router LLM to pick one letter from allowed."""
    if not allowed:
        logging.warning("route_choice called with empty allowed list.")
        return allowed[0] if allowed else "F"

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
    payload_r = {"model": ROUTER_MODEL_ID, "messages": router_messages}
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

    logging.warning("Router fallback to first allowed: %s", allowed[0])
    return allowed[0]


# ───────────────────── Credits Endpoint ────────────────────────

def get_credits() -> tuple[float|None,float|None,float|None]:
    """Fetch total / used credits from the API."""
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

# Initialize session state
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

    # 1) Token-Jar gauges pinned at the top
    st.subheader("Daily Token-Jars")
    cols = st.columns(len(MODEL_MAP))
    for i, m in enumerate(sorted(MODEL_MAP)):
        left, _, _ = remaining(m)
        lim, _, _ = PLAN[m]
        pct = 1.0 if lim > 900_000 else max(0.0, left / lim if lim>0 else 0.0)
        fill = int(pct * 100)
        color = "#4caf50" if pct > .5 else "#ff9800" if pct > .25 else "#f44336"
        cols[i].markdown(f"""
            <div style="width:44px;margin:auto;text-align:center;font-size:12px">
              <div style="border:2px solid #555;border-radius:6px;height:60px;position:relative;overflow:hidden;background:#f0f0f0;">
                <div style="position:absolute;bottom:0;width:100%;height:{fill}%;background:{color};"></div>
                <div style="position:absolute;top:2px;width:100%;font-size:18px">{EMOJI[m]}</div>
                <div style="position:absolute;bottom:2px;width:100%;font-size:10px;font-weight:bold">{m}</div>
              </div>
              <span>{'∞' if lim>900_000 else left}</span>
            </div>""", unsafe_allow_html=True)
    st.markdown("---")

    # 2) New Chat button
    if st.button("➕ New chat", use_container_width=True):
        st.session_state.sid = _new_sid()
        st.experimental_rerun()

    # 3) Chat session list
    st.subheader("Chats")
    for sid in sorted(sessions.keys(), key=int, reverse=True):
        title = sessions[sid]["title"][:25] or "Untitled"
        if st.button(title, key=sid, use_container_width=True):
            st.session_state.sid = sid
            st.experimental_rerun()

    st.markdown("---")

    # 4) Model-routing info
    st.subheader("Model-Routing Map")
    st.caption(f"Router engine: `{ROUTER_MODEL_ID}`")
    with st.expander("Letters → Models"):
        for k in MODEL_MAP:
            st.markdown(f"**{k}**: {MODEL_DESCRIPTIONS[k]} (max_output={MAX_TOKENS[k]})")

    st.markdown("---")

    # 5) Live credit stats
    tot, used, rem = (
        st.session_state.credits["total"],
        st.session_state.credits["used"],
        st.session_state.credits["remaining"],
    )
    with st.expander("Account stats (credits)"):
        if st.button("Refresh Credits", key="refresh_credits"):
            st.session_state.credits = dict(zip(
                ("total","used","remaining"),
                get_credits()
            ))
            st.session_state.credits_ts = time.time()
            tot, used, rem = (
                st.session_state.credits["total"],
                st.session_state.credits["used"],
                st.session_state.credits["remaining"]
            )
        if tot is None:
            st.warning("Could not fetch credits.")
        else:
            st.markdown(f"**Purchased:** {tot:.2f} cr")
            st.markdown(f"**Used:** {used:.2f} cr")
            st.markdown(f"**Remaining:** {rem:.2f} cr")
            st.caption(f"Last updated: {datetime.fromtimestamp(st.session_state.credits_ts).strftime('%Y-%m-%d %H:%M:%S')}")


# ────────────────────────── Main Chat Panel ─────────────────────

sid = st.session_state.sid
if sid not in sessions:
    st.session_state.sid = _new_sid()
    sid = st.session_state.sid

chat = sessions[sid]["messages"]

# Display existing messages
for msg in chat:
    avatar = "👤" if msg["role"] == "user" else EMOJI.get(msg.get("model","F"), "🤖")
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Input box
if prompt := st.chat_input("Ask anything…"):
    # Append user message
    chat.append({"role":"user","content":prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Determine allowed models by daily quota
    allowed = [k for k in MODEL_MAP if remaining(k)[0] > 0]
    if not allowed:
        st.error("❗ All daily quotas exhausted. Try again tomorrow.")
        st.stop()

    # Route to a model
    chosen = route_choice(prompt, allowed)
    model_id = MODEL_MAP[chosen]
    max_out = MAX_TOKENS[chosen]

    # Stream the assistant response
    with st.chat_message("assistant", avatar=EMOJI[chosen]):
        placeholder, full = st.empty(), ""
        stream_ok = True
        for chunk, err in streamed(model_id, chat, max_out):
            if err:
                full = f"❗ **API Error**: {err}"
                placeholder.error(full)
                stream_ok = False
                break
            if chunk:
                full += chunk
                placeholder.markdown(full + "▌")
        placeholder.markdown(full)

    # Save assistant message
    chat.append({"role":"assistant","content":full,"model":chosen})

    # Record usage & refresh token jars
    if stream_ok:
        record_use(chosen)
        # auto-title the session if it's still "New chat"
        if sessions[sid]["title"] == "New chat":
            sessions[sid]["title"] = _autoname(prompt)
        _save(SESS_FILE, sessions)
        # Immediately rerun to refresh sidebar jars
        st.experimental_rerun()


# ───────────────────────── Self-Relaunch ──────────────────────

if __name__ == "__main__" and os.getenv("_IS_STRL") != "1":
    os.environ["_IS_STRL"] = "1"
    port = os.getenv("PORT", "8501")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", __file__,
        "--server.port", port, "--server.address", "0.0.0.0"
    ], check=False)
