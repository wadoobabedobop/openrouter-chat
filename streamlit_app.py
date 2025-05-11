#!/usr/bin/env python3
"""
OpenRouter Streamlit Chat — Replit edition
• Persistent chat sessions
• Daily/weekly/monthly quotas (“6-2-1 / 3-1 / Unlimited”)
• Pretty ‘token-jar’ gauges
• Model-routing panel restored
• Live credit/usage stats (GET /credits)
"""

# ───────── Imports ─────────
import json, logging, os, sys, subprocess, time, requests, math
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo
import streamlit as st

# ───────── Basic config ─────
OPENROUTER_API_KEY  = "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"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT     = 120

MODEL_MAP = {
    "A": "google/gemini-2.5-pro-preview",
    "B": "openai/o4-mini",
    "C": "openai/chatgpt-4o-latest",
    "D": "deepseek/deepseek-r1",
    "E": "x-ai/grok-3-beta",
    "F": "google/gemini-2.5-flash-preview"      # also used for router
}
ROUTER_MODEL_ID     = MODEL_MAP["F"]
MAX_TOKENS          = {"A":16_000,"B":8_000,"C":16_000,"D":8_000,"E":4_000,"F":8_000}

PLAN = {          # daily, weekly, monthly “limits”
    "A": (6,45,180), "B": (2,15,60), "C": (1,8,30),
    "D": (3,25,100),"E": (1,10,40), "F": (999_999,50,190)
}
EMOJI = {"A":"🌟","B":"🔷","C":"🟥","D":"🟢","E":"🟡","F":"🌀"}
TZ    = ZoneInfo("Australia/Sydney")

# ───────── Files ────────────
DATA_DIR   = Path(__file__).parent
SESS_FILE  = DATA_DIR/"chat_sessions.json"
QUOTA_FILE = DATA_DIR/"quotas.json"

# ───────── Small utils ──────
def _load(path, default):     # lenient JSON loader
    try: return json.loads(path.read_text())
    except: return default

def _save(path, obj):
    path.write_text(json.dumps(obj, indent=2))

def _today():       return date.today().isoformat()
def _yweek():       return datetime.now(TZ).strftime("%G-%V")
def _ymonth():      return datetime.now(TZ).strftime("%Y-%m")

# ───────── Persistent quota ─
def _reset(bucket, key, stamp, zeros):
    if bucket.get(key)!=stamp:
        bucket[key]=stamp; bucket[key+"_u"]=zeros.copy()

def _load_quota():
    zeros = {m:0 for m in MODEL_MAP}
    q     = _load(QUOTA_FILE,{})
    _reset(q,"d",_today(),zeros)
    _reset(q,"w",_yweek(),zeros)
    _reset(q,"m",_ymonth(),zeros)
    _save(QUOTA_FILE,q); return q
quota = _load_quota()

def remaining(m):
    used_d,used_w,used_m = quota["d_u"][m],quota["w_u"][m],quota["m_u"][m]
    lim_d,lim_w,lim_m    = PLAN[m]
    return lim_d-used_d, lim_w-used_w, lim_m-used_m

def record_use(m):
    quota["d_u"][m]+=1; quota["w_u"][m]+=1; quota["m_u"][m]+=1
    _save(QUOTA_FILE,quota)

# ───────── Persistent sessions ─
sessions = _load(SESS_FILE,{})
def _new_sid():
    sid=str(int(time.time()*1000)); sessions[sid]={"title":"New chat","messages":[]}
    _save(SESS_FILE,sessions); return sid

# ───────── Logging ──────────
logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(message)s")

def api_post(payload, *, stream=False, timeout=DEFAULT_TIMEOUT):
    h={"Authorization":f"Bearer {OPENROUTER_API_KEY}","Content-Type":"application/json"}
    return requests.post(f"{OPENROUTER_API_BASE}/chat/completions",
                         headers=h,json=payload,stream=stream,timeout=timeout)

# ───────── OpenRouter helpers ─
def streamed(model, msgs):
    p={"model":model,"messages":msgs,"stream":True}
    with api_post(p,stream=True) as r:
        r.raise_for_status()
        for ln in r.iter_lines():
            if not ln or not ln.startswith(b"data: "): continue
            data=ln[6:].decode().strip()
            if data=="[DONE]": break
            chunk=json.loads(data)
            if "error" in chunk:
                yield None, chunk["error"].get("message","Error"); return
            delta=chunk["choices"][0]["delta"].get("content")
            if delta: yield delta,None

def route_choice(user_msg, allowed):
    prompt=f"Choose ONE of {allowed}. No explanation."
    msgs=[{"role":"system","content":prompt},
          {"role":"user","content":user_msg}]
    try:
        r=api_post({"model":ROUTER_MODEL_ID,"messages":msgs})
        r.raise_for_status()
        text=r.json()["choices"][0]["message"]["content"].strip().upper()
        for ch in text:
            if ch in allowed: return ch
    except Exception as e: logging.warning("Router fail %s",e)
    return allowed[0]

# ───────── Credit stats (/credits) ─
def get_credits():
    try:
        r=requests.get(f"{OPENROUTER_API_BASE}/credits",
                       headers={"Authorization":f"Bearer {OPENROUTER_API_KEY}"},
                       timeout=10)
        r.raise_for_status()
        dat=r.json()["data"]; bought=dat["total_credits"]; used=dat["total_usage"]
        return bought, used, bought-used
    except Exception as e:
        logging.warning("Could not fetch credits: %s", e)
        return None, None, None
CRED_TOTAL, CRED_USED, CRED_LEFT = get_credits()

# ───────── Streamlit page config ─
st.set_page_config(page_title="OpenRouter Chat", layout="wide",
                   initial_sidebar_state="expanded")

# Session state
if "sid" not in st.session_state: st.session_state.sid=_new_sid()
if "credits_ts" not in st.session_state: st.session_state.credits_ts=time.time()

# ───────── Sidebar UI ─────────────────────────────
with st.sidebar:
    st.image("https://avatars.githubusercontent.com/u/130328222?s=200&v=4", width=48)
    st.title("OpenRouter Chat")

    if st.button("➕  New chat", use_container_width=True):
        st.session_state.sid=_new_sid(); st.rerun()

    # chat list
    st.subheader("Chats")
    for sid,obj in list(sessions.items())[::-1]:
        lbl=obj["title"][:25] or "Untitled"
        if st.button(lbl, key=sid, use_container_width=True):
            st.session_state.sid=sid; st.rerun()

    st.markdown("---")
    # pretty JAR display
    st.subheader("Daily Token-Jars")
    jar_cols=st.columns(len(MODEL_MAP))
    for idx,m in enumerate("ABCDEF"):
        left,maxd,_=remaining(m)[0],PLAN[m][0],PLAN[m][1]
        pct = 1 if PLAN[m][0]>900_000 else max(0,left/PLAN[m][0])
        fill = int(pct*100)
        jar_html=f"""
        <div style="width:44px;margin:auto;text-align:center;font-size:12px">
          <div style="border:2px solid #555;border-radius:6px;height:60px;
                      position:relative;overflow:hidden;background:#fdfdfd;">
              <div style="position:absolute;bottom:0;width:100%;height:{fill}%;
                          background:#4caf50;"></div>
              <div style="position:absolute;top:2px;width:100%;
                          font-size:18px">{EMOJI[m]}</div>
          </div>
          <span>{'∞' if PLAN[m][0]>900_000 else left}</span>
        </div>"""
        jar_cols[idx].markdown(jar_html, unsafe_allow_html=True)

    # routing info panel
    st.markdown("---")
    st.subheader("Model-routing map")
    st.caption(f"Router engine: **{ROUTER_MODEL_ID}**")
    with st.expander("Letters → Models"):
        for k,v in MODEL_MAP.items():
            st.markdown(f"**{k}** → `{v}`")

    # account stats
    with st.expander("Account stats (credits)"):
        if CRED_TOTAL is None:
            st.warning("Couldn’t fetch /credits endpoint.")
        else:
            st.markdown(f"**Purchased:** {CRED_TOTAL:.2f} cr")
            st.markdown(f"**Used:** {CRED_USED:.2f} cr")
            st.markdown(f"**Remaining:** {CRED_LEFT:.2f} cr")

# ───────── Chat panel ─────────────────────────────
chat = sessions[st.session_state.sid]["messages"]

# show history
for m in chat:
    avatar="👤" if m["role"]=="user" else "🤖"
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])

# input
if prompt := st.chat_input("Ask anything…"):
    chat.append({"role":"user","content":prompt})
    with st.chat_message("user", avatar="👤"): st.markdown(prompt)

    allowed=[m for m in MODEL_MAP if remaining(m)[0]>0]
    route=route_choice(prompt,allowed)
    model=MODEL_MAP[route]
    with st.chat_message("assistant", avatar="🤖"):
        box, full = st.empty(), ""
        for chunk, err in streamed(model, chat):
            if err: full=f"❗ {err}"; box.error(full); break
            if chunk: full+=chunk; box.markdown(full+"▌")
        box.markdown(full)
    chat.append({"role":"assistant","content":full})
    record_use(route); _save(SESS_FILE,sessions)

# ───────── Self-relaunch when run via `python main.py` ─
if __name__=="__main__" and os.getenv("_IS_STRL")!="1":
    os.environ["_IS_STRL"]="1"
    port=os.getenv("PORT","8501")
    subprocess.run(["streamlit","run",__file__,
                   "--server.port",port,"--server.address","0.0.0.0"])
