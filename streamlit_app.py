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
OPENROUTER_API_KEY  = "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" # Replace if this is a placeholder
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT     = 120

# YOUR ORIGINAL MODEL_MAP
MODEL_MAP = {
    "A": "google/gemini-2.5-pro-preview",
    "B": "openai/o4-mini",
    "C": "openai/chatgpt-4o-latest",
    "D": "deepseek/deepseek-r1",
    "E": "x-ai/grok-3-beta",
    "F": "google/gemini-2.5-flash-preview"      # also used for router
}
ROUTER_MODEL_ID     = MODEL_MAP["F"]
# Max TOKENS for OUTPUT of the main LLM.
MAX_TOKENS          = {"A":16_000,"B":8_000,"C":16_000,"D":8_000,"E":4_000,"F":8_000}

PLAN = {
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
def _load(path, default):
    try: return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError): return default

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

def remaining(m_key):
    for period_prefix in ["d", "w", "m"]:
        usage_key_full = f"{period_prefix}_u"
        if usage_key_full not in quota: quota[usage_key_full] = {}
        if m_key not in quota[usage_key_full]: quota[usage_key_full][m_key] = 0
    used_d = quota["d_u"].get(m_key,0)
    used_w = quota["w_u"].get(m_key,0)
    used_m = quota["m_u"].get(m_key,0)
    lim_d,lim_w,lim_m = PLAN[m_key]
    return lim_d-used_d, lim_w-used_w, lim_m-used_m

def record_use(m_key):
    for period_prefix in ["d", "w", "m"]:
        usage_key_full = f"{period_prefix}_u"
        if usage_key_full not in quota: quota[usage_key_full] = {}
        if m_key not in quota[usage_key_full]: quota[usage_key_full][m_key] = 0
        quota[usage_key_full][m_key]+=1
    _save(QUOTA_FILE,quota)

# ───────── Persistent sessions ─
sessions = _load(SESS_FILE,{})
def _new_sid():
    sid=str(int(time.time()*1000)); sessions[sid]={"title":"New chat","messages":[]}
    _save(SESS_FILE,sessions); return sid

# ───────── Logging ──────────
logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)

def api_post(payload, *, stream=False, timeout=DEFAULT_TIMEOUT):
    h={"Authorization":f"Bearer {OPENROUTER_API_KEY}","Content-Type":"application/json"}
    # Simplified logging for payload to avoid excessive length
    logged_payload_info = {k: (v if k != "messages" else f"<{len(v)} messages>") for k, v in payload.items()}
    logging.info(f"API POST to /chat/completions. Model: {payload.get('model')}, Stream: {stream}, Payload keys: {list(payload.keys())}")
    if "messages" in payload and payload["messages"]:
         logging.info(f"  First user message (truncated): {str(next((m.get('content') for m in payload['messages'] if m.get('role') == 'user'), 'N/A'))[:100]}")
    return requests.post(f"{OPENROUTER_API_BASE}/chat/completions",
                         headers=h,json=payload,stream=stream,timeout=timeout)

# ───────── OpenRouter helpers ─
# This function is for the MAIN LLM response and NEEDS max_tokens
def streamed(model_id, msgs, max_tokens_for_model_output):
    p = {
        "model": model_id,
        "messages": msgs,
        "stream": True,
        "max_tokens": max_tokens_for_model_output  # CRITICAL: Max tokens for the *output* of the main LLM
    }
    with api_post(p,stream=True) as r:
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            error_content = r.text
            logging.error(f"API Error during stream for model {model_id}: {e}. Response: {error_content}")
            yield None, f"API Error ({e.response.status_code}): {error_content}"
            return

        for ln in r.iter_lines():
            if not ln: continue
            if ln.startswith(b"data: "):
                data=ln[6:].decode('utf-8').strip()
                if data=="[DONE]": break
                try:
                    chunk=json.loads(data)
                except json.JSONDecodeError:
                    logging.error(f"Failed to decode JSON chunk: '{data}'")
                    yield None, "Error decoding stream data."
                    return
                if "error" in chunk:
                    err_msg = chunk["error"].get("message","Unknown error from API in stream")
                    logging.error(f"Stream chunk error for model {model_id}: {err_msg}")
                    yield None, err_msg; return
                delta_content = chunk.get("choices",[{}])[0].get("delta",{}).get("content")
                if delta_content is not None:
                    yield delta_content, None
            else:
                logging.warning(f"Unexpected non-event-stream line: {ln}")


# REVERTED route_choice to be like your original. No max_tokens in its own API call.
def route_choice(user_msg, allowed_model_keys): # allowed_model_keys are "A", "B", etc.
    if not allowed_model_keys:
        logging.warning("route_choice called with no allowed_model_keys. Returning None.")
        return None

    # Original simple prompt structure
    # The router model needs to see the full user_msg to make its choice.
    # We do NOT specify max_tokens for the router's output here; let it use its default.
    # The parsing logic `for ch in text:` will find the first valid character.
    router_prompt_text = f"Choose ONE of {allowed_model_keys}. No explanation."
    msgs_for_router = [
        {"role": "system", "content": router_prompt_text},
        {"role": "user", "content": user_msg} # The actual user's prompt
    ]
    
    payload_for_router = {
        "model": ROUTER_MODEL_ID, # ROUTER_MODEL_ID is MODEL_MAP["F"]
        "messages": msgs_for_router
        # NO "max_tokens" here for the router's own output.
    }
    try:
        logging.info(f"Calling router model {ROUTER_MODEL_ID} with user message (first 100 chars): '{user_msg[:100]}...' and allowed keys: {allowed_model_keys}")
        r = api_post(payload_for_router) # stream=False is default
        r.raise_for_status()
        response_json = r.json()
        
        raw_router_response_text = response_json.get("choices",[{}])[0].get("message",{}).get("content","").strip().upper()
        logging.info(f"Router model ({ROUTER_MODEL_ID}) raw response: '{raw_router_response_text}'")

        for char_code in raw_router_response_text: # Iterate through characters in response
            if char_code in allowed_model_keys:
                logging.info(f"Router successfully chose model key: {char_code}")
                return char_code
        
        logging.warning(f"Router model did not return a valid choice from {allowed_model_keys} in its response '{raw_router_response_text}'. Falling back to first allowed: {allowed_model_keys[0]}.")
        return allowed_model_keys[0] # Original fallback

    except requests.exceptions.RequestException as e:
        logging.error(f"Router API call to {ROUTER_MODEL_ID} failed: {e}. Response: {e.response.text if e.response else 'No response'}. Falling back to first allowed: {allowed_model_keys[0] if allowed_model_keys else 'None'}.")
        return allowed_model_keys[0] if allowed_model_keys else None
    except (KeyError, IndexError, AttributeError, TypeError) as e: # Added TypeError
        response_text_for_log = r.text if 'r' in locals() and hasattr(r, 'text') else 'N/A'
        logging.error(f"Error parsing router response for {ROUTER_MODEL_ID}: {e}. Raw Response: {response_text_for_log}. Falling back to {allowed_model_keys[0] if allowed_model_keys else 'None'}.")
        return allowed_model_keys[0] if allowed_model_keys else None

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

def update_credits_state():
    CRED_TOTAL, CRED_USED, CRED_LEFT = get_credits()
    st.session_state.CRED_TOTAL = CRED_TOTAL
    st.session_state.CRED_USED = CRED_USED
    st.session_state.CRED_LEFT = CRED_LEFT
    st.session_state.credits_ts = time.time()

# ───────── Streamlit page config ─
st.set_page_config(page_title="OpenRouter Chat", layout="wide",
                   initial_sidebar_state="expanded")

if "sid" not in st.session_state: st.session_state.sid=_new_sid()
if "credits_ts" not in st.session_state or \
   any(cred_key not in st.session_state for cred_key in ["CRED_TOTAL", "CRED_USED", "CRED_LEFT"]):
    update_credits_state()

# ───────── Sidebar UI ─────────────────────────────
with st.sidebar:
    st.image("https://avatars.githubusercontent.com/u/130328222?s=200&v=4", width=48)
    st.title("OpenRouter Chat")

    if st.button("➕  New chat", use_container_width=True):
        st.session_state.sid=_new_sid(); st.rerun()

    st.subheader("Chats")
    sorted_sids = sorted(sessions.keys(), key=lambda x: int(x), reverse=True)
    for sid_key in sorted_sids:
        obj = sessions.get(sid_key, {"title": "Error: Missing Session"})
        lbl=obj.get("title","Untitled")[:25] or "Untitled"
        if st.button(lbl, key=f"chatbtn_{sid_key}", use_container_width=True):
            st.session_state.sid=sid_key; st.rerun()

    st.markdown("---")
    st.subheader("Daily Token-Jars")
    model_keys_for_jars = sorted(MODEL_MAP.keys())
    jar_cols=st.columns(len(model_keys_for_jars))
    for idx,m_key in enumerate(model_keys_for_jars):
        rem_d, _, _ = remaining(m_key)
        plan_d, _, _ = PLAN[m_key]
        pct_float = 1.0 if plan_d > 900_000 else max(0.0, float(rem_d) / plan_d if plan_d > 0 else 0.0)
        fill = int(pct_float * 100)
        color = '#4caf50'
        if pct_float <= 0.25: color = '#f44336'
        elif pct_float <= 0.5: color = '#ff9800'
        jar_html=f"""<div style="width:44px;margin:auto;text-align:center;font-size:12px"><div style="border:2px solid #555;border-radius:6px;height:60px;position:relative;overflow:hidden;background:#f0f0f0;"><div style="position:absolute;bottom:0;width:100%;height:{fill}%;background:{color};"></div><div style="position:absolute;top:2px;left:0;right:0;font-size:18px; line-height:1;">{EMOJI.get(m_key, "❓")}</div><div style="position:absolute;bottom:2px;left:0;right:0; font-size:10px; font-weight:bold;">{m_key}</div></div><span>{'∞' if plan_d > 900_000 else rem_d}</span></div>"""
        jar_cols[idx].markdown(jar_html, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Model-routing map")
    st.caption(f"Router engine: **{ROUTER_MODEL_ID}**")
    with st.expander("Letters → Models"):
        for k_map,v_map in MODEL_MAP.items():
            st.markdown(f"**{k_map}** → `{v_map}` (Max Output Tokens: {MAX_TOKENS.get(k_map, 'N/A')})")

    with st.expander("Account stats (credits)"):
        if st.button("Refresh Credits", key="refresh_credits_btn"):
            update_credits_state()
        if st.session_state.CRED_TOTAL is None: st.warning("Couldn’t fetch /credits endpoint.")
        else:
            st.markdown(f"**Purchased:** {st.session_state.CRED_TOTAL:.2f} cr")
            st.markdown(f"**Used:** {st.session_state.CRED_USED:.2f} cr")
            st.markdown(f"**Remaining:** {st.session_state.CRED_LEFT:.2f} cr")
            st.caption(f"Last updated: {datetime.fromtimestamp(st.session_state.credits_ts).strftime('%Y-%m-%d %H:%M:%S')}")

# ───────── Chat panel ─────────────────────────────
if st.session_state.sid not in sessions:
    logging.warning(f"Session ID {st.session_state.sid} not found. Creating new.")
    st.session_state.sid = _new_sid()

chat_session = sessions[st.session_state.sid]
chat_messages = chat_session.setdefault("messages", [])

for msg in chat_messages:
    avatar="👤" if msg["role"]=="user" else EMOJI.get(msg.get("model_key", "F"), "🤖")
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg.get("content", "*empty message*"))

if prompt := st.chat_input("Ask anything…"):
    chat_messages.append({"role":"user","content":prompt})
    with st.chat_message("user", avatar="👤"): st.markdown(prompt)

    allowed_model_keys = [m_key for m_key in MODEL_MAP if remaining(m_key)[0] > 0]

    if not allowed_model_keys:
        st.error("❗ No models available due to daily quota limits. Please try again later.")
        logging.warning("No models available due to quota limits for current prompt.")
    else:
        chosen_route_key = route_choice(prompt, allowed_model_keys) # This is "A", "B", etc.

        if chosen_route_key is None or chosen_route_key not in MODEL_MAP:
            st.error(f"❗ Model routing failed or no models available. Please check logs or try again.")
            logging.error(f"Routing failed critically. chosen_route_key: {chosen_route_key}. Allowed: {allowed_model_keys}")
        else:
            actual_model_id = MODEL_MAP[chosen_route_key]
            # THIS IS THE CRITICAL MAX_TOKENS for the chosen model's *output*
            max_output_tokens_for_selected_model = MAX_TOKENS[chosen_route_key]

            with st.chat_message("assistant", avatar=EMOJI.get(chosen_route_key, "🤖")):
                response_box = st.empty()
                full_response_content = ""
                api_messages = list(chat_messages) # Send a copy
                stream_successful = True

                # Call `streamed` with the `max_output_tokens_for_selected_model`
                for chunk, error_msg in streamed(actual_model_id, api_messages, max_output_tokens_for_selected_model):
                    if error_msg:
                        full_response_content = f"❗ **API Error for {actual_model_id}**:\n\n{error_msg}"
                        response_box.error(full_response_content)
                        logging.error(f"Streaming error for model {actual_model_id}: {error_msg}")
                        stream_successful = False
                        break 
                    if chunk is not None:
                        full_response_content += chunk
                        response_box.markdown(full_response_content + "▌")
                
                response_box.markdown(full_response_content)

            chat_messages.append({
                "role": "assistant",
                "content": full_response_content,
                "model_key": chosen_route_key
            })
            
            if stream_successful or not (full_response_content.startswith("❗ **API Error") and "credit" in full_response_content.lower()):
                 record_use(chosen_route_key)

            if chat_session.get("title") == "New chat" and len(chat_messages) > 1:
                try:
                    first_user_message = next((m['content'] for m in chat_messages if m['role'] == 'user'), "Chat")
                    title_candidate = first_user_message[:30]
                    chat_session["title"] = title_candidate + ("..." if len(first_user_message) > 30 else "")
                except Exception as e:
                    logging.error(f"Error generating title: {e}")
                    chat_session["title"] = "Chat Interaction"
            _save(SESS_FILE, sessions)

# ───────── Self-relaunch when run via `python main.py` ─
if __name__=="__main__" and os.getenv("_IS_STRL")!="1":
    os.environ["_IS_STRL"]="1"
    port=os.getenv("PORT","8501")
    script_path = os.path.abspath(__file__)
    logging.info(f"Starting Streamlit server on port {port} for script: {script_path}")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", script_path,
                       "--server.port",port,"--server.address","0.0.0.0"], check=True)
    except subprocess.CalledProcessError as e: logging.error(f"Failed to start Streamlit: {e}")
    except FileNotFoundError: logging.error("Streamlit command not found. Is Streamlit installed and in PATH?")
