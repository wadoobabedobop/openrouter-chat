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
OPENROUTER_API_KEY  = "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" # Replace with your actual key if this is a placeholder
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT     = 120

# RESTORED ORIGINAL MODEL_MAP
MODEL_MAP = {
    "A": "google/gemini-2.5-pro-preview",
    "B": "openai/o4-mini",
    "C": "openai/chatgpt-4o-latest",
    "D": "deepseek/deepseek-r1",
    "E": "x-ai/grok-3-beta",
    "F": "google/gemini-2.5-flash-preview"      # also used for router
}
ROUTER_MODEL_ID     = MODEL_MAP["F"]
# Max TOKENS for OUTPUT. Ensure these are reasonable for your use case.
MAX_TOKENS          = {"A":16_000,"B":8_000,"C":16_000,"D":8_000,"E":4_000,"F":8_000} # Kept from previous, aligns with your original keys

PLAN = {          # daily, weekly, monthly “limits”
    "A": (6,45,180), "B": (2,15,60), "C": (1,8,30),
    "D": (3,25,100),"E": (1,10,40), "F": (999_999,50,190) # "F" seems to be a generous/router model
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
    zeros = {m:0 for m in MODEL_MAP} # Uses keys from MODEL_MAP
    q     = _load(QUOTA_FILE,{})
    _reset(q,"d",_today(),zeros)
    _reset(q,"w",_yweek(),zeros)
    _reset(q,"m",_ymonth(),zeros)
    _save(QUOTA_FILE,q); return q
quota = _load_quota()

def remaining(m_key): # m_key is "A", "B", etc.
    # Initialize usage counts if model key is new or missing
    for period_prefix in ["d", "w", "m"]:
        usage_key_full = f"{period_prefix}_u"
        if usage_key_full not in quota: quota[usage_key_full] = {} # Ensure the usage dict exists
        if m_key not in quota[usage_key_full]: quota[usage_key_full][m_key] = 0

    used_d = quota["d_u"].get(m_key,0)
    used_w = quota["w_u"].get(m_key,0)
    used_m = quota["m_u"].get(m_key,0)
    
    lim_d,lim_w,lim_m = PLAN[m_key]
    return lim_d-used_d, lim_w-used_w, lim_m-used_m

def record_use(m_key): # m_key is "A", "B", etc.
    for period_prefix in ["d", "w", "m"]:
        usage_key_full = f"{period_prefix}_u"
        if usage_key_full not in quota: quota[usage_key_full] = {} # Ensure the usage dict exists
        if m_key not in quota[usage_key_full]: quota[usage_key_full][m_key] = 0 # Initialize if model key missing
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
    # Log only a part of messages to avoid excessive logging, especially if it contains PII
    logged_payload = payload.copy()
    if "messages" in logged_payload and len(logged_payload["messages"]) > 0:
        # Log first and last message content for context, or just number of messages
        if len(logged_payload["messages"]) > 1:
            logged_payload["messages"] = f"[{len(logged_payload['messages'])} messages, first: '{str(logged_payload['messages'][0]['content'])[:50]}...', last: '{str(logged_payload['messages'][-1]['content'])[:50]}...']"
        else:
            logged_payload["messages"] = f"[{len(logged_payload['messages'])} messages, content: '{str(logged_payload['messages'][0]['content'])[:100]}...']"

    logging.info(f"API POST to /chat/completions with payload: {json.dumps(logged_payload)}")
    return requests.post(f"{OPENROUTER_API_BASE}/chat/completions",
                         headers=h,json=payload,stream=stream,timeout=timeout)

# ───────── OpenRouter helpers ─
def streamed(model_id, msgs, max_tokens_for_model): # Added max_tokens_for_model
    p = {
        "model": model_id,
        "messages": msgs,
        "stream": True,
        "max_tokens": max_tokens_for_model  # Explicitly set max_tokens
    }
    # logging.info(f"Streaming request payload: {p}") # Can be verbose
    with api_post(p,stream=True) as r:
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            error_content = r.text
            logging.error(f"API Error during stream: {e}. Response status: {e.response.status_code}. Response body: {error_content}")
            yield None, f"API Error ({e.response.status_code}): {error_content}" # Provide more detailed error
            return

        for ln in r.iter_lines():
            if not ln: continue
            if ln.startswith(b"data: "):
                data=ln[6:].decode('utf-8').strip() # Specify utf-8 decoding
                if data=="[DONE]": break
                try:
                    chunk=json.loads(data)
                except json.JSONDecodeError:
                    logging.error(f"Failed to decode JSON chunk: '{data}'")
                    yield None, "Error decoding stream data."
                    return

                if "error" in chunk:
                    err_msg = chunk["error"].get("message","Unknown error from API in stream")
                    logging.error(f"Stream chunk error: {err_msg} (Full chunk: {chunk})")
                    yield None, err_msg; return
                
                delta_obj = chunk.get("choices",[{}])[0].get("delta",{})
                delta_content = delta_obj.get("content")

                if delta_content: # Ensure delta_content is not None
                    yield delta_content, None
            else:
                logging.warning(f"Received unexpected non-event-stream line: {ln}")


def route_choice(user_msg, allowed_model_keys): # allowed_model_keys are "A", "B", etc.
    if not allowed_model_keys:
        logging.warning("route_choice called with no allowed_model_keys.")
        return None

    prompt_content = (
        f"You are a model routing assistant. Your task is to choose the most suitable Large Language Model for the user's query. "
        f"Based on the user's message below, select ONLY ONE model letter from the following list of available models: {allowed_model_keys}. "
        f"Do NOT provide any explanation, reasoning, preamble, or any text other than the single capital letter corresponding to your choice. "
        f"User message: \"{user_msg}\"\n\n"
        f"Chosen model letter (from {allowed_model_keys}):"
    )
    msgs=[{"role":"system","content": "You are a helpful assistant that responds with only a single capital letter."}, # Simplified system prompt
          {"role":"user","content":prompt_content}]
    
    router_max_output_tokens = 5 # A single letter, maybe a few extra chars if it misbehaves.

    payload = {
        "model": ROUTER_MODEL_ID, # This is already the full model ID string
        "messages": msgs,
        "max_tokens": router_max_output_tokens,
        "temperature": 0.1 # Lower temperature for more deterministic routing
    }
    try:
        r = api_post(payload)
        r.raise_for_status()
        response_json = r.json()
        text = response_json.get("choices",[{}])[0].get("message",{}).get("content","").strip().upper()
        
        logging.info(f"Router model ({ROUTER_MODEL_ID}) raw response: '{text}' for prompt regarding allowed keys: {allowed_model_keys}")

        for char_code in text: # Iterate through characters in response
            if char_code in allowed_model_keys:
                logging.info(f"Router chose model key: {char_code}")
                return char_code
        
        logging.warning(f"Router model did not return a valid choice from {allowed_model_keys}. Got: '{text}'. Falling back to first allowed: {allowed_model_keys[0]}.")
        return allowed_model_keys[0]

    except requests.exceptions.RequestException as e:
        logging.error(f"Router API call failed: {e}. Response: {e.response.text if e.response else 'No response'}. Falling back to first allowed model: {allowed_model_keys[0]}.")
        return allowed_model_keys[0]
    except (KeyError, IndexError, AttributeError) as e:
        logging.error(f"Error parsing router response: {e}. Raw Response: {r.text if 'r' in locals() else 'N/A'}. Falling back to {allowed_model_keys[0]}.")
        return allowed_model_keys[0]

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
    sorted_sids = sorted(sessions.keys(), key=lambda x: int(x), reverse=True) # Sort by SID as int
    for sid_key in sorted_sids:
        obj = sessions.get(sid_key, {"title": "Error: Missing Session"}) # Defensive get
        lbl=obj.get("title","Untitled")[:25] or "Untitled"
        if st.button(lbl, key=f"chatbtn_{sid_key}", use_container_width=True):
            st.session_state.sid=sid_key; st.rerun()

    st.markdown("---")
    st.subheader("Daily Token-Jars")
    model_keys_for_jars = sorted(MODEL_MAP.keys())
    jar_cols=st.columns(len(model_keys_for_jars))
    for idx,m_key in enumerate(model_keys_for_jars):
        rem_d, _, _ = remaining(m_key) # We only need daily remaining for the jar display
        plan_d, _, _ = PLAN[m_key]
        
        pct_float = 1.0 if plan_d > 900_000 else max(0.0, float(rem_d) / plan_d if plan_d > 0 else 0.0)
        fill = int(pct_float * 100)
        color = '#4caf50' # Green
        if pct_float <= 0.25: color = '#f44336' # Red if 25% or less
        elif pct_float <= 0.5: color = '#ff9800' # Orange if 50% or less
        
        jar_html=f"""
        <div style="width:44px;margin:auto;text-align:center;font-size:12px">
          <div style="border:2px solid #555;border-radius:6px;height:60px;
                      position:relative;overflow:hidden;background:#f0f0f0;">
              <div style="position:absolute;bottom:0;width:100%;height:{fill}%;
                          background:{color};"></div>
              <div style="position:absolute;top:2px;left:0;right:0;
                          font-size:18px; line-height:1;">{EMOJI.get(m_key, "❓")}</div>
              <div style="position:absolute;bottom:2px;left:0;right:0; font-size:10px; font-weight:bold;">{m_key}</div>
          </div>
          <span>{'∞' if plan_d > 900_000 else rem_d}</span>
        </div>"""
        jar_cols[idx].markdown(jar_html, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Model-routing map")
    st.caption(f"Router engine: **{ROUTER_MODEL_ID}**") # RESTORED Original caption
    with st.expander("Letters → Models"):
        for k_map,v_map in MODEL_MAP.items():
            st.markdown(f"**{k_map}** → `{v_map}` (Max Output: {MAX_TOKENS.get(k_map, 'N/A')} tokens)")

    with st.expander("Account stats (credits)"):
        if st.button("Refresh Credits", key="refresh_credits_btn"):
            update_credits_state()
            # st.rerun() # Rerun can be disruptive if user is typing. Consider if needed.
        
        if st.session_state.CRED_TOTAL is None:
            st.warning("Couldn’t fetch /credits endpoint.")
        else:
            st.markdown(f"**Purchased:** {st.session_state.CRED_TOTAL:.2f} cr")
            st.markdown(f"**Used:** {st.session_state.CRED_USED:.2f} cr")
            st.markdown(f"**Remaining:** {st.session_state.CRED_LEFT:.2f} cr")
            st.caption(f"Last updated: {datetime.fromtimestamp(st.session_state.credits_ts).strftime('%Y-%m-%d %H:%M:%S')}")


# ───────── Chat panel ─────────────────────────────
if st.session_state.sid not in sessions:
    logging.warning(f"Session ID {st.session_state.sid} not found in sessions. Creating new one.")
    st.session_state.sid = _new_sid()

chat_session = sessions[st.session_state.sid]
chat_messages = chat_session.setdefault("messages", []) # Ensure messages list exists

for msg in chat_messages:
    avatar="👤" if msg["role"]=="user" else EMOJI.get(msg.get("model_key", "F"), "🤖") # Use "F" as default model_key for avatar if missing
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg.get("content", "*empty message*")) # Handle if content is missing

if prompt := st.chat_input("Ask anything…"):
    chat_messages.append({"role":"user","content":prompt})
    with st.chat_message("user", avatar="👤"): st.markdown(prompt)

    allowed_model_keys = [m_key for m_key in MODEL_MAP if remaining(m_key)[0] > 0]

    if not allowed_model_keys:
        st.error("❗ No models available due to daily quota limits. Please try again later.")
        logging.warning("No models available due to quota limits.")
    else:
        chosen_route_key = route_choice(prompt, allowed_model_keys) # This is "A", "B", etc.

        if chosen_route_key is None or chosen_route_key not in MODEL_MAP:
            st.error(f"❗ Model routing failed. Could not determine a valid model route. Please try again or select a model manually if functionality exists.")
            logging.error(f"Routing failed. chosen_route_key: {chosen_route_key}. Allowed: {allowed_model_keys}")
        else:
            actual_model_id = MODEL_MAP[chosen_route_key] # Full model ID string
            max_output_tokens_for_model = MAX_TOKENS[chosen_route_key]

            with st.chat_message("assistant", avatar=EMOJI.get(chosen_route_key, "🤖")):
                response_box = st.empty()
                full_response_content = ""
                
                # Prepare messages for API. Consider history length limits for specific models.
                # For now, sending a copy of all current chat messages.
                api_messages = list(chat_messages) 

                stream_successful = True
                for chunk, error_msg in streamed(actual_model_id, api_messages, max_output_tokens_for_model):
                    if error_msg:
                        full_response_content = f"❗ **API Error for {actual_model_id}**:\n\n{error_msg}"
                        response_box.error(full_response_content)
                        logging.error(f"Streaming error for model {actual_model_id}: {error_msg}")
                        stream_successful = False
                        break 
                    if chunk is not None: # Ensure chunk is not None before concatenation
                        full_response_content += chunk
                        response_box.markdown(full_response_content + "▌")
                
                response_box.markdown(full_response_content)

            chat_messages.append({
                "role": "assistant",
                "content": full_response_content,
                "model_key": chosen_route_key # Store which model key was used
            })
            
            # Only record use if the stream didn't fail immediately due to a blocking error
            # (e.g., auth, credits). If it streams some and then errors, usage might still count.
            if stream_successful or not full_response_content.startswith("❗ **API Error"):
                 record_use(chosen_route_key)

            if chat_session.get("title") == "New chat" and len(chat_messages) > 1: # At least one user and one assistant message
                try:
                    first_user_message = next((m['content'] for m in chat_messages if m['role'] == 'user'), "Chat")
                    title_candidate = first_user_message[:30]
                    chat_session["title"] = title_candidate + ("..." if len(first_user_message) > 30 else "")
                except Exception as e:
                    logging.error(f"Error generating title: {e}")
                    chat_session["title"] = "Chat Interaction"

            _save(SESS_FILE, sessions)
            # No st.rerun() here, to keep the chat flow natural. It will rerun on next input.

# ───────── Self-relaunch when run via `python main.py` ─
if __name__=="__main__" and os.getenv("_IS_STRL")!="1":
    os.environ["_IS_STRL"]="1"
    port=os.getenv("PORT","8501")
    script_path = os.path.abspath(__file__)
    logging.info(f"Starting Streamlit server on port {port} for script: {script_path}")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", script_path,
                       "--server.port",port,"--server.address","0.0.0.0"], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to start Streamlit: {e}")
    except FileNotFoundError:
        logging.error("Streamlit command not found. Is Streamlit installed and in PATH?")
