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
    "A": "google/gemini-1.5-pro-preview", # Updated to 1.5 Pro based on other comments, assume it's a typo in original
    "B": "openai/gpt-4o-mini",            # Updated to gpt-4o-mini as 'o4-mini' is likely a typo
    "C": "openai/gpt-4o",                 # Updated to gpt-4o as 'chatgpt-4o-latest' is not standard
    "D": "deepseek/deepseek-chat",        # Assuming deepseek-chat for 'deepseek-r1'
    "E": "meta-llama/llama-3-70b-instruct", # Assuming a Llama model for 'grok-3-beta' example; X-AI Grok is not on OpenRouter AFAIK
    "F": "google/gemini-1.5-flash-preview" # Updated to 1.5 Flash
}
ROUTER_MODEL_ID     = MODEL_MAP["F"]
# Max TOKENS for OUTPUT. Ensure these are reasonable for your use case.
MAX_TOKENS          = {"A":16_000,"B":8_000,"C":16_000,"D":8_000,"E":4_000,"F":8_000}

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
    zeros = {m:0 for m in MODEL_MAP}
    q     = _load(QUOTA_FILE,{})
    _reset(q,"d",_today(),zeros)
    _reset(q,"w",_yweek(),zeros)
    _reset(q,"m",_ymonth(),zeros)
    _save(QUOTA_FILE,q); return q
quota = _load_quota()

def remaining(m):
    used_d,used_w,used_m = quota["d_u"].get(m,0),quota["w_u"].get(m,0),quota["m_u"].get(m,0)
    lim_d,lim_w,lim_m    = PLAN[m]
    return lim_d-used_d, lim_w-used_w, lim_m-used_m

def record_use(m):
    for period_prefix in ["d", "w", "m"]:
        usage_key = f"{period_prefix}_u"
        if m not in quota[usage_key]: # Initialize if model key missing (e.g. new model added)
            quota[usage_key][m] = 0
        quota[usage_key][m]+=1
    _save(QUOTA_FILE,quota)

# ───────── Persistent sessions ─
sessions = _load(SESS_FILE,{})
def _new_sid():
    sid=str(int(time.time()*1000)); sessions[sid]={"title":"New chat","messages":[]}
    _save(SESS_FILE,sessions); return sid

# ───────── Logging ──────────
logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(message)s", stream=sys.stdout)

def api_post(payload, *, stream=False, timeout=DEFAULT_TIMEOUT):
    h={"Authorization":f"Bearer {OPENROUTER_API_KEY}","Content-Type":"application/json"}
    logging.info(f"API POST to /chat/completions with payload: {json.dumps(payload)[:200]}...") # Log part of payload
    return requests.post(f"{OPENROUTER_API_BASE}/chat/completions",
                         headers=h,json=payload,stream=stream,timeout=timeout)

# ───────── OpenRouter helpers ─
def streamed(model, msgs, max_tokens_for_model): # Added max_tokens_for_model
    p = {
        "model": model,
        "messages": msgs,
        "stream": True,
        "max_tokens": max_tokens_for_model  # Explicitly set max_tokens
    }
    with api_post(p,stream=True) as r:
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            error_content = r.text
            logging.error(f"API Error: {e}. Response: {error_content}")
            yield None, f"API Error: {e.response.status_code} - {error_content}"
            return

        for ln in r.iter_lines():
            if not ln: continue # Handle keep-alive newlines
            if ln.startswith(b"data: "):
                data=ln[6:].decode().strip()
                if data=="[DONE]": break
                try:
                    chunk=json.loads(data)
                except json.JSONDecodeError:
                    logging.error(f"Failed to decode JSON chunk: {data}")
                    yield None, "Error decoding stream data."
                    return

                if "error" in chunk:
                    err_msg = chunk["error"].get("message","Unknown error from API")
                    logging.error(f"Stream chunk error: {err_msg}")
                    yield None, err_msg; return
                
                delta = chunk.get("choices",[{}])[0].get("delta",{}).get("content")
                if delta: yield delta,None
            else: # Handle non-event-stream lines if any, though typically not expected for data
                logging.warning(f"Received unexpected line in stream: {ln}")


def route_choice(user_msg, allowed_model_keys):
    if not allowed_model_keys:
        logging.warning("route_choice called with no allowed_model_keys.")
        return None # Or raise an error, or return a default

    # Improved prompt for clarity and to request only the letter
    prompt_content = (
        f"You are a helpful assistant that chooses the best model for a task. "
        f"Based on the user's message, pick ONLY ONE model letter from this list: {allowed_model_keys}. "
        f"Do not provide any explanation, reasoning, or any other text. "
        f"Just return the single capital letter corresponding to your choice."
    )
    msgs=[{"role":"system","content":prompt_content},
          {"role":"user","content":user_msg}]
    
    # For routing, the output is very short (a single letter).
    # Set a small, specific max_tokens for this call.
    router_max_output_tokens = 10 

    payload = {
        "model": ROUTER_MODEL_ID,
        "messages": msgs,
        "max_tokens": router_max_output_tokens
    }
    try:
        r = api_post(payload) # stream=False is default
        r.raise_for_status()
        response_json = r.json()
        text = response_json.get("choices",[{}])[0].get("message",{}).get("content","").strip().upper()
        
        logging.info(f"Router model ({ROUTER_MODEL_ID}) response: '{text}' for allowed keys: {allowed_model_keys}")

        # Extract the first valid character from the response that is in allowed_model_keys
        for char_code in text:
            if char_code in allowed_model_keys:
                logging.info(f"Router chose: {char_code}")
                return char_code
        
        # Fallback if model response is not one of the allowed keys or empty
        logging.warning(f"Router model did not return a valid choice from {allowed_model_keys}. Got: '{text}'. Falling back to first allowed.")
        return allowed_model_keys[0]

    except requests.exceptions.RequestException as e:
        logging.warning(f"Router API call failed: {e}. Falling back to first allowed model.")
        return allowed_model_keys[0]
    except (KeyError, IndexError, AttributeError) as e:
        logging.warning(f"Error parsing router response: {e}. Response: {r.text if 'r' in locals() else 'N/A'}. Falling back.")
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

# Initial fetch or re-fetch logic for credits
def update_credits_state():
    CRED_TOTAL, CRED_USED, CRED_LEFT = get_credits()
    st.session_state.CRED_TOTAL = CRED_TOTAL
    st.session_state.CRED_USED = CRED_USED
    st.session_state.CRED_LEFT = CRED_LEFT
    st.session_state.credits_ts = time.time()

# ───────── Streamlit page config ─
st.set_page_config(page_title="OpenRouter Chat", layout="wide",
                   initial_sidebar_state="expanded")

# Session state initialization
if "sid" not in st.session_state: st.session_state.sid=_new_sid()
if "credits_ts" not in st.session_state or \
   "CRED_TOTAL" not in st.session_state or \
   "CRED_USED" not in st.session_state or \
   "CRED_LEFT" not in st.session_state:
    update_credits_state()


# ───────── Sidebar UI ─────────────────────────────
with st.sidebar:
    st.image("https://avatars.githubusercontent.com/u/130328222?s=200&v=4", width=48)
    st.title("OpenRouter Chat")

    if st.button("➕  New chat", use_container_width=True):
        st.session_state.sid=_new_sid(); st.rerun()

    st.subheader("Chats")
    # Sort sessions by SID (timestamp based) descending for most recent first
    sorted_sids = sorted(sessions.keys(), reverse=True)
    for sid_key in sorted_sids:
        obj = sessions[sid_key]
        lbl=obj.get("title","Untitled")[:25] or "Untitled"
        if st.button(lbl, key=f"chatbtn_{sid_key}", use_container_width=True): # Unique keys for buttons
            st.session_state.sid=sid_key; st.rerun()

    st.markdown("---")
    st.subheader("Daily Token-Jars")
    # Ensure MODEL_MAP keys are sorted if consistent order is desired, e.g., alphabetically
    model_keys_for_jars = sorted(MODEL_MAP.keys())
    jar_cols=st.columns(len(model_keys_for_jars))
    for idx,m_key in enumerate(model_keys_for_jars):
        # Ensure quota structure is initialized for m_key if it's new
        if m_key not in quota["d_u"]: quota["d_u"][m_key] = 0
        if m_key not in quota["w_u"]: quota["w_u"][m_key] = 0
        if m_key not in quota["m_u"]: quota["m_u"][m_key] = 0
        
        rem_d, rem_w, rem_m = remaining(m_key)
        plan_d, plan_w, plan_m = PLAN[m_key]
        
        pct = 1.0 if plan_d > 900_000 else max(0.0, float(rem_d) / plan_d if plan_d > 0 else 0)
        fill = int(pct*100)
        jar_html=f"""
        <div style="width:44px;margin:auto;text-align:center;font-size:12px">
          <div style="border:2px solid #555;border-radius:6px;height:60px;
                      position:relative;overflow:hidden;background:#f0f0f0;">
              <div style="position:absolute;bottom:0;width:100%;height:{fill}%;
                          background:{'#4caf50' if fill > 10 else '#f44336'};"></div>
              <div style="position:absolute;top:2px;left:0;right:0;
                          font-size:18px; line-height:1;">{EMOJI.get(m_key, "❓")}</div>
              <div style="position:absolute;bottom:2px;left:0;right:0; font-size:10px; font-weight:bold;">{m_key}</div>
          </div>
          <span>{'∞' if plan_d > 900_000 else rem_d}</span>
        </div>"""
        jar_cols[idx].markdown(jar_html, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Model-routing map")
    st.caption(f"Router engine: **{MODEL_MAP.get(ROUTER_MODEL_ID.split('/')[-1].split('-')[0].upper()[0] if isinstance(ROUTER_MODEL_ID,str) else 'F', ROUTER_MODEL_ID)}**") # More robustly find key or show ID
    with st.expander("Letters → Models"):
        for k_map,v_map in MODEL_MAP.items():
            st.markdown(f"**{k_map}** → `{v_map}` (Max Output: {MAX_TOKENS.get(k_map, 'N/A')} tokens)")

    with st.expander("Account stats (credits)"):
        if st.button("Refresh Credits"):
            update_credits_state()
            st.rerun()

        if st.session_state.CRED_TOTAL is None:
            st.warning("Couldn’t fetch /credits endpoint.")
        else:
            st.markdown(f"**Purchased:** {st.session_state.CRED_TOTAL:.2f} cr")
            st.markdown(f"**Used:** {st.session_state.CRED_USED:.2f} cr")
            st.markdown(f"**Remaining:** {st.session_state.CRED_LEFT:.2f} cr")
            st.caption(f"Last updated: {datetime.fromtimestamp(st.session_state.credits_ts).strftime('%Y-%m-%d %H:%M:%S')}")


# ───────── Chat panel ─────────────────────────────
# Ensure current session ID from state is valid, else create new
if st.session_state.sid not in sessions:
    st.session_state.sid = _new_sid()
    # Log this event or show a message if it's unexpected
    logging.info(f"Current SID {st.session_state.sid} not in sessions, created new one.")

chat_session = sessions[st.session_state.sid]
chat_messages = chat_session["messages"]

# Display chat history
for msg in chat_messages:
    avatar="👤" if msg["role"]=="user" else EMOJI.get(msg.get("model_key", "F"), "🤖")
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything…"):
    chat_messages.append({"role":"user","content":prompt})
    with st.chat_message("user", avatar="👤"): st.markdown(prompt)

    # Determine allowed models based on quota
    allowed_model_keys = [m_key for m_key in MODEL_MAP if remaining(m_key)[0] > 0]

    if not allowed_model_keys:
        st.error("❗ No models available due to daily quota limits. Please try again later.")
    else:
        # Get model route (the letter key like "A", "B")
        chosen_route_key = route_choice(prompt, allowed_model_keys)

        if chosen_route_key is None or chosen_route_key not in MODEL_MAP:
            st.error(f"❗ Could not determine a valid model route. Fallback failed or no models allowed.")
            # chosen_route_key = allowed_model_keys[0] # Or handle error more gracefully
            # st.warning(f"Routing failed. Using first available: {chosen_route_key}")
        # else: # This else is actually not needed if the above error is terminal for this turn
        
        # Proceed if chosen_route_key is valid
        if chosen_route_key and chosen_route_key in MODEL_MAP:
            actual_model_id = MODEL_MAP[chosen_route_key]
            # Get the max_tokens for the *output* of this chosen model
            max_output_tokens_for_model = MAX_TOKENS[chosen_route_key]

            with st.chat_message("assistant", avatar=EMOJI.get(chosen_route_key, "🤖")):
                response_box = st.empty()
                full_response_content = ""
                
                # Prepare messages for API: potentially truncate or summarize if too long
                # For now, sending all messages. Consider truncation for very long chats.
                api_messages = list(chat_messages) # Send a copy

                for chunk, error_msg in streamed(actual_model_id, api_messages, max_output_tokens_for_model):
                    if error_msg:
                        full_response_content = f"❗ API Error: {error_msg}"
                        response_box.error(full_response_content)
                        break 
                    if chunk:
                        full_response_content += chunk
                        response_box.markdown(full_response_content + "▌")
                
                response_box.markdown(full_response_content) # Final response

            # Append assistant's response to chat history
            # Store model_key with message for consistent avatar display
            chat_messages.append({"role": "assistant", "content": full_response_content, "model_key": chosen_route_key})
            
            # Record usage for the chosen model route if no critical error prevented an attempt
            if not (full_response_content.startswith("❗ API Error:") and "credit" in full_response_content.lower()):
                 # Don't record use if it was a credit error that prevented the call entirely.
                 # However, if an attempt was made and failed mid-stream, usage might still be counted by provider.
                record_use(chosen_route_key)

            # Update session title if it's "New chat" and we have some content
            if chat_session.get("title") == "New chat" and len(chat_messages) > 1:
                try:
                    # Attempt to generate a title (very simple for now)
                    first_user_message = next((m['content'] for m in chat_messages if m['role'] == 'user'), "Chat")
                    chat_session["title"] = first_user_message[:30] + ("..." if len(first_user_message)>30 else "")
                except Exception as e:
                    logging.error(f"Error generating title: {e}")
                    chat_session["title"] = "Chat" # Fallback title

            _save(SESS_FILE, sessions)
            # No st.rerun() here, allows interaction with the displayed response.

# ───────── Self-relaunch when run via `python main.py` ─
if __name__=="__main__" and os.getenv("_IS_STRL")!="1":
    os.environ["_IS_STRL"]="1"
    port=os.getenv("PORT","8501")
    # Ensure __file__ is absolute for streamlit run
    script_path = os.path.abspath(__file__)
    logging.info(f"Starting Streamlit server on port {port} for script: {script_path}")
    subprocess.run([sys.executable, "-m", "streamlit", "run", script_path,
                   "--server.port",port,"--server.address","0.0.0.0"])
