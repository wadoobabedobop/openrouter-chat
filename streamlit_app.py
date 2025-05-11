#!/usr/bin/env python3
"""
OpenRouter Streamlit Chat — Replit edition
• Persistent chat sessions
• Daily/weekly/monthly quotas (“6-2-1 / 3-1 / Unlimited”)
• Pretty ‘token-jar’ gauges
• Model-routing panel restored
• Live credit/usage stats (GET /credits)
• LLM-generated chat titles
• Immediate token-jar updates
"""

# ───────── Imports ─────────
import json, logging, os, sys, subprocess, time, requests, math
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo
import streamlit as st

# ───────── Basic config ─────
OPENROUTER_API_KEY  = "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" # Replace if placeholder
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT     = 120

MODEL_MAP = {
    "A": "google/gemini-2.5-pro-preview",
    "B": "openai/o4-mini",
    "C": "openai/chatgpt-4o-latest",
    "D": "deepseek/deepseek-r1",
    "E": "x-ai/grok-3-beta",
    "F": "google/gemini-2.5-flash-preview"
}
ROUTER_MODEL_ID     = "mistralai/mistral-7b-instruct:free"
TITLE_GENERATION_MODEL_ID = "mistralai/mistral-7b-instruct:free" # Can be same as router or another cheap model

MAX_TOKENS          = {"A":16_000,"B":8_000,"C":16_000,"D":8_000,"E":4_000,"F":8_000}

PLAN = {
    "A": (6,45,180), "B": (2,15,60), "C": (1,8,30),
    "D": (3,25,100),"E": (1,10,40), "F": (999_999,50,190)
}
EMOJI = {"A":"🌟","B":"🔷","C":"🟥","D":"🟢","E":"🟡","F":"🌀"}
TZ    = ZoneInfo("Australia/Sydney")

MODEL_DESCRIPTIONS = {
    "A": "🌟 (google/gemini-2.5-pro-preview) – anything truly mission-critical or creative where you crave top quality. Expensive.",
    "B": "🔷 (openai/o4-mini) – mid-stakes reasoning, good for slow and reasonably cheap responses. Cheap.",
    "C": "🟥 (openai/chatgpt-4o-latest) – for queries needing nice formatting or empathetic responses. Expensive.",
    "D": "🟢 (deepseek/deepseek-r1) – brainstorming, writing - good when you need cheap reasoning. Cheap.",
    "E": "🟡 (x-ai/grok-3-beta) – when you want an edgier style or a second opinion on reasoning. Expensive.",
    "F": "🌀 (google/gemini-2.5-flash-preview) – clarifications, follow-ups, and most general use; effectively free."
}

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

# Load quota once at the start. It will be modified in-memory.
# st.rerun() will cause the script to re-execute, but this global `quota` object persists in the Python process's memory
# for the lifetime of the Streamlit app server process for that user session, unless explicitly reloaded from file.
# `record_use` updates this in-memory `quota` object.
# `_load_quota` is primarily for initializing from file on first run or if the app restarts.
_initial_quota_load_done = False
quota = {}

def load_quota_from_file():
    global quota, _initial_quota_load_done
    zeros = {m:0 for m in MODEL_MAP}
    q_from_file = _load(QUOTA_FILE,{})
    _reset(q_from_file,"d",_today(),zeros)
    _reset(q_from_file,"w",_yweek(),zeros)
    _reset(q_from_file,"m",_ymonth(),zeros)
    _save(QUOTA_FILE, q_from_file) # Save potential resets
    quota = q_from_file # Update the global quota object
    _initial_quota_load_done = True
    logging.info("Quota loaded/re-initialized from file.")

if not _initial_quota_load_done:
    load_quota_from_file()


def remaining(m_key):
    # Ensure quota structure exists for the key
    for period_prefix in ["d", "w", "m"]:
        usage_key_full = f"{period_prefix}_u"
        if usage_key_full not in quota: quota[usage_key_full] = {}
        if m_key not in quota[usage_key_full]: quota[usage_key_full][m_key] = 0
            
    used_d = quota["d_u"].get(m_key,0)
    used_w = quota["w_u"].get(m_key,0)
    used_m = quota["m_u"].get(m_key,0)
    
    if m_key not in PLAN: # Handle case where a model key might be in MODEL_MAP but not PLAN
        logging.warning(f"Model key {m_key} not found in PLAN. Returning 0 limits.")
        return 0,0,0
        
    lim_d,lim_w,lim_m = PLAN[m_key]
    return lim_d-used_d, lim_w-used_w, lim_m-used_m

def record_use(m_key):
    global quota # Ensure we're modifying the global quota object
    for period_prefix in ["d", "w", "m"]:
        usage_key_full = f"{period_prefix}_u"
        if usage_key_full not in quota: quota[usage_key_full] = {}
        if m_key not in quota[usage_key_full]: quota[usage_key_full][m_key] = 0
        quota[usage_key_full][m_key]+=1
    _save(QUOTA_FILE,quota) # Save updated quota to disk
    logging.info(f"Recorded use for model {m_key}. Daily remaining: {remaining(m_key)[0]}")


# ───────── Persistent sessions ─
sessions = _load(SESS_FILE,{})
def _new_sid():
    sid=str(int(time.time()*1000)); sessions[sid]={"title":"New chat","messages":[]}
    _save(SESS_FILE,sessions); return sid

# ───────── Logging ──────────
logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)

def api_post(payload, *, stream=False, timeout=DEFAULT_TIMEOUT):
    h={"Authorization":f"Bearer {OPENROUTER_API_KEY}","Content-Type":"application/json"}
    # Log essential info, avoid logging full messages if too long or sensitive
    log_payload = payload.copy()
    if "messages" in log_payload:
        log_payload["messages"] = f"<{len(log_payload['messages'])} messages>" if len(log_payload["messages"]) > 1 else str(log_payload["messages"])[:200]
    logging.info(f"API POST. Model: {payload.get('model')}, Stream: {stream}, Max_tokens: {payload.get('max_tokens')}, Payload: {log_payload}")

    return requests.post(f"{OPENROUTER_API_BASE}/chat/completions",
                         headers=h,json=payload,stream=stream,timeout=timeout)

# ───────── OpenRouter helpers ─
def streamed(model_id, msgs, max_tokens_for_model_output):
    p = { "model": model_id, "messages": msgs, "stream": True, "max_tokens": max_tokens_for_model_output }
    with api_post(p,stream=True) as r:
        try: r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logging.error(f"API Stream Error for {model_id}: {e}. Response: {r.text}")
            yield None, f"API Error ({e.response.status_code}): {r.text}"; return
        for ln in r.iter_lines():
            if not ln: continue
            if ln.startswith(b"data: "):
                data=ln[6:].decode('utf-8').strip()
                if data=="[DONE]": break
                try: chunk=json.loads(data)
                except json.JSONDecodeError: logging.error(f"JSONDecodeError: '{data}'"); yield None, "Stream decode error."; return
                if "error" in chunk: logging.error(f"Stream Chunk Error: {chunk['error']}"); yield None, chunk["error"].get("message","API error in chunk"); return
                delta_content = chunk.get("choices",[{}])[0].get("delta",{}).get("content")
                if delta_content is not None: yield delta_content, None
            else: logging.warning(f"Unexpected stream line: {ln}")


def route_choice(user_msg, allowed_model_keys):
    if not allowed_model_keys: logging.warning("route_choice: no allowed_model_keys."); return None
    system_prompt_lines = [
        "You are an intelligent model routing assistant. Your task is to choose the most suitable Large Language Model for the user's query.",
        "Based on the user's message, select ONLY ONE model letter from the following list of *currently available* models.",
        "Here are the descriptions of the available models (model letter, emoji, actual model ID, description):"
    ]
    for key in allowed_model_keys:
        system_prompt_lines.append(f"- {key}: {MODEL_DESCRIPTIONS.get(key, f'({MODEL_MAP.get(key, EMOJI.get(key,"?"))}) No specific description.')}")
    system_prompt_lines.append("\nConsider the user's query carefully. Respond with ONLY the single capital letter of your choice. No other text.")
    final_system_prompt = "\n".join(system_prompt_lines)
    msgs_for_router = [{"role": "system", "content": final_system_prompt}, {"role": "user", "content": user_msg}]
    payload_for_router = {"model": ROUTER_MODEL_ID, "messages": msgs_for_router, "temperature": 0.1}
    try:
        logging.info(f"Router ({ROUTER_MODEL_ID}) choosing from: {allowed_model_keys}")
        r = api_post(payload_for_router)
        r.raise_for_status()
        response_json = r.json()
        raw_text = response_json.get("choices",[{}])[0].get("message",{}).get("content","").strip().upper()
        logging.info(f"Router ({ROUTER_MODEL_ID}) raw response: '{raw_text}'")
        for char_code in raw_text:
            if char_code in allowed_model_keys: logging.info(f"Router chose: {char_code}"); return char_code
        logging.warning(f"Router bad choice '{raw_text}'. Defaulting."); return allowed_model_keys[0]
    except Exception as e:
        logging.error(f"Router API call failed: {e}. Defaulting."); return allowed_model_keys[0] if allowed_model_keys else None

def generate_chat_title(chat_messages_for_title, title_model):
    if len(chat_messages_for_title) < 1: return None # Need at least one message

    # Prepare a concise context for the title generation model
    context = ""
    user_query = ""
    assistant_response = ""

    if chat_messages_for_title[0]['role'] == 'user':
        user_query = chat_messages_for_title[0]['content']
        context += f"User: {user_query[:200]}...\n" # Truncate for brevity
    if len(chat_messages_for_title) > 1 and chat_messages_for_title[1]['role'] == 'assistant':
        assistant_response = chat_messages_for_title[1]['content']
        context += f"Assistant: {assistant_response[:200]}...\n" # Truncate

    if not user_query: # Should not happen if we check len >= 1
        return None

    title_prompt = (
        f"Based on the following start of a conversation, suggest a very short and concise title (3-5 words ideally, max 7 words) for this chat session. "
        f"The title should capture the main topic or question. Do not use quotation marks in the title. Only return the title itself.\n\n"
        f"CONVERSATION START:\n{context.strip()}\n\nCONCISE TITLE:"
    )
    
    title_payload = {
        "model": title_model,
        "messages": [{"role": "user", "content": title_prompt}],
        "max_tokens": 20, # Titles should be short
        "temperature": 0.3
    }
    try:
        logging.info(f"Requesting title generation from {title_model} based on: '{context.strip()[:100]}...'")
        r = api_post(title_payload)
        r.raise_for_status()
        response_json = r.json()
        generated_title = response_json.get("choices",[{}])[0].get("message",{}).get("content","").strip()
        # Clean up title: remove quotes, newlines, ensure reasonable length
        generated_title = generated_title.replace('"', '').replace("'", "").replace("\n", " ").strip()
        if generated_title and 0 < len(generated_title) <= 50 : # Basic validation
            logging.info(f"Generated title: '{generated_title}'")
            return generated_title
        else:
            logging.warning(f"Title generation produced invalid or empty title: '{generated_title}'")
            return None
    except Exception as e:
        logging.error(f"Title generation API call failed: {e}")
        return None


# ───────── Credit stats (/credits) ─
def get_credits():
    try:
        r=requests.get(f"{OPENROUTER_API_BASE}/credits", headers={"Authorization":f"Bearer {OPENROUTER_API_KEY}"}, timeout=10)
        r.raise_for_status()
        dat=r.json()["data"]; return dat["total_credits"], dat["total_usage"], dat["total_credits"]-dat["total_usage"]
    except Exception as e: logging.warning(f"Could not fetch credits: {e}"); return None, None, None

def update_credits_state():
    st.session_state.CRED_TOTAL, st.session_state.CRED_USED, st.session_state.CRED_LEFT = get_credits()
    st.session_state.credits_ts = time.time()

# ───────── Streamlit page config ─
st.set_page_config(page_title="OpenRouter Chat", layout="wide", initial_sidebar_state="expanded")

if "sid" not in st.session_state: st.session_state.sid = _new_sid()
if "credits_ts" not in st.session_state or \
   any(cred_key not in st.session_state for cred_key in ["CRED_TOTAL", "CRED_USED", "CRED_LEFT"]):
    update_credits_state()

# ───────── Sidebar UI ─────────────────────────────
with st.sidebar:
    st.image("https://avatars.githubusercontent.com/u/130328222?s=200&v4", width=48)
    st.title("OpenRouter Chat")

    if st.button("➕  New chat", use_container_width=True):
        st.session_state.sid = _new_sid()
        st.rerun() # Rerun to reflect the new session

    # Daily Token-Jars first for visibility
    st.subheader("Daily Token-Jars")
    model_keys_for_jars = sorted(MODEL_MAP.keys())
    jar_cols=st.columns(len(model_keys_for_jars))
    for idx,m_key in enumerate(model_keys_for_jars):
        rem_d, _, _ = remaining(m_key)
        plan_d, _, _ = PLAN.get(m_key, (0,0,0)) # Default plan if key missing
        pct_float = 1.0 if plan_d > 900_000 else max(0.0, float(rem_d) / plan_d if plan_d > 0 else 0.0)
        fill = int(pct_float * 100)
        color = '#4caf50' if pct_float > 0.5 else ('#ff9800' if pct_float > 0.25 else '#f44336')
        jar_html=f"""<div style="width:44px;margin:auto;text-align:center;font-size:12px"><div style="border:2px solid #555;border-radius:6px;height:60px;position:relative;overflow:hidden;background:#f0f0f0;"><div style="position:absolute;bottom:0;width:100%;height:{fill}%;background:{color};"></div><div style="position:absolute;top:2px;left:0;right:0;font-size:18px; line-height:1;">{EMOJI.get(m_key, "❓")}</div><div style="position:absolute;bottom:2px;left:0;right:0; font-size:10px; font-weight:bold;">{m_key}</div></div><span>{'∞' if plan_d > 900_000 else rem_d}</span></div>"""
        jar_cols[idx].markdown(jar_html, unsafe_allow_html=True)
    st.markdown("---") # Separator after jars

    st.subheader("Chats")
    sorted_sids = sorted(sessions.keys(), key=lambda x: int(x), reverse=True)
    for sid_key in sorted_sids:
        obj = sessions.get(sid_key, {"title": "Error: Missing Session"})
        lbl=obj.get("title","Untitled")[:25] or "Untitled"
        if st.button(lbl, key=f"chatbtn_{sid_key}", use_container_width=True):
            st.session_state.sid=sid_key; st.rerun()
    st.markdown("---")

    st.subheader("Model-routing map")
    st.caption(f"Router engine: **{ROUTER_MODEL_ID}**")
    with st.expander("Letters → Models (Descriptions for Router)"):
        for key_desc, desc_val in MODEL_DESCRIPTIONS.items():
            if key_desc in MODEL_MAP:
                st.markdown(f"**{key_desc}**: {desc_val} `(Max Output: {MAX_TOKENS.get(key_desc, 'N/A')})`")
    with st.expander("Account stats (credits)"):
        if st.button("Refresh Credits", key="refresh_credits_btn"): update_credits_state(); st.rerun()
        if st.session_state.CRED_TOTAL is None: st.warning("Couldn’t fetch /credits endpoint.")
        else:
            st.markdown(f"**Purchased:** {st.session_state.CRED_TOTAL:.2f} cr")
            st.markdown(f"**Used:** {st.session_state.CRED_USED:.2f} cr")
            st.markdown(f"**Remaining:** {st.session_state.CRED_LEFT:.2f} cr")
            st.caption(f"Last updated: {datetime.fromtimestamp(st.session_state.credits_ts).strftime('%Y-%m-%d %H:%M:%S')}")

# ───────── Chat panel ─────────────────────────────
if st.session_state.sid not in sessions:
    logging.warning(f"Session ID {st.session_state.sid} not found. Creating new.")
    st.session_state.sid = _new_sid() # This also saves sessions

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
        logging.warning("No models available due to quota limits.")
    else:
        chosen_route_key = route_choice(prompt, allowed_model_keys)
        if not chosen_route_key: # Should always get a default from route_choice if allowed_model_keys is not empty
            st.error("❗ Model routing critically failed. Please check logs.")
        else:
            actual_model_id = MODEL_MAP[chosen_route_key]
            max_output_tokens_for_selected_model = MAX_TOKENS[chosen_route_key]
            with st.chat_message("assistant", avatar=EMOJI.get(chosen_route_key, "🤖")):
                response_box = st.empty(); full_response_content = ""; stream_successful = True
                for chunk, error_msg in streamed(actual_model_id, list(chat_messages), max_output_tokens_for_selected_model): # Pass a copy of messages
                    if error_msg:
                        full_response_content = f"❗ **API Error for {actual_model_id}**:\n\n{error_msg}"
                        response_box.error(full_response_content); stream_successful = False; break 
                    if chunk is not None: full_response_content += chunk; response_box.markdown(full_response_content + "▌")
                response_box.markdown(full_response_content)
            chat_messages.append({"role": "assistant", "content": full_response_content, "model_key": chosen_route_key})
            
            if stream_successful or not (full_response_content.startswith("❗ **API Error") and "credit" in full_response_content.lower()):
                 record_use(chosen_route_key) # This updates global `quota` and saves to file

            # Attempt to generate a title if it's still "New chat" and we have at least user + assistant message
            if chat_session.get("title") == "New chat" and len(chat_messages) >= 2:
                try:
                    # Use first user message and first assistant response for title generation context
                    title_context_messages = [chat_messages[0], chat_messages[-1]] # User query and latest assistant response
                    new_title = generate_chat_title(title_context_messages, TITLE_GENERATION_MODEL_ID)
                    if new_title:
                        chat_session["title"] = new_title
                        logging.info(f"Generated title for chat {st.session_state.sid}: '{new_title}'")
                    else: # Fallback to simple title if LLM fails
                        chat_session["title"] = chat_messages[0]['content'][:30] + "..." if chat_messages[0]['content'] else "Chat"
                        logging.warning(f"LLM title generation failed or invalid. Used fallback for chat {st.session_state.sid}.")
                except Exception as e:
                    logging.error(f"Error during title generation: {e}")
                    chat_session["title"] = chat_messages[0]['content'][:30] + "..." if chat_messages[0]['content'] else "Chat" # Fallback

            _save(SESS_FILE, sessions) # Save session data (including new title and messages)
            st.rerun() # CRITICAL: This will re-run the script, updating the UI including token jars and chat list with new title.

# ───────── Self-relaunch when run via `python main.py` ─
if __name__=="__main__" and os.getenv("_IS_STRL")!="1":
    os.environ["_IS_STRL"]="1"; port=os.getenv("PORT","8501"); script_path = os.path.abspath(__file__)
    logging.info(f"Starting Streamlit server on port {port} for script: {script_path}")
    try: subprocess.run([sys.executable, "-m", "streamlit", "run", script_path, "--server.port",port,"--server.address","0.0.0.0"], check=True)
    except Exception as e: logging.error(f"Failed to start Streamlit: {e}")
