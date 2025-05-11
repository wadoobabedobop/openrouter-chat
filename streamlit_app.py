#!/usr/bin/env python3
"""
OpenRouter Streamlit Chat - Full Edition with Agentic Search

• Persistent chat sessions
• Daily/weekly/monthly quotas
• Pretty ‘token-jar’ gauges (fixed at top)
• Detailed model-routing panel (Mistral router)
• Live credit/usage stats (GET /credits)
• Auto-titling of new chats
• Comprehensive logging
• Self-relaunch under `python main.py`
• Agentic web search using Tavily
"""

# ───────────────────────── Imports ─────────────────────────

import json, logging, os, sys, subprocess, time, requests
from datetime import datetime, date
from pathlib import Path
from zoneinfo import ZoneInfo
import streamlit as st

# ────────────────────────── Configuration ───────────────────

OPENROUTER_API_KEY = "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" # Replace with your actual key or environment variable
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT     = 120

# Tavily Search Configuration
TAVILY_API_KEY     = "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" # Replace with your Tavily API key
TAVILY_SEARCH_BASE = "https://api.tavily.com/v1/search"

# Fallback Model Configuration (used when other quotas are exhausted)

FALLBACK_MODEL_ID = "deepseek/deepseek-chat-v3-0324:free"
FALLBACK_MODEL_KEY = "*FALLBACK*"
FALLBACK_MODEL_EMOJI = "🆓"
FALLBACK_MODEL_MAX_TOKENS = 8000

# Model definitions (standard, quota-tracked models)
MODEL_MAP = {
    "A": "google/gemini-2.5-pro-preview", "B": "openai/o4-mini",
    "C": "openai/chatgpt-4o-latest", "D": "deepseek/deepseek-r1",
    "F": "google/gemini-2.5-flash-preview"
}
ROUTER_MODEL_ID = "mistralai/mistral-7b-instruct:free"
MAX_TOKENS = {"A": 16_000, "B": 8_000, "C": 16_000, "D": 8_000, "F": 8_000}
PLAN = {
    "A": (10, 70, 300), "B": (5, 35, 150), "C": (1, 7, 30),
    "D": (4, 28, 120), "F": (180, 500, 2000)
}
EMOJI = { "A": "🌟", "B": "🔷", "C": "🟥", "D": "🟢", "F": "🌀" }
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
    if block.get(key) != stamp: block[key] = stamp; block[f"{key}_u"] = active_zeros.copy()

def _load_quota():
    zeros = {k: 0 for k in MODEL_MAP}
    q = _load(QUOTA_FILE, {})
    for pukey in ("d_u", "w_u", "m_u"):
        if pukey in q:
            udict = q[pukey]
            for k_rem in [k for k in udict if k not in MODEL_MAP]: del udict[k_rem]
    _reset(q, "d", _today(), zeros); _reset(q, "w", _yweek(), zeros); _reset(q, "m", _ymonth(), zeros)
    _save(QUOTA_FILE, q); return q
quota = _load_quota()

def remaining(key: str):
    ud,uw,um=quota.get("d_u",{}).get(key,0),quota.get("w_u",{}).get(key,0),quota.get("m_u",{}).get(key,0)
    if key not in PLAN: logging.error(f"Unknown key for remaining: {key}"); return 0,0,0
    ld,lw,lm = PLAN[key]; return ld-ud,lw-uw,lm-um

def record_use(key: str):
    if key not in MODEL_MAP: logging.warning(f"Record use for non-standard key: {key}"); return
    for blk in ("d_u","w_u","m_u"):
        if blk not in quota: quota[blk]={k:0 for k in MODEL_MAP}
        quota[blk][key]=quota[blk].get(key,0)+1
    _save(QUOTA_FILE, quota)

# ───────────────────── Session Management ───────────────────────
sessions = _load(SESS_FILE, {})
def _new_sid():
    sid = str(int(time.time()*1000)); sessions[sid]={"title":"New chat","messages":[]}; _save(SESS_FILE,sessions); return sid
def _autoname(seed:str)->str:
    cand = " ".join(seed.strip().split()[:3]) or "Chat"; return (cand[:25]+"…") if len(cand)>25 else cand

# ─────────────────────────── Logging ────────────────────────────
# Set to logging.DEBUG for very detailed agent loop messages during development
# Add %(filename)s:%(lineno)d to format for easier debugging
LOGGING_LEVEL = logging.DEBUG # <<<< MODIFIED HERE for detailed logs >>>>
logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
    stream=sys.stdout
)


# ─────────────────── API Calls, Search & Agent Logic ───────────────────

def api_post(payload: dict, *, stream: bool=False, timeout: int=DEFAULT_TIMEOUT):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    # Selective logging for api_post to avoid overly verbose message history in routine calls
    # More detailed payload logging is done within run_agentic_chat if LOGGING_LEVEL is DEBUG
    log_summary = {k: v for k, v in payload.items() if k != "messages"}
    log_summary["num_messages"] = len(payload.get("messages", []))
    logging.debug(f"API POST (Summary) -> model={payload.get('model')}, stream={stream}, details: {log_summary}")

    return requests.post(
        f"{OPENROUTER_API_BASE}/chat/completions",
        headers=headers, json=payload, stream=stream, timeout=timeout
    )

def search_tavily(query: str, limit: int = 5) -> dict:
    logging.info(f"EXEC TAVILY SEARCH: query='{query}', limit={limit}")
    headers = {
        "Authorization": f"Bearer {TAVILY_API_KEY}",
        "Content-Type": "application/json"
    }
    json_payload = {
        "query": query, "max_results": limit,
        "include_answer": True, "include_raw_content": False
    }
    try:
        r = requests.post(TAVILY_SEARCH_BASE, headers=headers, json=json_payload, timeout=15)
        r.raise_for_status()
        results = r.json()
        logging.info(f"TAVILY SEARCH SUCCESS: Results count: {len(results.get('results',[]))}, Answer: {'yes' if results.get('answer') else 'no'}")
        return results
    except Exception as e:
        logging.error(f"TAVILY SEARCH FAILED: {e}", exc_info=True)
        return {"error": str(e), "results": []}


FUNCTIONS = [{
    "name": "search_web",
    "description": (
        "Search the web for recent information, current events, or specific facts. "
        "Use this tool whenever a user's query explicitly or implicitly asks for up-to-date information "
        "that is likely outside your training data (e.g., 'today's news', 'latest X results', 'weather in Y right now', 'current price of Z')."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "A concise and specific search query. Be as specific as possible."},
            "limit": {"type": "integer", "description": "Max number of search results (default 3, max 10)."}
        }, "required": ["query"]
    }
}]

def run_agentic_chat(model: str, messages: list, max_tokens_out: int):
    interaction = messages.copy()

    base_personality = "You are a helpful and friendly assistant."
    tool_guidance = (
        "You have a special capability: the 'search_web' tool. This tool allows you to access real-time information from the internet. "
        "**You MUST use the 'search_web' tool when the user's query implies a need for:**\n"
        "- Current events (e.g., 'What's today's news?', 'latest updates on topic X')\n"
        "- Real-time data (e.g., 'current stock price of Y', 'weather in city Z right now')\n"
        "- Specific facts or information very likely to be newer than your training data.\n"
        "**Do NOT attempt to answer these types of questions from your internal knowledge alone.** "
        "When using 'search_web', formulate a clear and effective 'query'. You can also specify a 'limit' for results (default is 3). "
        "After receiving search results, use them to construct your final answer. If search results are unhelpful, state that clearly. "
        "For general conversation, creative tasks, or questions about information likely within your training data, you can answer directly without using tools."
    )
    final_system_prompt_content = f"{base_personality}\n\n{tool_guidance}"

    if not interaction or interaction[0].get("role") != "system":
        interaction.insert(0, {"role": "system", "content": final_system_prompt_content})
    else:
        interaction[0]["content"] = final_system_prompt_content
        logging.debug("AGENT: Overwrote existing system prompt with tool guidance.")

    max_tool_uses = 3; tool_uses_count = 0

    while tool_uses_count < max_tool_uses:
        current_iteration_info = f"AGENT LOOP (Iteration {tool_uses_count + 1}/{max_tool_uses})"
        logging.debug(f"{current_iteration_info}: Sending to API. Interaction depth: {len(interaction)}")

        payload_to_send = {
            "model": model, "messages": interaction, "max_tokens": max_tokens_out,
            "functions": FUNCTIONS, "function_call": "auto", "stream": False
        }
        if LOGGING_LEVEL == logging.DEBUG: # Log full payload only if DEBUG is on
            logging.debug(f"{current_iteration_info}: Payload to OpenRouter API:\n{json.dumps(payload_to_send, indent=2)}")

        response_data = None # Initialize to ensure it's defined for finally block or error logging
        try:
            resp = api_post(payload_to_send)
            resp.raise_for_status() # Raises HTTPError for 4xx/5xx responses
            response_data = resp.json() # Parse JSON response
        except requests.exceptions.HTTPError as e:
            err_body = "No response body or not text."
            if e.response is not None: err_body = e.response.text
            err_msg = f"HTTP {e.response.status_code if e.response is not None else 'Unknown'}: {err_body}"
            logging.error(f"{current_iteration_info}: API HTTPError: {err_msg}", exc_info=True)
            if LOGGING_LEVEL == logging.DEBUG: logging.error(f"Failed payload was:\n{json.dumps(payload_to_send, indent=2)}")
            return f"❗ **API Error (Agent Loop)**: {err_msg}"
        except json.JSONDecodeError as e:
            logging.error(f"{current_iteration_info}: API JSONDecodeError: {e}. Response text: {resp.text if 'resp' in locals() else 'Response object not available'}", exc_info=True)
            if LOGGING_LEVEL == logging.DEBUG: logging.error(f"Failed payload was:\n{json.dumps(payload_to_send, indent=2)}")
            return f"❗ **API Error (Agent Loop)**: Could not decode API's JSON response. {resp.text[:200] if 'resp' in locals() else ''}"
        except Exception as e: # Catch other unexpected errors (network issues, etc.)
            logging.error(f"{current_iteration_info}: API call failed (Generic Exception): {e}", exc_info=True)
            if LOGGING_LEVEL == logging.DEBUG: logging.error(f"Failed payload was:\n{json.dumps(payload_to_send, indent=2)}")
            return f"❗ **API Error (Agent Loop)**: An unexpected error occurred: {str(e)}"

        # --- CRITICAL CHECK for expected response structure ---
        if not isinstance(response_data, dict) or \
           not response_data.get("choices") or \
           not isinstance(response_data["choices"], list) or \
           not response_data["choices"] or \
           not isinstance(response_data["choices"][0], dict) or \
           "message" not in response_data["choices"][0] or \
           not isinstance(response_data["choices"][0]["message"], dict):

            logging.error(f"{current_iteration_info}: Unexpected API response structure.")
            # Log the ENTIRE problematic response_data for diagnosis
            logging.error(f"Full problematic response data:\n{json.dumps(response_data, indent=2)}")
            if LOGGING_LEVEL == logging.DEBUG: logging.error(f"Payload that resulted in this unexpected response:\n{json.dumps(payload_to_send, indent=2)}")

            # Check if OpenRouter provided a specific error object within the malformed response
            if isinstance(response_data, dict) and "error" in response_data:
                api_error_detail = response_data["error"]
                if isinstance(api_error_detail, dict):
                    api_error_message = api_error_detail.get("message", json.dumps(api_error_detail))
                else:
                    api_error_message = str(api_error_detail)
                logging.error(f"{current_iteration_info}: API returned an error object: {api_error_message}")
                return f"❗ **API Error**: {api_error_message}"
            return "❗ **API Error**: Received an improperly structured response from the model. Please check the application logs for details."

        choice = response_data["choices"][0]["message"] # Now confident this structure exists

        if choice.get("function_call"):
            tool_uses_count += 1
            fc = choice["function_call"]; tool_name = fc["name"]
            logging.info(f"{current_iteration_info}: Model wants to call '{tool_name}' with args: {fc.get('arguments')}")

            if tool_uses_count >= max_tool_uses and tool_name == "search_web":
                 logging.warning(f"{current_iteration_info}: Max tool uses ({max_tool_uses}) reached. Forcing synthesis with placeholder error for tool.")
                 interaction.append(choice)
                 interaction.append({"role":"function", "name":tool_name, "content": json.dumps({"error": "Max tool uses. Please synthesize answer based on prior info."})})
                 continue # Send this back to the LLM to synthesize.

            try:
                args = json.loads(fc["arguments"])
            except json.JSONDecodeError:
                logging.error(f"{current_iteration_info}: Bad JSON args for {tool_name}: {fc.get('arguments')}", exc_info=True)
                interaction.append(choice)
                interaction.append({"role":"function", "name":tool_name, "content":json.dumps({"error":"Invalid JSON arguments provided by model."})})
                continue

            if tool_name == "search_web":
                query = args.get("query"); limit = args.get("limit", 3)
                tool_result_dict = {} # To store the dictionary from search_tavily

                if not query:
                    logging.warning(f"{current_iteration_info}: Search query missing for search_web. Args: {args}")
                    tool_result_dict = {"error": "Search query required by function."}
                else:
                    logging.info(f"{current_iteration_info}: Executing Tavily search for query='{query}', limit={limit}")
                    tool_result_dict = search_tavily(query, limit) # This returns a Python dictionary

                # <<< START ENHANCED LOGGING >>>
                logging.debug(f"{current_iteration_info}: Raw dictionary from search_tavily:\n{json.dumps(tool_result_dict, indent=2)}")
                tool_result_content_str = json.dumps(tool_result_dict) # Convert dict to JSON string for the API
                logging.debug(f"{current_iteration_info}: JSON string content for 'function' role (to be sent to LLM):\n{tool_result_content_str}")
                # <<< END ENHANCED LOGGING >>>

                interaction.append(choice) # The assistant's message with the function_call object
                interaction.append({"role":"function", "name":tool_name, "content":tool_result_content_str})
            else: # Unknown function
                logging.warning(f"{current_iteration_info}: Unknown function called: {tool_name}")
                interaction.append(choice)
                interaction.append({"role":"function", "name":tool_name, "content":json.dumps({"error":f"Unknown function: {tool_name}"})})
            continue # Go to the next iteration of the while loop to send function result to LLM

        final_content = choice.get("content")
        if final_content is not None:
            logging.info(f"{current_iteration_info}: Loop finished, model provided final text response: '{final_content[:100]}...'");
            return final_content

        # Fallback if no content and no function call (should be rare)
        logging.warning(f"{current_iteration_info}: No function call and no content in message. Full choice: {json.dumps(choice, indent=2)}")
        return "The model's response was empty or not in the expected format. Please try again."

    # After loop: Max tool uses reached
    logging.warning(f"AGENT: Exited due to max tool uses ({max_tool_uses}). Attempting final synthesis.")
    # Add a user message to guide the model for final synthesis
    interaction.append({"role":"user", "content":"Based on our conversation and any information gathered (or errors encountered), provide your best final answer. If you couldn't get information, please state that clearly."})
    payload_final = {"model":model, "messages":interaction, "max_tokens":max_tokens_out, "stream":False} # No functions for final synthesis
    if LOGGING_LEVEL == logging.DEBUG: logging.debug(f"AGENT (Final Attempt after max tool uses): Payload:\n{json.dumps(payload_final, indent=2)}")
    try:
        resp_final = api_post(payload_final); resp_final.raise_for_status()
        response_data_final = resp_final.json()
        if isinstance(response_data_final, dict) and \
           response_data_final.get("choices") and isinstance(response_data_final["choices"], list) and \
           response_data_final["choices"] and isinstance(response_data_final["choices"][0], dict) and \
           "message" in response_data_final["choices"][0] and isinstance(response_data_final["choices"][0]["message"], dict):
            final_choice_msg = response_data_final["choices"][0]["message"]
            final_answer = final_choice_msg.get("content")
            if final_answer:
                return final_answer
            else:
                logging.warning("AGENT (Final Attempt): Model gave no content in final synthesis.")
                return "Max tool uses reached, and the model's final attempt to synthesize an answer was empty."
        else:
            logging.error(f"AGENT (Final Attempt): Unexpected structure in final API response: {json.dumps(response_data_final, indent=2)}")
            return "Max tool uses reached; there was an error forming the final summary due to an unexpected API response structure."
    except Exception as e_final:
        logging.error(f"AGENT (Final Attempt): API call failed during forced final response: {e_final}", exc_info=True)
        return "Max tool uses reached; an error occurred during the final attempt to summarize."


# ───────────────────────── Model Routing ───────────────────────
def route_choice(user_msg: str, allowed: list[str]) -> str:
    if not allowed: return "F" if "F" in MODEL_MAP else (list(MODEL_MAP.keys())[0] if MODEL_MAP else FALLBACK_MODEL_KEY)
    if len(allowed)==1: logging.info(f"Router: Only one model allowed {allowed[0]}, selecting it directly."); return allowed[0]

    sys_lines = ["You are an intelligent model-routing assistant. Select ONLY one letter from available models (A,B,C,D,F):"]
    for k in allowed:
        if k in MODEL_DESCRIPTIONS: sys_lines.append(f"- {k}: {MODEL_DESCRIPTIONS[k]}")
    sys_lines.extend(["Based on user's query, choose letter for best balance of quality, speed, cost. Consider if query needs reasoning, creativity, facts, or quick general response. ONLY the single capital letter."])

    router_msgs = [{"role":"system","content":"\n".join(sys_lines)}, {"role":"user","content":user_msg}]
    payload_r = {"model":ROUTER_MODEL_ID, "messages":router_msgs, "max_tokens":10} # Increased max_tokens slightly for router
    try:
        r = api_post(payload_r); r.raise_for_status()
        raw_text = r.json()["choices"][0]["message"]["content"].strip().upper()
        logging.info(f"Router raw response: '{raw_text}'")
        # Extract first valid character
        for char_code in raw_text:
            if char_code in allowed:
                logging.info(f"Router selected model: '{char_code}'")
                return char_code
        logging.warning(f"Router response '{raw_text}' contained no valid characters from allowed: {allowed}. Falling back.")
    except Exception as e: logging.error(f"Router call error: {e}", exc_info=True)

    # Fallback logic if router fails or gives invalid response
    fallback_choice = "F" if "F" in allowed else allowed[0] # Prefer F if available and allowed
    logging.warning(f"Router falling back to: '{fallback_choice}'")
    return fallback_choice

# ───────────────────── Credits Endpoint ───────────────────────
def get_credits():
    try:
        r = requests.get(f"{OPENROUTER_API_BASE}/credits", headers={"Authorization":f"Bearer {OPENROUTER_API_KEY}"}, timeout=10)
        r.raise_for_status(); d = r.json()["data"]; return d["total_credits"], d["total_usage"], d["total_credits"]-d["total_usage"]
    except Exception as e: logging.warning(f"Could not fetch /credits: {e}", exc_info=True); return None,None,None

# ───────────────────────── Streamlit UI (largely unchanged) ────────────────────────
st.set_page_config(page_title="OpenRouter Chat", layout="wide", initial_sidebar_state="expanded")
if "sid" not in st.session_state: st.session_state.sid = _new_sid()
if "credits" not in st.session_state:
    st.session_state.credits=dict(zip(("total","used","remaining"),get_credits())); st.session_state.credits_ts=time.time()

with st.sidebar:
    st.image("https://avatars.githubusercontent.com/u/130328222?s=200&v=4",width=50); st.title("OpenRouter Chat")
    st.subheader("Daily Jars (Msgs Left)")
    active_mk_sorted = sorted(MODEL_MAP.keys()); cols = st.columns(len(active_mk_sorted))
    for i, mk_key in enumerate(active_mk_sorted):
        l_rem,_,_=remaining(mk_key); lim_plan,_,_=PLAN[mk_key]; pct_val=1.0 if lim_plan>900_000 else max(0.0,l_rem/lim_plan if lim_plan>0 else 0.0)
        fill_val=int(pct_val*100); color_val="#4caf50" if pct_val>.5 else "#ff9800" if pct_val>.25 else "#f44336"
        cols[i].markdown(f"""<div style="width:44px;margin:auto;text-align:center;"><div style="height:60px;border:1px solid #ccc;border-radius:7px;background:#f5f5f5;position:relative;overflow:hidden;box-shadow:inset 0 1px 2px rgba(0,0,0,0.07),0 1px 1px rgba(0,0,0,0.05);"><div style="position:absolute;bottom:0;width:100%;height:{fill_val}%;background:{color_val};box-shadow:inset 0 2px 2px rgba(255,255,255,0.3);"></div><div style="position:absolute;top:2px;width:100%;font-size:18px;">{EMOJI[mk_key]}</div><div style="position:absolute;bottom:2px;width:100%;font-size:11px;font-weight:bold;color:#555;">{mk_key}</div></div><span style="display:block;margin-top:4px;font-size:11px;font-weight:600;color:#333;">{'∞' if lim_plan>900_000 else l_rem}</span></div>""",unsafe_allow_html=True)
    st.markdown("---")
    if st.button("➕ New chat",use_container_width=True): st.session_state.sid=_new_sid(); st.rerun()
    st.subheader("Chats"); sorted_sids_list = sorted(sessions.keys(),key=lambda s_item:int(s_item),reverse=True)
    for sid_k in sorted_sids_list:
        title_disp = sessions[sid_k]["title"][:25]+("…" if len(sessions[sid_k]["title"])>25 else "") or "Untitled"
        if st.button(title_disp,key=f"sb_{sid_k}",use_container_width=True):
            if st.session_state.sid!=sid_k: st.session_state.sid=sid_k; st.rerun()
    st.markdown("---"); st.subheader("Model-Routing Map"); st.caption(f"Router: `{ROUTER_MODEL_ID}`")
    with st.expander("Letters → Models"):
        for k_mod in sorted(MODEL_MAP.keys()): st.markdown(f"**{k_mod}**: {MODEL_DESCRIPTIONS[k_mod]} (max_out={MAX_TOKENS[k_mod]:,})")
    st.markdown("---"); tot_c,used_c,rem_c = st.session_state.credits["total"],st.session_state.credits["used"],st.session_state.credits["remaining"]
    with st.expander("Account stats (credits)"):
        if st.button("Refresh Credits", key="refresh_credits_sidebar"):
            st.session_state.credits=dict(zip(("total","used","remaining"),get_credits())); st.session_state.credits_ts=time.time()
            tot_c,used_c,rem_c = st.session_state.credits["total"],st.session_state.credits["used"],st.session_state.credits["remaining"]; st.rerun()
        if tot_c is None: st.warning("Could not fetch credits.")
        else: st.markdown(f"**Purchased:** {tot_c:.2f} cr\n\n**Used:** {used_c:.2f} cr\n\n**Remaining:** {rem_c:.2f} cr")
        try: st.caption(f"Last updated: {datetime.fromtimestamp(st.session_state.credits_ts).strftime('%Y-%m-%d %H:%M:%S')}")
        except: st.caption("Last updated: N/A")

# ────────────────────────── Main Chat Panel ─────────────────────
current_sid = st.session_state.sid
if current_sid not in sessions:
    logging.warning(f"Session ID {current_sid} not found in sessions. Creating new one.")
    st.error("Chat session not found. New one created."); current_sid=_new_sid(); st.session_state.sid=current_sid; st.rerun()
chat_history = sessions[current_sid]["messages"]

for msg_data in chat_history:
    role_msg = msg_data["role"]
    if role_msg=="user": avatar_disp="👤"
    elif role_msg=="assistant": avatar_disp = FALLBACK_MODEL_EMOJI if msg_data.get("model")==FALLBACK_MODEL_KEY else EMOJI.get(msg_data.get("model","F"), "🤖")
    else: # Skip system or function messages for display
        if LOGGING_LEVEL == logging.DEBUG: logging.debug(f"Skipping display of message with role: {role_msg}")
        continue
    with st.chat_message(role_msg, avatar=avatar_disp): st.markdown(msg_data["content"])

if prompt := st.chat_input("Ask anything…"):
    chat_history.append({"role":"user", "content":prompt})
    with st.chat_message("user", avatar="👤"): st.markdown(prompt)

    allowed_std_mods = [k for k in MODEL_MAP if remaining(k)[0]>0]
    use_fb_flag=False; chosen_mdl_key=FALLBACK_MODEL_KEY; mdl_id_api_call=FALLBACK_MODEL_ID
    max_tok_api_call=FALLBACK_MODEL_MAX_TOKENS; avatar_assist_resp=FALLBACK_MODEL_EMOJI

    if not allowed_std_mods:
        st.info(f"{FALLBACK_MODEL_EMOJI} Quotas exhausted. Using fallback model."); use_fb_flag=True
        logging.info(f"All standard quotas used. Using fallback: {FALLBACK_MODEL_ID}")
    else:
        routed_k = route_choice(prompt, allowed_std_mods)
        if routed_k in MODEL_MAP: # Ensure router picked a valid standard model
            chosen_mdl_key=routed_k; mdl_id_api_call=MODEL_MAP[chosen_mdl_key]
            max_tok_api_call=MAX_TOKENS[chosen_mdl_key]; avatar_assist_resp=EMOJI[chosen_mdl_key]
        else: # Router might have failed or picked something not in MODEL_MAP somehow
            logging.warning(f"Router picked invalid key '{routed_k}'. Using fallback logic for standard models.")
            # Basic fallback: pick first available standard model or ultimate fallback
            if allowed_std_mods: # Should usually be true if we are in this else block
                chosen_mdl_key = allowed_std_mods[0]
                mdl_id_api_call=MODEL_MAP[chosen_mdl_key]
                max_tok_api_call=MAX_TOKENS[chosen_mdl_key]; avatar_assist_resp=EMOJI[chosen_mdl_key]
            else: # Should not happen if initial check for allowed_std_mods passed
                st.info(f"{FALLBACK_MODEL_EMOJI} Quotas exhausted or routing error. Using fallback model."); use_fb_flag=True
                logging.info(f"Routing error and no standard quotas. Using fallback: {FALLBACK_MODEL_ID}")


    with st.chat_message("assistant", avatar=avatar_assist_resp):
        spinner_txt = f"Thinking with {mdl_id_api_call.split('/')[-1].split(':')[0]}..."
        if chosen_mdl_key != FALLBACK_MODEL_KEY: spinner_txt += f" ({chosen_mdl_key})"
        with st.spinner(spinner_txt):
            # Prepare a clean copy of history for the agent; it will add its own system prompt.
            # Only send 'user' and 'assistant' roles to the agent.
            agent_history = [m for m in chat_history if m["role"] in ("user", "assistant")]
            final_ans = run_agentic_chat(mdl_id_api_call, agent_history, max_tok_api_call)
        st.markdown(final_ans)

    # Append the actual response from the assistant to the persistent chat_history
    # We store the chosen_mdl_key to correctly show emoji later.
    chat_history.append({"role":"assistant", "content":final_ans, "model":chosen_mdl_key})

    api_call_ok = not final_ans.startswith("❗ **API Error")
    if api_call_ok and not use_fb_flag and chosen_mdl_key in MODEL_MAP : record_use(chosen_mdl_key)

    if sessions[current_sid]["title"]=="New chat":
        user_prompts_hist=[m["content"] for m in chat_history if m["role"]=="user"]
        if user_prompts_hist: # Ensure there is at least one user prompt
            sessions[current_sid]["title"]=_autoname(user_prompts_hist[-1])
        else: # Fallback if no user prompts somehow (shouldn't happen in normal flow)
            sessions[current_sid]["title"] = "Chat"
            
    _save(SESS_FILE, sessions); st.rerun()

# ───────────────────────── Self-Relaunch ──────────────────────
if __name__ == "__main__" and os.getenv("_IS_STRL") != "1":
    os.environ["_IS_STRL"]="1"; port_num=os.getenv("PORT","8501")
    cmd_list=[sys.executable,"-m","streamlit","run",__file__,"--server.port",port_num,"--server.address","0.0.0.0"]
    logging.info(f"Relaunching: {' '.join(cmd_list)}"); subprocess.run(cmd_list, check=False)
