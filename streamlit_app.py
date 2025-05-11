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
FALLBACK_MODEL_KEY = "*FALLBACK*"  # Internal key, not for display in jars or regular selection
FALLBACK_MODEL_EMOJI = "🆓"        # Emoji for the fallback model
FALLBACK_MODEL_MAX_TOKENS = 8000   # Max output tokens for the fallback model

# Model definitions (standard, quota-tracked models)

MODEL_MAP = {
    "A": "google/gemini-2.5-pro-preview",
    "B": "openai/o4-mini",
    "C": "openai/chatgpt-4o-latest",
    "D": "deepseek/deepseek-r1",
    "F": "google/gemini-2.5-flash-preview"
}

# Router uses Mistral 7B Instruct

ROUTER_MODEL_ID = "mistralai/mistral-7b-instruct:free"

# Token limits for outputs

MAX_TOKENS = {
    "A": 16_000, "B": 8_000, "C": 16_000,
    "D": 8_000, "F": 8_000
}

# Quota plan: (daily, weekly, monthly) messages

PLAN = {
    "A": (10, 10 * 7, 10 * 30),
    "B": (5, 5 * 7, 5 * 30),
    "C": (1, 1 * 7, 1 * 30),
    "D": (4, 4 * 7, 4 * 30),
    "F": (180, 500, 2000)
}

# Emojis for jars (does not include fallback model)

EMOJI = { "A": "🌟", "B": "🔷", "C": "🟥", "D": "🟢", "F": "🌀" }

# Descriptions shown to the router (does not include fallback model)

MODEL_DESCRIPTIONS = {
    "A": "🌟 (gemini-2.5-pro-preview) – top-quality, creative, expensive.",
    "B": "🔷 (o4-mini) – mid-stakes reasoning, cost-effective.",
    "C": "🟥 (chatgpt-4o-latest) – polished/empathetic, pricier.",
    "D": "🟢 (deepseek-r1) – cheap factual reasoning.",
    "F": "🌀 (gemini-2.5-flash-preview) – quick, free-tier, general purpose."
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

def _today():
    return date.today().isoformat()

def _yweek():
    return datetime.now(TZ).strftime("%G-%V")

def _ymonth():
    return datetime.now(TZ).strftime("%Y-%m")

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
    _reset(q, "d", _today(), zeros)
    _reset(q, "w", _yweek(), zeros)
    _reset(q, "m", _ymonth(), zeros)
    _save(QUOTA_FILE, q)
    return q

quota = _load_quota()

def remaining(key: str):
    ud = quota.get("d_u", {}).get(key, 0)
    uw = quota.get("w_u", {}).get(key, 0)
    um = quota.get("m_u", {}).get(key, 0)
    if key not in PLAN:
        logging.error(f"Attempted to get remaining quota for unknown key: {key}")
        return 0, 0, 0
    ld, lw, lm = PLAN[key]
    return ld - ud, lw - uw, lm - um

def record_use(key: str):
    if key not in MODEL_MAP:
        logging.warning(f"Attempted to record usage for unknown or non-standard model key: {key}")
        return
    for blk_key in ("d_u", "w_u", "m_u"):
        if blk_key not in quota:
            quota[blk_key] = {k: 0 for k in MODEL_MAP}
        quota[blk_key][key] = quota[blk_key].get(key, 0) + 1
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
# Set to logging.DEBUG for very detailed agent loop messages during development
logging.basicConfig(
    level=logging.INFO,  # Can be set to logging.DEBUG for dev
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stdout
)

# ─────────────────── API Calls, Search & Agent Logic ───────────────────

def api_post(payload: dict, *, stream: bool=False, timeout: int=DEFAULT_TIMEOUT):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    log_info = {k: v for k, v in payload.items() if k != "messages"} # Avoid logging full message history
    log_info["num_messages"] = len(payload.get("messages", []))
    first_message_role = payload.get("messages",[{}])[0].get("role", "unknown")
    logging.debug(f"API POST -> model={payload.get('model')}, stream={stream}, first_msg_role={first_message_role}, details: {log_info}")
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
    # Tavily POST request body uses 'query' and 'max_results' according to their API docs
    json_payload = {
        "query": query,
        "max_results": limit,
        "include_answer": True,       # Request Tavily's concise answer if available
        "include_raw_content": False  # Usually not needed for chat, reduces token count
    }
    try:
        r = requests.post(TAVILY_SEARCH_BASE, headers=headers, json=json_payload, timeout=15)
        r.raise_for_status()
        results = r.json()
        logging.info(f"TAVILY SEARCH SUCCESS: Results count: {len(results.get('results',[]))}, Answer available: {'yes' if results.get('answer') else 'no'}")
        return results
    except Exception as e:
        logging.error(f"TAVILY SEARCH FAILED: {e}")
        return {"error": str(e), "results": []} # Ensure 'results' key exists for consistent error handling by agent


FUNCTIONS = [{
    "name": "search_web", # Name of the function the model will call
    "description": (
        "Search the web for recent information, current events, or specific facts. "
        "Use this tool whenever a user's query explicitly or implicitly asks for up-to-date information "
        "that is likely outside your training data (e.g., 'today's news', 'latest X results', 'weather in Y right now', 'current price of Z')."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A concise and specific search query to find the required information. Be as specific as possible for best results."
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of search results to return (default is 3, maximum is 10)."
            }
        },
        "required": ["query"] # 'query' is mandatory
    }
}]

def run_agentic_chat(model: str, messages: list, max_tokens_out: int):
    interaction = messages.copy() # Work on a copy to not modify original session history directly

    # --- System Prompt Construction for Reliable Tool Use ---
    base_personality = "You are a helpful and friendly assistant."
    tool_guidance = (
        "You have a special capability: the 'search_web' tool. This tool allows you to access real-time information from the internet. "
        "**You MUST use the 'search_web' tool when the user's query implies a need for:**\n"
        "- Current events (e.g., 'What's today's news?', 'latest updates on topic X', 'recent developments in Y')\n"
        "- Real-time data (e.g., 'current stock price of AAPL', 'weather in London right now', 'live scores for game Z')\n"
        "- Specific facts or information that is very likely to be newer than your knowledge cutoff or highly dynamic.\n"
        "**Do NOT attempt to answer these types of questions from your internal knowledge alone.** If you don't know something that requires fresh information, use the search tool.\n\n"
        "When you decide to use 'search_web':\n"
        "1. Provide a clear and effective 'query' in the arguments.\n"
        "2. You can optionally specify a 'limit' for the number of results (default is 3 if not specified).\n"
        "After the search results are provided back to you (as a new message with role 'function'), use them to formulate your final answer to the user. "
        "If the search results are unhelpful or don't contain the answer, clearly state that.\n\n"
        "For general conversation, creative tasks, summarization of provided text, or questions about information likely to be stable and within your general training data, you can answer directly without necessarily using tools."
    )
    final_system_prompt_content = f"{base_personality}\n\n{tool_guidance}"

    # Ensure this system prompt is the first message and is correctly set
    if not interaction or interaction[0].get("role") != "system":
        interaction.insert(0, {"role": "system", "content": final_system_prompt_content})
    else:
        # If a system message exists, overwrite its content to prioritize our tool guidance
        interaction[0]["content"] = final_system_prompt_content
        logging.debug("AGENT: Overwrote existing system prompt with tool guidance.")

    max_tool_uses = 3  # Limit iterations to prevent runaway loops or excessive cost
    tool_uses_count = 0

    while tool_uses_count < max_tool_uses:
        logging.debug(f"AGENT LOOP (Iteration {tool_uses_count + 1}/{max_tool_uses}): Sending to API. Current interaction depth: {len(interaction)}")
        # For detailed debugging, uncomment to see the exact messages sent:
        # logging.debug(f"AGENT INTERACTION (Iteration {tool_uses_count + 1}):\n{json.dumps(interaction, indent=2)}")

        payload = {
            "model": model,
            "messages": interaction,
            "max_tokens": max_tokens_out,
            "functions": FUNCTIONS,
            "function_call": "auto",  # Let the model decide if/when to call the search tool
            "stream": False           # Agentic turns are simpler handled non-streamed for now
        }
        try:
            resp = api_post(payload)
            resp.raise_for_status()
            response_data = resp.json()
        except requests.exceptions.HTTPError as e:
            err_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            logging.error(f"AGENT LOOP API HTTPError: {err_msg}")
            return f"❗ **API Error in Agent Loop**: {err_msg}"
        except Exception as e:
            logging.error(f"AGENT LOOP API call failed: {e}")
            return f"❗ **API Error in Agent Loop**: {str(e)}"

        if not response_data.get("choices") or not response_data["choices"][0].get("message"):
            logging.error(f"AGENT LOOP: Unexpected API response structure: {response_data}")
            return "❗ **API Error**: Received an unexpected response from the model."

        choice = response_data["choices"][0]["message"]

        if choice.get("function_call"):
            tool_uses_count += 1
            fc = choice["function_call"]
            tool_name = fc["name"]
            logging.info(f"AGENT: Model wants to call function '{tool_name}' with args: {fc.get('arguments')}")

            # Safety break: if max tool uses reached and it's another search, guide it to synthesize
            if tool_uses_count >= max_tool_uses and tool_name == "search_web":
                 logging.warning(f"AGENT: Max tool uses ({max_tool_uses}) reached. Forcing model to synthesize response without new search.")
                 interaction.append(choice) # Append the model's request to call the function
                 interaction.append({
                     "role": "function",
                     "name": tool_name,
                     "content": json.dumps({"error": "Maximum tool uses reached. Please formulate a response based on information gathered so far or clearly state what you are still missing."})
                 })
                 # The loop will continue, and the next iteration will hit the outer 'tool_uses_count < max_tool_uses' check
                 continue

            try:
                args = json.loads(fc["arguments"])
            except json.JSONDecodeError:
                logging.error(f"AGENT: Failed to parse function arguments JSON: {fc.get('arguments')}")
                interaction.append(choice) # Append model's malformed request
                interaction.append({"role": "function", "name": tool_name, "content": json.dumps({"error": "Invalid function call arguments: Not valid JSON."})})
                continue # Let model retry or acknowledge error

            if tool_name == "search_web":
                query = args.get("query")
                limit = args.get("limit", 3) # Use a default limit for search if not provided by model
                if not query:
                    logging.warning("AGENT: 'search_web' called without a query.")
                    tool_result_content = json.dumps({"error": "Search query is required but was not provided."})
                else:
                    tool_result_content = json.dumps(search_tavily(query, limit))

                interaction.append(choice) # Append model's request to call the function
                interaction.append({"role": "function", "name": tool_name, "content": tool_result_content})
                # Loop again: model will see the function result and continue
            else:
                logging.warning(f"AGENT: Model called an unknown function: {tool_name}")
                interaction.append(choice) # Append model's request
                interaction.append({"role": "function", "name": tool_name, "content": json.dumps({"error": f"The function '{tool_name}' is not a known or available tool."})})
            continue # Continue to next iteration of the while loop to let model process tool result or error

        # If no function_call, it should be a final assistant text response
        final_content = choice.get("content")
        if final_content is not None:
            logging.info(f"AGENT LOOP: Finished. Model provided final text response.")
            return final_content
        else:
            # This case (no function call, no content) should be rare with good models
            logging.warning("AGENT LOOP: Model returned no function call and no content. Returning a placeholder message.")
            return "I am unable to provide a specific response at this time. Please try rephrasing your request."

    # If loop finishes due to max_tool_uses without a final content response
    logging.warning(f"AGENT LOOP: Exited due to max tool uses ({max_tool_uses}) without a direct final content response. Attempting one last synthesis.")
    # Add a user message to prompt for a final answer based on gathered info
    interaction.append({
        "role": "user",
        "content": "Based on our conversation and any information gathered from tools so far, please provide your best final answer now. If you still cannot answer the original request, please explain why."
    })
    payload_final_attempt = {
        "model": model,
        "messages": interaction,
        "max_tokens": max_tokens_out,
        "stream": False # No functions expected here, just direct answer
    }
    try:
        resp_final = api_post(payload_final_attempt)
        resp_final.raise_for_status()
        final_choice = resp_final.json()["choices"][0]["message"]
        final_answer_content = final_choice.get("content", "I have reached my operational limit for using tools in this turn and cannot fully process your request with further searches at this moment.")
        logging.info(f"AGENT LOOP: Forced final response after max tool uses: '{final_answer_content[:100]}...'")
        return final_answer_content
    except Exception as e_final:
        logging.error(f"AGENT LOOP: Failed to get forced final response after max tool uses: {e_final}")
        return "I have reached my operational limit for tool usage and encountered an issue while trying to provide a final summary. Please try again or rephrase."


# Streamed (simple, non-agentic) - kept for reference or if direct streaming becomes an option
def streamed(model: str, messages: list, max_tokens_out: int):
    payload = {"model": model, "messages": messages, "stream": True, "max_tokens": max_tokens_out}
    with api_post(payload, stream=True) as r:
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            text = r.text
            logging.error(f"Stream HTTPError {e.response.status_code}: {text}")
            yield None, f"HTTP {e.response.status_code}: {text}"
            return

        for line in r.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8")
            if line_str.startswith(": OPENROUTER PROCESSING"):
                logging.info(f"OpenRouter PING: {line_str.strip()}")
                continue
            if not line_str.startswith("data: "):
                logging.warning(f"Unexpected non-event-stream line: {line_str}")
                continue
            data = line_str[6:].strip()
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
        logging.warning("route_choice called with empty allowed list. Defaulting to 'F'.")
        return "F" if "F" in MODEL_MAP else (list(MODEL_MAP.keys())[0] if MODEL_MAP else FALLBACK_MODEL_KEY)

    if len(allowed) == 1:
        logging.info(f"Router: Only one model allowed ('{allowed[0]}'), selecting it directly.")
        return allowed[0]

    system_lines = [
        "You are an intelligent model-routing assistant.",
        "Select ONLY one letter from the following available models (A, B, C, D, F):"
    ]
    for k in allowed:
        if k in MODEL_DESCRIPTIONS:
            system_lines.append(f"- {k}: {MODEL_DESCRIPTIONS[k]}")
        else:
            logging.warning(f"Model key '{k}' found in 'allowed' list but not in MODEL_DESCRIPTIONS.")

    system_lines.extend([
        "Based on the user's query, choose the letter that best balances quality, speed, and cost-sensitivity.",
        "Consider if the query implies a need for complex reasoning, creativity, factual recall, or just a quick general response.",
        "Respond with ONLY the single capital letter corresponding to your choice. No extra text or explanation."
    ])
    router_messages = [
        {"role": "system", "content": "\n".join(system_lines)},
        {"role": "user",   "content": user_msg}
    ]
    payload_r = {"model": ROUTER_MODEL_ID, "messages": router_messages, "max_tokens": 10}
    try:
        r = api_post(payload_r)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip().upper()
        logging.info(f"Router raw response: '{text}'")
        for ch_char_code in text: # Iterate through characters of the response
            if ch_char_code in allowed:
                logging.info(f"Router selected model: '{ch_char_code}'")
                return ch_char_code
        logging.warning(f"Router response '{text}' did not contain an allowed model key. Allowed: {allowed}")
    except Exception as e:
        logging.error(f"Router call error: {e}")

    # Smarter fallback: if F is allowed, prefer it. Otherwise, first in allowed list.
    fallback_choice = "F" if "F" in allowed else allowed[0]
    logging.warning(f"Router failed to get valid choice or errored. Fallback to model: '{fallback_choice}'")
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

# ───────────────────────── Sidebar UI ───────────────────────────
with st.sidebar:
    st.image("https://avatars.githubusercontent.com/u/130328222?s=200&v=4", width=50)
    st.title("OpenRouter Chat")

    st.subheader("Daily Jars (Msgs Left)")
    active_model_keys_sorted = sorted(MODEL_MAP.keys()) # Iterate only over active models
    cols = st.columns(len(active_model_keys_sorted))
    for i, m_key in enumerate(active_model_keys_sorted):
        left, _, _ = remaining(m_key)
        lim, _, _  = PLAN[m_key]
        pct = 1.0 if lim > 900_000 else max(0.0, left / lim if lim > 0 else 0.0)
        fill = int(pct * 100)
        color = "#4caf50" if pct > .5 else "#ff9800" if pct > .25 else "#f44336"
        cols[i].markdown(f"""
            <div style="width:44px; margin:auto; text-align:center;">
              <div style="height:60px; border:1px solid #ccc; border-radius:7px; background:#f5f5f5; position:relative; overflow:hidden; box-shadow: inset 0 1px 2px rgba(0,0,0,0.07), 0 1px 1px rgba(0,0,0,0.05);">
                <div style="position:absolute; bottom:0; width:100%; height:{fill}%; background:{color}; box-shadow: inset 0 2px 2px rgba(255,255,255,0.3); box-sizing: border-box;"></div>
                <div style="position:absolute; top:2px; width:100%; font-size:18px; line-height:1;">{EMOJI[m_key]}</div>
                <div style="position:absolute; bottom:2px; width:100%; font-size:11px; font-weight:bold; color:#555; line-height:1;">{m_key}</div>
              </div>
              <span style="display:block; margin-top:4px; font-size:11px; font-weight:600; color:#333; line-height:1;">
                {'∞' if lim > 900_000 else left}
              </span>
            </div>""", unsafe_allow_html=True)
    st.markdown("---")

    if st.button("➕ New chat", use_container_width=True):
        st.session_state.sid = _new_sid()
        st.rerun()

    st.subheader("Chats")
    sorted_sids_list = sorted(sessions.keys(), key=lambda s: int(s), reverse=True)
    for sid_key in sorted_sids_list:
        title_text = sessions[sid_key]["title"][:25] + ("…" if len(sessions[sid_key]["title"]) > 25 else "") or "Untitled"
        if st.button(title_text, key=f"session_button_{sid_key}", use_container_width=True):
            if st.session_state.sid != sid_key:
                st.session_state.sid = sid_key
                st.rerun()
    st.markdown("---")

    st.subheader("Model-Routing Map")
    st.caption(f"Router engine: `{ROUTER_MODEL_ID}`")
    with st.expander("Letters → Models"):
        for k_model in sorted(MODEL_MAP.keys()): # Iterate only over active models
            st.markdown(f"**{k_model}**: {MODEL_DESCRIPTIONS[k_model]} (max_output={MAX_TOKENS[k_model]:,})")
    st.markdown("---")

    tot_cred, used_cred, rem_cred = st.session_state.credits["total"], st.session_state.credits["used"], st.session_state.credits["remaining"]
    with st.expander("Account stats (credits)", expanded=False):
        if st.button("Refresh Credits", key="refresh_credits_button_sidebar"):
            st.session_state.credits = dict(zip(("total","used","remaining"), get_credits()))
            st.session_state.credits_ts = time.time()
            tot_cred, used_cred, rem_cred = st.session_state.credits["total"], st.session_state.credits["used"], st.session_state.credits["remaining"]
            st.rerun()
        if tot_cred is None:
            st.warning("Could not fetch credits.")
        else:
            st.markdown(f"**Purchased:** {tot_cred:.2f} cr")
            st.markdown(f"**Used:** {used_cred:.2f} cr")
            st.markdown(f"**Remaining:** {rem_cred:.2f} cr")
            try:
                last_updated_str_sidebar = datetime.fromtimestamp(st.session_state.credits_ts).strftime('%Y-%m-%d %H:%M:%S')
                st.caption(f"Last updated: {last_updated_str_sidebar}")
            except TypeError:
                st.caption("Last updated: N/A")

# ────────────────────────── Main Chat Panel ─────────────────────
current_sid = st.session_state.sid
if current_sid not in sessions: # Should ideally not happen if UI is used correctly
    st.error("Selected chat session not found. Creating a new one.")
    current_sid = _new_sid()
    st.session_state.sid = current_sid
    st.rerun()

chat_history = sessions[current_sid]["messages"]

# Display chat messages from history
# Function calls and their results (role: "function") are part of chat_history for the agent,
# but we generally don't display them directly to the user in the chat UI for cleanliness.
for msg_idx, msg_data in enumerate(chat_history):
    role = msg_data["role"]
    avatar_char = "👤" # Default for user
    if role == "assistant":
        model_key_used_for_msg = msg_data.get("model", FALLBACK_MODEL_KEY) # Get actual model key stored
        avatar_char = FALLBACK_MODEL_EMOJI if model_key_used_for_msg == FALLBACK_MODEL_KEY else EMOJI.get(model_key_used_for_msg, "🤖")
        with st.chat_message(role, avatar=avatar_char):
            st.markdown(msg_data["content"])
    elif role == "user":
        with st.chat_message(role, avatar=avatar_char):
            st.markdown(msg_data["content"])
    # We skip displaying messages with role "function" or other internal roles

if prompt := st.chat_input("Ask anything…"):
    # Append user message to main history immediately. The agent will use this copy.
    chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Determine model to use (routing or fallback)
    allowed_standard_models = [k for k in MODEL_MAP if remaining(k)[0] > 0]

    use_fallback_model_flag = False
    chosen_model_key_for_api_call = FALLBACK_MODEL_KEY # Default to fallback
    model_id_to_use_for_api_call = FALLBACK_MODEL_ID
    max_tokens_for_api_call = FALLBACK_MODEL_MAX_TOKENS
    avatar_for_assistant_response = FALLBACK_MODEL_EMOJI

    if not allowed_standard_models:
        st.info(f"{FALLBACK_MODEL_EMOJI} All standard model daily quotas exhausted. Using free fallback model.")
        use_fallback_model_flag = True
        logging.info(f"All standard quotas used. Using fallback model: {FALLBACK_MODEL_ID}")
    else:
        routed_key_result = route_choice(prompt, allowed_standard_models)
        chosen_model_key_for_api_call = routed_key_result
        model_id_to_use_for_api_call = MODEL_MAP[chosen_model_key_for_api_call]
        max_tokens_for_api_call = MAX_TOKENS[chosen_model_key_for_api_call]
        avatar_for_assistant_response = EMOJI[chosen_model_key_for_api_call]

    with st.chat_message("assistant", avatar=avatar_for_assistant_response):
        spinner_text = f"Thinking with {model_id_to_use_for_api_call.split('/')[-1].split(':')[0]}..."
        if chosen_model_key_for_api_call != FALLBACK_MODEL_KEY: # Add model letter if not fallback
            spinner_text += f" (Model {chosen_model_key_for_api_call})"
        with st.spinner(spinner_text):
            # Run agentic chat - it handles potential tool calls and returns the final answer.
            # It uses the `chat_history` which now includes the latest user prompt.
            final_answer_from_agent = run_agentic_chat(model_id_to_use_for_api_call, chat_history, max_tokens_for_api_call)
        st.markdown(final_answer_from_agent)

    # Append final assistant message to history.
    # `run_agentic_chat` appends tool call/result messages to its *internal* `interaction` copy.
    # We only save the *final* user-facing assistant response to the main `chat_history`.
    chat_history.append({
        "role": "assistant",
        "content": final_answer_from_agent,
        "model": chosen_model_key_for_api_call # Store which model (key) produced this
    })

    # We assume run_agentic_chat completed okay if it didn't return an error string
    api_call_was_successful = not final_answer_from_agent.startswith("❗ **API Error")

    if api_call_was_successful:
        if not use_fallback_model_flag: # Only record use for standard, non-fallback models
            record_use(chosen_model_key_for_api_call)

        if sessions[current_sid]["title"] == "New chat":
            # Use the original user prompt for autonaming, even if agent searched
            user_prompts_in_history = [m["content"] for m in chat_history if m["role"] == "user"]
            sessions[current_sid]["title"] = _autoname(user_prompts_in_history[-1] if user_prompts_in_history else "Chat")

    _save(SESS_FILE, sessions)
    st.rerun() # Rerun to update UI with new messages and possibly new title/quota

# ───────────────────────── Self-Relaunch ──────────────────────
if __name__ == "__main__" and os.getenv("_IS_STRL") != "1":
    os.environ["_IS_STRL"] = "1"
    port = os.getenv("PORT", "8501")
    cmd = [
        sys.executable, "-m", "streamlit", "run", __file__,
        "--server.port", port,
        "--server.address", "0.0.0.0",
    ]
    logging.info(f"Relaunching with Streamlit: {' '.join(cmd)}")
    subprocess.run(cmd, check=False)
