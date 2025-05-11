#!/usr/bin/env python3
# -*- coding: utf-8 -*- # THIS SHOULD BE LINE 2
"""
OpenRouter Streamlit Chat — Multi-User Edition
• User accounts with persistent, per-user chat sessions & API keys
• Per-user daily/monthly quotas based on their API key usage
• Pretty ‘token-jar’ gauges (fixed at top)
• Detailed model-routing panel (Mistral router)
• Live credit/usage stats (GET /credits) per user
• Auto-titling of new chats
• Comprehensive logging
• In-app API Key configuration (via Settings panel) per user
"""

# ------------------------- Imports ------------------------- #
import json, logging, os, sys, time, requests
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo # Python 3.9+
import streamlit as st

# NEW IMPORTS FOR MULTI-USER
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.exc import IntegrityError
import bcrypt # For secure password hashing

# -------------------------- Configuration --------------------------- #
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT     = 120

FALLBACK_MODEL_ID = "deepseek/deepseek-chat-v3-0324:free"
FALLBACK_MODEL_KEY = "_FALLBACK_"
FALLBACK_MODEL_EMOJI = "🆓"
FALLBACK_MODEL_MAX_TOKENS = 8000

MODEL_MAP = {
    "A": "google/gemini-2.5-pro-preview", "B": "openai/o4-mini",
    "C": "openai/chatgpt-4o-latest", "D": "deepseek/deepseek-r1",
    "F": "google/gemini-2.5-flash-preview"
}
ROUTER_MODEL_ID = "google/gemini-2.0-flash-exp:free"
MAX_HISTORY_CHARS_FOR_ROUTER = 3000

MAX_TOKENS = {
    "A": 16_000, "B": 8_000, "C": 16_000, "D": 8_000, "F": 8_000
}

NEW_PLAN_CONFIG = { # These limits now apply PER USER
    "A": (10, 200, 5000, 100000, 5000, 100000, 3, 3 * 3600),
    "B": (10, 200, 5000, 100000, 5000, 100000, 0, 0),
    "C": (10, 200, 5000, 100000, 5000, 100000, 0, 0),
    "D": (10, 200, 5000, 100000, 5000, 100000, 0, 0),
    "F": (150, 3000, 75000, 1500000, 75000, 1500000, 0, 0)
}
EMOJI = {"A": "🌟", "B": "🔷", "C": "🟥", "D": "🟢", "F": "🌀"}
MODEL_DESCRIPTIONS = {
    "A": "🌟 (gemini-2.5-pro-preview) – top-quality, creative, expensive.",
    "B": "🔷 (o4-mini) – mid-stakes reasoning, cost-effective.",
    "C": "🟥 (chatgpt-4o-latest) – polished/empathetic, pricier.",
    "D": "🟢 (deepseek-r1) – cheap factual reasoning.",
    "F": "🌀 (gemini-2.5-flash-preview) – quick, free-tier, general purpose."
}
ROUTER_MODEL_GUIDANCE = {
    "A": "(Model A: Top-Tier Quality & Capability)...", # Keep your detailed guidance
    "B": "(Model B: Solid Mid-Tier All-Rounder)...",
    "C": "(Model C: High Quality, Polished & Empathetic)...",
    "D": "(Model D: Cost-Effective Factual & Technical)...",
    "F": "(Model F: Fast & Economical for Simple Tasks)..."
} # Ensure these are fully populated from your previous code.

TZ = ZoneInfo("Australia/Sydney")
DATA_DIR = Path(__file__).parent
# CONFIG_FILE is no longer used for API key globally.
# SESS_FILE and QUOTA_FILE are replaced by the database.

# --- Database Setup ---
DATABASE_URL = f"sqlite:///{DATA_DIR / 'multiuser_chat_data.db'}" # Changed DB name
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    openrouter_api_key = Column(String, nullable=True) # User's own API key

    sessions = relationship("ChatSessionDB", back_populates="user", cascade="all, delete-orphan")
    quotas = relationship("UserQuotaDB", back_populates="user", uselist=False, cascade="all, delete-orphan") # One-to-one

class ChatSessionDB(Base):
    __tablename__ = "chat_sessions_db"
    id = Column(String, primary_key=True, index=True) # SID
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, default="New chat")
    created_at = Column(DateTime, default=lambda: datetime.now(TZ)) # Use TZ-aware default
    messages_json = Column(Text, default="[]")

    user = relationship("User", back_populates="sessions")

class UserQuotaDB(Base):
    __tablename__ = "user_quotas_db"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False, index=True) # user_id is unique for one-to-one
    
    d_stamp = Column(String, nullable=True)
    m_stamp = Column(String, nullable=True)
    d_u_json = Column(Text, default="{}")
    m_u_json = Column(Text, default="{}")
    d_it_u_json = Column(Text, default="{}")
    m_it_u_json = Column(Text, default="{}")
    d_ot_u_json = Column(Text, default="{}")
    m_ot_u_json = Column(Text, default="{}")
    model_A_3h_calls_json = Column(Text, default="[]")

    user = relationship("User", back_populates="quotas")

Base.metadata.create_all(bind=engine)

# ------------------------ Helper Functions (General) -----------------------
def _today(): return datetime.now(TZ).date().isoformat()
def _ymonth(): return datetime.now(TZ).strftime("%Y-%m")

def format_token_count(num):
    if num is None: return "N/A"
    num = float(num)
    if num < 1000: return str(int(num))
    elif num < 1_000_000:
        formatted_num = f"{num/1000:.1f}"; return formatted_num.replace(".0", "") + "k"
    else:
        formatted_num = f"{num/1_000_000:.1f}"; return formatted_num.replace(".0", "") + "M"

def is_api_key_valid(api_key_value): # Generic validator
    return api_key_value and isinstance(api_key_value, str) and api_key_value.startswith("sk-or-")

def _autoname(seed: str) -> str:
    words = seed.strip().split(); cand = " ".join(words[:3]) or "Chat"
    return (cand[:25] + "…") if len(cand) > 25 else cand

# ------------------------ Auth Helper Functions -----------------------
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def db_create_user(db, username, password, api_key):
    hashed_pass = hash_password(password)
    db_user = User(username=username, hashed_password=hashed_pass, openrouter_api_key=api_key)
    try:
        db.add(db_user); db.commit(); db.refresh(db_user)
        init_user_quota_record(db, db_user.id) # Create initial quota for the new user
        return db_user
    except IntegrityError: db.rollback(); return None # Username probably exists

def db_get_user(db, username):
    return db.query(User).filter(User.username == username).first()

def init_user_quota_record(db, user_id):
    existing_quota = db.query(UserQuotaDB).filter(UserQuotaDB.user_id == user_id).first()
    if existing_quota:
        logging.info(f"Quota record already exists for user_id: {user_id}")
        return existing_quota

    empty_model_zeros_json = json.dumps({k: 0 for k in MODEL_MAP.keys()})
    quota_record = UserQuotaDB(
        user_id=user_id, d_stamp=_today(), m_stamp=_ymonth(),
        d_u_json=empty_model_zeros_json, m_u_json=empty_model_zeros_json,
        d_it_u_json=empty_model_zeros_json, m_it_u_json=empty_model_zeros_json,
        d_ot_u_json=empty_model_zeros_json, m_ot_u_json=empty_model_zeros_json,
        model_A_3h_calls_json="[]"
    )
    db.add(quota_record); db.commit(); db.refresh(quota_record)
    logging.info(f"Initialized quota record for user_id: {user_id}")
    return quota_record


# --------------------- Session Management (Per User from DB) -----------------------
def load_user_sessions_from_db(db, user_id: int) -> dict:
    user_sessions_db = db.query(ChatSessionDB).filter(ChatSessionDB.user_id == user_id).order_by(ChatSessionDB.created_at.desc()).all()
    sessions_dict = {}
    for sess_db in user_sessions_db:
        try: messages = json.loads(sess_db.messages_json)
        except json.JSONDecodeError: messages = []; logging.error(f"JSONDecodeError for session {sess_db.id}")
        sessions_dict[sess_db.id] = {"title": sess_db.title, "messages": messages, "created_at": sess_db.created_at}
    return sessions_dict

def save_user_session_to_db(db, user_id: int, sid: str, session_data: dict):
    sess_db = db.query(ChatSessionDB).filter(ChatSessionDB.id == sid, ChatSessionDB.user_id == user_id).first()
    messages_str = json.dumps(session_data.get("messages", []))
    if sess_db:
        sess_db.title = session_data.get("title", "New chat")
        sess_db.messages_json = messages_str
    else: # Create new
        sess_db = ChatSessionDB(
            id=sid, user_id=user_id, title=session_data.get("title", "New chat"),
            messages_json=messages_str, created_at=session_data.get("created_at", datetime.now(TZ))
        )
        db.add(sess_db)
    try: db.commit()
    except Exception as e: db.rollback(); logging.error(f"Error saving session {sid} for user {user_id}: {e}")

def db_delete_session(db, user_id: int, sid: str):
    sess_db = db.query(ChatSessionDB).filter(ChatSessionDB.id == sid, ChatSessionDB.user_id == user_id).first()
    if sess_db: db.delete(sess_db); db.commit(); logging.info(f"Deleted session {sid} for user {user_id}")

def _delete_unused_blank_sessions_db(db, user_id: int, keep_sid: str = None):
    sids_to_delete = []
    user_sessions_dict = load_user_sessions_from_db(db, user_id)
    for sid, data in user_sessions_dict.items():
        if sid == keep_sid: continue
        if data.get("title") == "New chat" and not data.get("messages"): sids_to_delete.append(sid)
    if sids_to_delete:
        for sid_del in sids_to_delete: db_delete_session(db, user_id, sid_del)
        return True
    return False

def _new_sid_db(db, user_id: int):
    _delete_unused_blank_sessions_db(db, user_id, keep_sid=None)
    sid = str(int(time.time() * 1000)) # Ensure unique SID
    new_session_data = {"title": "New chat", "messages": [], "created_at": datetime.now(TZ)}
    save_user_session_to_db(db, user_id, sid, new_session_data)
    return sid

# --------------------- Quota Management (Per User from DB) ------------------------
def get_user_quota_record(db, user_id: int) -> UserQuotaDB:
    quota_record = db.query(UserQuotaDB).filter(UserQuotaDB.user_id == user_id).first()
    if not quota_record: # Should have been created on user registration
        logging.warning(f"Quota record missing for user {user_id}, re-initializing.")
        return init_user_quota_record(db, user_id)
    return quota_record

def _reset_user_quota_period(quota_record: UserQuotaDB, period_prefix: str, current_stamp: str, model_keys_zeros_json: str) -> bool:
    data_changed = False; period_stamp_attr = f"{period_prefix}_stamp"
    if getattr(quota_record, period_stamp_attr) != current_stamp:
        setattr(quota_record, period_stamp_attr, current_stamp)
        for usage_type_suffix in ["_u", "_it_u", "_ot_u"]:
            setattr(quota_record, f"{period_prefix}{usage_type_suffix}_json", model_keys_zeros_json)
        data_changed = True
        logging.info(f"User {quota_record.user_id} quota '{period_prefix}' reset (stamp: {current_stamp}).")
    else: # Ensure all models exist in current period usage JSONs
        for usage_type_suffix in ["_u", "_it_u", "_ot_u"]:
            usage_json_attr = f"{period_prefix}{usage_type_suffix}_json"
            try: current_usage_dict = json.loads(getattr(quota_record, usage_json_attr, "{}"))
            except json.JSONDecodeError: current_usage_dict = {}
            
            needs_json_update = False
            model_keys_zeros = json.loads(model_keys_zeros_json)
            for model_k_map in model_keys_zeros.keys():
                if model_k_map not in current_usage_dict:
                    current_usage_dict[model_k_map] = 0; needs_json_update = True
            if needs_json_update:
                setattr(quota_record, usage_json_attr, json.dumps(current_usage_dict)); data_changed = True
                logging.info(f"User {quota_record.user_id} updated models in '{usage_json_attr}'.")
    return data_changed

def _ensure_user_quota_data_is_current(db, user_id: int) -> UserQuotaDB:
    quota_record = get_user_quota_record(db, user_id)
    now_d_stamp, now_m_stamp = _today(), _ymonth()
    data_was_modified = False
    active_model_keys_json = json.dumps({k: 0 for k in MODEL_MAP.keys()})

    # Clean obsolete models from usage JSONs
    for p_prefix in ["d", "m"]:
        for u_suffix in ["_u", "_it_u", "_ot_u"]:
            attr_name = f"{p_prefix}{u_suffix}_json"
            try: current_dict = json.loads(getattr(quota_record, attr_name, '{}'))
            except json.JSONDecodeError: current_dict = {}
            cleaned_dict = {k: v for k,v in current_dict.items() if k in MODEL_MAP}
            if len(cleaned_dict) != len(current_dict):
                setattr(quota_record, attr_name, json.dumps(cleaned_dict)); data_was_modified = True

    if _reset_user_quota_period(quota_record, "d", now_d_stamp, active_model_keys_json): data_was_modified = True
    if _reset_user_quota_period(quota_record, "m", now_m_stamp, active_model_keys_json): data_was_modified = True

    if "A" in NEW_PLAN_CONFIG and NEW_PLAN_CONFIG["A"][7] > 0: # Prune Model A 3h calls
        try: model_a_calls = json.loads(quota_record.model_A_3h_calls_json)
        except: model_a_calls = []
        pruned_calls = [ts for ts in model_a_calls if time.time() - ts < NEW_PLAN_CONFIG["A"][7]]
        if len(pruned_calls) != len(model_a_calls):
            quota_record.model_A_3h_calls_json = json.dumps(pruned_calls); data_was_modified = True
            logging.info(f"User {user_id}: Pruned Model A 3h calls.")
    if data_was_modified:
        try: db.commit()
        except Exception as e: db.rollback(); logging.error(f"DB error on quota save for user {user_id}: {e}")
    return quota_record # Return the (potentially updated) record

def get_user_quota_usage_and_limits(db, user_id: int, model_key: str):
    if model_key not in NEW_PLAN_CONFIG: return {}
    user_quota_record = _ensure_user_quota_data_is_current(db, user_id)
    plan = NEW_PLAN_CONFIG[model_key]
    limits = {"limit_daily_msg": plan[0], "limit_monthly_msg": plan[1], "limit_daily_in_tokens": plan[2], 
              "limit_monthly_in_tokens": plan[3], "limit_daily_out_tokens": plan[4], 
              "limit_monthly_out_tokens": plan[5], "limit_3hr_msg": plan[6] if plan[6] > 0 else float('inf')}
    try:
        d_u, m_u = json.loads(user_quota_record.d_u_json), json.loads(user_quota_record.m_u_json)
        d_it_u, m_it_u = json.loads(user_quota_record.d_it_u_json), json.loads(user_quota_record.m_it_u_json)
        d_ot_u, m_ot_u = json.loads(user_quota_record.d_ot_u_json), json.loads(user_quota_record.m_ot_u_json)
        model_a_calls = json.loads(user_quota_record.model_A_3h_calls_json)
    except json.JSONDecodeError: return {**{k.replace("limit","used"):0 for k in limits}, **limits} # Safety
    
    usage = {"used_daily_msg": d_u.get(model_key,0), "used_monthly_msg": m_u.get(model_key,0),
             "used_daily_in_tokens": d_it_u.get(model_key,0), "used_monthly_in_tokens": m_it_u.get(model_key,0),
             "used_daily_out_tokens": d_ot_u.get(model_key,0), "used_monthly_out_tokens": m_ot_u.get(model_key,0),
             "used_3hr_msg": len(model_a_calls) if model_key == "A" and plan[6] > 0 else 0}
    return {**usage, **limits}

def is_user_model_available(db, user_id: int, model_key: str) -> bool:
    if model_key not in NEW_PLAN_CONFIG: return False
    stats = get_user_quota_usage_and_limits(db, user_id, model_key)
    if not stats: return False
    if stats["used_daily_msg"] >= stats["limit_daily_msg"]: return False
    if stats["used_monthly_msg"] >= stats["limit_monthly_msg"]: return False
    if stats["used_daily_in_tokens"] >= stats["limit_daily_in_tokens"]: return False
    # ... (all other checks from previous version)
    if stats["used_monthly_in_tokens"] >= stats["limit_monthly_in_tokens"]: return False
    if stats["used_daily_out_tokens"] >= stats["limit_daily_out_tokens"]: return False
    if stats["used_monthly_out_tokens"] >= stats["limit_monthly_out_tokens"]: return False
    if model_key == "A" and stats["limit_3hr_msg"] != float('inf') and stats["used_3hr_msg"] >= stats["limit_3hr_msg"]: return False
    return True

def get_user_remaining_daily_messages(db, user_id: int, model_key: str) -> int:
    if model_key not in NEW_PLAN_CONFIG: return 0
    stats = get_user_quota_usage_and_limits(db, user_id, model_key); return max(0, stats.get("limit_daily_msg",0) - stats.get("used_daily_msg",0)) if stats else 0

def record_user_use(db, user_id: int, model_key: str, prompt_tokens: int, completion_tokens: int):
    if model_key not in MODEL_MAP: return
    user_quota_record = _ensure_user_quota_data_is_current(db, user_id)
    def _update_json_field(record_attr_name, key, inc_val):
        try: current_dict = json.loads(getattr(user_quota_record, record_attr_name, "{}"))
        except: current_dict = {}
        current_dict.setdefault(key, 0); current_dict[key] += inc_val
        setattr(user_quota_record, record_attr_name, json.dumps(current_dict))

    _update_json_field("d_u_json", model_key, 1); _update_json_field("m_u_json", model_key, 1)
    _update_json_field("d_it_u_json", model_key, prompt_tokens); _update_json_field("m_it_u_json", model_key, prompt_tokens)
    _update_json_field("d_ot_u_json", model_key, completion_tokens); _update_json_field("m_ot_u_json", model_key, completion_tokens)
    if model_key == "A" and NEW_PLAN_CONFIG["A"][6] > 0:
        try: model_a_calls = json.loads(user_quota_record.model_A_3h_calls_json)
        except: model_a_calls = []
        model_a_calls.append(time.time()); user_quota_record.model_A_3h_calls_json = json.dumps(model_a_calls)
    try: db.commit(); logging.info(f"Recorded usage for user {user_id}, model '{model_key}'.")
    except Exception as e: db.rollback(); logging.error(f"DB error recording use for user {user_id}: {e}")

# --------------------- API Calls (User-Specific Key) -----------------------
def api_post_user(user_api_key: str, payload: dict, *, stream: bool=False, timeout: int=DEFAULT_TIMEOUT):
    if not is_api_key_valid(user_api_key): raise ValueError("User API Key invalid.")
    headers = {"Authorization": f"Bearer {user_api_key}", "Content-Type": "application/json"}
    logging.info(f"API POST (User Key) model={payload.get('model')}, stream={stream}")
    try:
        response = requests.post(f"{OPENROUTER_API_BASE}/chat/completions", headers=headers, json=payload, stream=stream, timeout=timeout)
        response.raise_for_status(); return response
    except requests.exceptions.HTTPError as e: logging.error(f"API POST User failed {e.response.status_code}: {e.response.text}"); raise

def streamed_user(user_api_key:str, model: str, messages: list, max_tokens_out: int):
    payload = {"model": model, "messages": messages, "stream": True, "max_tokens": max_tokens_out}
    st.session_state.pop("last_stream_usage", None)
    try:
        with api_post_user(user_api_key, payload, stream=True) as r:
            for line in r.iter_lines():
                if not line: continue
                line_str = line.decode("utf-8")
                if line_str.startswith(": OPENROUTER PROCESSING") or not line_str.startswith("data: "): continue
                data = line_str[6:].strip()
                if data == "[DONE]": break
                try: chunk = json.loads(data)
                except json.JSONDecodeError: logging.error(f"Bad JSON: {data}"); yield None, "Error decoding chunk", False; return
                if "error" in chunk:
                    msg = chunk["error"].get("message", "API error"); is_auth_fail = "invalid_api_key" in msg.lower()
                    logging.error(f"API chunk error: {msg}"); yield None, msg, is_auth_fail; return
                if "usage" in chunk and chunk["usage"]: st.session_state.last_stream_usage = chunk["usage"]
                delta = chunk["choices"][0]["delta"].get("content")
                if delta is not None: yield delta, None, False
    except ValueError as ve: logging.error(f"Streamed VE: {ve}"); yield None, str(ve), True; # Key format issue likely
    except requests.exceptions.HTTPError as e:
        is_auth_http = (e.response.status_code == 401)
        logging.error(f"Stream HTTPError {e.response.status_code}: {e.response.text}"); yield None, f"HTTP Error: {e.response.status_code}", is_auth_http
    except Exception as e: logging.error(f"Streamed general Ex: {e}"); yield None, f"Request failed: {e}", False

# --------------------- Model Routing (User-Specific Key) -----------------------
def route_choice_user(user_api_key: str, user_msg: str, allowed: list[str], chat_history: list) -> str:
    # Determine fallback_choice_letter (same as before)
    if "F" in allowed: fallback_choice_letter = "F"
    elif allowed: fallback_choice_letter = allowed[0]
    elif "F" in MODEL_MAP: fallback_choice_letter = "F"
    elif MODEL_MAP: fallback_choice_letter = list(MODEL_MAP.keys())[0]
    else: return FALLBACK_MODEL_KEY
    if not allowed: return FALLBACK_MODEL_KEY
    if len(allowed) == 1: return allowed[0]

    # Construct history_context_str (same as before)
    history_segments = [] # ... (full logic as before)
    current_chars = 0
    relevant_history_for_router = chat_history[:-1] 
    for msg in reversed(relevant_history_for_router):
        role = msg.get("role", "assistant").capitalize() 
        content = msg.get("content", "")
        segment = f"{role}: {content}\n" 
        if current_chars + len(segment) > MAX_HISTORY_CHARS_FOR_ROUTER: break
        history_segments.append(segment)
        current_chars += len(segment)
    history_context_str = "".join(reversed(history_segments)).strip()
    if not history_context_str: history_context_str = "No prior conversation history for this session."


    # Construct system_prompt_parts (same as before, using ROUTER_MODEL_GUIDANCE)
    system_prompt_parts = [
        "You are an expert AI model routing assistant...", # Your detailed prompt
        # ... all parts ...
    ]
    system_prompt_parts.append("\nAvailable Models (select one letter):")
    for k_model_key in allowed:
        description = ROUTER_MODEL_GUIDANCE.get(k_model_key, f"(Model {k_model_key} - General purpose description; details not found).")
        system_prompt_parts.append(f"- {k_model_key}: {description}")
    # ... append all specific guidance and instructions ...
    system_prompt_parts.append("\nSpecific Selection Guidance (apply rigorously to the 'Latest User Query'):") # ... and so on
    if "F" in allowed: system_prompt_parts.append("  - If 'F' is available AND the 'Latest User Query' is simple (e.g., basic factual question, quick definition, short summary of <200 words, simple classification), CHOOSE 'F'.")
    # ... (add all other model specific guidance lines from your original code) ...
    if "D" in allowed: system_prompt_parts.append("  - If 'D' is available AND the 'Latest User Query' is primarily factual, technical, code-related, or requires straightforward logical deduction, AND 'F' (if available) is too basic, STRONGLY PREFER 'D'.")
    if "B" in allowed: system_prompt_parts.append("  - If 'B' is available, AND 'F'/'D' (if available) are insufficient for the 'Latest User Query's' general reasoning, drafting, or moderate creative needs, 'B' is a good general-purpose choice. Prefer 'B' over 'A'/'C' if peak quality/style isn't explicitly demanded.")
    system_prompt_parts.append("\n  - Guidance for Expensive Models (A, C) - Use Sparingly for 'Latest User Query':")
    if "C" in allowed: system_prompt_parts.append("    - CHOOSE 'C' ONLY if the 'Latest User Query' *explicitly requires or strongly implies a need for* a highly polished, empathetic, human-like conversational tone, or involves nuanced, multi-turn creative collaboration where its specific stylistic strengths are indispensable AND 'B' (if available) is clearly inadequate.")
    if "A" in allowed: system_prompt_parts.append("    - CHOOSE 'A' ONLY if the 'Latest User Query' involves *exceptionally* complex, multi-layered reasoning, requires generation of extensive, high-stakes creative content, or tasks demanding the absolute frontier of AI capability that *no other available model can credibly handle*.")

    system_prompt_parts.append("\nRecent Conversation History (context for the 'Latest User Query'):")
    system_prompt_parts.append(history_context_str)
    system_prompt_parts.append("\nINSTRUCTIONS: Based on all the above guidance and the provided conversation history, analyze the 'Latest User Query' (which will be the next message from the 'user' role). Then, respond with ONLY the single capital letter of your chosen model (e.g., A, B, C, D, or F). NO EXPLANATION, NO EXTRA TEXT, JUST THE LETTER.")

    final_system_message = "\n".join(system_prompt_parts)
    router_messages = [{"role": "system", "content": final_system_message}, {"role": "user", "content": user_msg}]
    payload_r = {"model": ROUTER_MODEL_ID, "messages": router_messages, "max_tokens": 10, "temperature": 0.1}

    try:
        r = api_post_user(user_api_key, payload_r) # Use user's key
        choice_data = r.json(); raw_text = choice_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()
        logging.info(f"Router raw: '{raw_text}' for query: '{user_msg}'")
        for char_in_resp in raw_text:
            if char_in_resp in allowed: st.session_state.user_router_api_key_valid = True; return char_in_resp
        logging.warning(f"Router bad response ('{raw_text}'). Fallback to '{fallback_choice_letter}'.")
        st.session_state.user_router_api_key_valid = True # Call was successful, response was just not a letter
        return fallback_choice_letter
    except ValueError: st.session_state.user_router_api_key_valid = False # Key format issue
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401: st.session_state.user_router_api_key_valid = False
    except Exception as e: logging.error(f"Router unexpected error: {e}") # Other errors don't mean key is bad
    
    logging.warning(f"Router call failed. Fallback to model letter: {fallback_choice_letter}")
    return fallback_choice_letter

# --------------------- Credits (User-Specific Key) -----------------------
def get_credits_user(user_api_key: str):
    if not is_api_key_valid(user_api_key): return None, None, None, False # Auth status
    try:
        r = requests.get(f"{OPENROUTER_API_BASE}/credits", headers={"Authorization": f"Bearer {user_api_key}"}, timeout=10)
        r.raise_for_status(); d = r.json()["data"]
        return d["total_credits"], d["total_usage"], d["total_credits"] - d["total_usage"], True
    except requests.exceptions.HTTPError as e:
        is_auth_fail = (e.response.status_code == 401)
        logging.warning(f"Credits HTTP {e.response.status_code} (User Key). AuthFail: {is_auth_fail}"); return None, None, None, not is_auth_fail
    except Exception as e: logging.warning(f"Credits Ex (User Key): {e}"); return None, None, None, True # Assume auth OK, network issue

# --------------------- UI Styling -----------------------
# load_custom_css() function - (Ensure this is defined as in the previous step)
def load_custom_css():
    css = f"""
    <style>
        :root {{
            --app-bg-color: rgb(250, 249, 245); --app-secondary-bg-color: #F0F2F6; 
            --app-text-color: #0F1116; --app-text-secondary-color: #5E6572;
            --app-primary-color: #0072C6; --app-divider-color: #E0E0E0;
            --border-radius-sm: 4px; --border-radius-md: 8px; --border-radius-lg: 12px;
            --spacing-sm: 0.5rem; --spacing-md: 1rem; --spacing-lg: 1.5rem;
            --shadow-light: 0 1px 3px 0 rgba(0,0,0,0.06), 0 1px 2px -1px rgba(0,0,0,0.05);
            --app-font: "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        body {{font-family: var(--app-font) !important; background-color: var(--app-bg-color) !important; color: var(--app-text-color) !important;}}
        .stApp {{background-color: var(--app-bg-color) !important;}}
        .main .block-container {{background-color: var(--app-bg-color); padding-top: 2rem;}}
        [data-testid="stSidebar"] {{background-color: var(--app-secondary-bg-color); padding: var(--spacing-lg) var(--spacing-md); border-right: 1px solid var(--app-divider-color);}}
        [data-testid="stSidebar"] .stImage > img {{border-radius: 50%; width: 48px !important; height: 48px !important; margin-right: var(--spacing-sm);}}
        [data-testid="stSidebar"] h1 {{font-size: 1.5rem !important; color: var(--app-primary-color); font-weight: 600; margin-bottom:0; padding-top:0.2rem;}}
        [data-testid="stSidebar"] .stButton > button {{border-radius: var(--border-radius-md); border: 1px solid var(--app-divider-color); padding: 0.6em 1em; font-size: 0.9em; background-color: transparent; color: var(--app-text-color); width: 100%; margin-bottom: var(--spacing-sm); text-align: left; font-weight: 500;}}
        [data-testid="stSidebar"] .stButton > button:hover:not(:disabled) {{border-color: var(--app-primary-color); background-color: color-mix(in srgb, var(--app-primary-color) 10%, transparent);}}
        [data-testid="stSidebar"] .stButton > button:disabled {{opacity:1.0; cursor:default; background-color: color-mix(in srgb, var(--app-primary-color) 20%, transparent) !important; border-left: 3px solid var(--app-primary-color) !important; border-top-color: var(--app-divider-color) !important; border-right-color: var(--app-divider-color) !important; border-bottom-color: var(--app-divider-color) !important; font-weight:600; color: var(--app-text-color);}}
        [data-testid="stSidebar"] [data-testid*="new_chat_button_top"] > button {{background-color: var(--app-primary-color); color:white; border-color:var(--app-primary-color); font-weight:600;}}
        [data-testid="stSidebar"] [data-testid*="new_chat_button_top"] > button:hover {{background-color: color-mix(in srgb, var(--app-primary-color) 85%, black); border-color: color-mix(in srgb, var(--app-primary-color) 85%, black);}}
        [data-testid="stSidebar"] [data-testid*="new_chat_button_top"] > button:disabled {{background-color: var(--app-primary-color) !important; color:white !important; border-color:var(--app-primary-color) !important; opacity:0.6 !important; cursor:not-allowed !important; border-left: 1px solid var(--app-primary-color) !important;}}
        [data-testid="stSidebar"] h3, [data-testid="stSidebar"] .stSubheader {{font-size:0.8rem !important; text-transform:uppercase; font-weight:700; color:var(--app-text-secondary-color); margin-top:var(--spacing-lg); margin-bottom:var(--spacing-sm); letter-spacing:0.05em;}}
        [data-testid="stSidebar"] [data-testid="stExpander"] {{border:1px solid var(--app-divider-color); border-radius:var(--border-radius-md); background-color:transparent; margin-bottom:var(--spacing-md);}}
        [data-testid="stSidebar"] [data-testid="stExpander"] summary {{padding:0.6rem var(--spacing-md) !important; font-size:0.85rem !important; font-weight:600 !important; text-transform:uppercase; color:var(--app-text-color) !important; border-bottom:1px solid var(--app-divider-color); border-top-left-radius:var(--border-radius-md); border-top-right-radius:var(--border-radius-md);}}
        [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {{background-color:color-mix(in srgb, var(--app-text-color) 5%, transparent);}}
        [data-testid="stSidebar"] [data-testid="stExpander"] div[data-testid="stExpanderDetails"] {{padding:var(--spacing-sm) var(--spacing-md) !important; background-color:var(--app-secondary-bg-color); border-bottom-left-radius:var(--border-radius-md); border-bottom-right-radius:var(--border-radius-md);}}
        [data-testid="stSidebar"] [data-testid="stExpander"][aria-label^="⚡ YOUR DAILY MODEL QUOTAS"] div[data-testid="stExpanderDetails"] {{padding-top:0.6rem !important; padding-bottom:0.2rem !important; padding-left:0.1rem !important; padding-right:0.1rem !important;}}
        [data-testid="stSidebar"] [data-testid="stExpander"][aria-label^="⚡ YOUR DAILY MODEL QUOTAS"] div[data-testid="stHorizontalBlock"] {{gap:0.25rem !important;}}
        .compact-quota-item {{display:flex; flex-direction:column; align-items:center; text-align:center; padding:0px 4px;}}
        .cq-info {{font-size:0.7rem; margin-bottom:3px; line-height:1.1; white-space:nowrap; color:var(--app-text-color);}}
        .cq-bar-track {{width:100%; height:8px; background-color:color-mix(in srgb, var(--app-text-color) 10%, transparent); border:1px solid var(--app-divider-color); border-radius:var(--border-radius-sm); overflow:hidden; margin-bottom:5px;}}
        .cq-bar-fill {{height:100%; border-radius:var(--border-radius-sm);}}
        .cq-value {{font-size:0.7rem; font-weight:bold; line-height:1;}}
        .settings-panel {{border:1px solid var(--app-divider-color); border-radius:var(--border-radius-md); padding:var(--spacing-md); margin-top:var(--spacing-sm); margin-bottom:var(--spacing-md); background-color:color-mix(in srgb, var(--app-bg-color) 60%, var(--app-secondary-bg-color) 40%);}}
        .settings-panel .stTextInput input {{border-color:color-mix(in srgb, var(--app-text-color) 30%, transparent) !important; background-color:var(--app-bg-color) !important; color:var(--app-text-color) !important;}}
        .settings-panel .stSubheader {{color:var(--app-text-color) !important; font-weight:600 !important;}}
        .settings-panel hr {{border-top:1px solid var(--app-divider-color); margin-top:0.5rem; margin-bottom:0.8rem;}}
        .detailed-quota-modelname {{font-weight:600; font-size:1.05em; margin-bottom:0.3rem; display:block; color:var(--app-primary-color);}}
        .detailed-quota-block {{font-size:0.87rem; line-height:1.6;}} .detailed-quota-block ul {{list-style-type:none; padding-left:0; margin-bottom:0.5rem;}} .detailed-quota-block li {{margin-bottom:0.15rem;}}
        [data-testid="stChatInputContainer"] {{background-color:var(--app-secondary-bg-color); border-top:1px solid var(--app-divider-color); padding:var(--spacing-sm) var(--spacing-md);}}
        [data-testid="stChatInput"] textarea {{border-color:color-mix(in srgb, var(--app-text-color) 30%, transparent) !important; border-radius:var(--border-radius-md) !important; background-color:var(--app-bg-color) !important; color:var(--app-text-color) !important;}}
        [data-testid="stChatMessage"] {{border-radius:var(--border-radius-lg); padding:var(--spacing-md) 1.25rem; margin-bottom:var(--spacing-md); box-shadow:var(--shadow-light); border:1px solid transparent; max-width:85%;}}
        [data-testid="stChatMessageUser"] {{background-color:var(--app-primary-color); color:white; margin-left:auto; border-top-right-radius:var(--border-radius-sm);}}
        [data-testid="stChatMessageAssistant"] {{background-color:var(--app-secondary-bg-color); color:var(--app-text-color); margin-right:auto; border-top-left-radius:var(--border-radius-sm); border:1px solid var(--app-divider-color);}}
        .sidebar-divider {{margin-top:var(--spacing-sm); margin-bottom:var(--spacing-sm); border:0; border-top:1px solid var(--app-divider-color);}}
        .auth-container {{max-width:400px; margin:auto; padding-top:3rem;}}
        .auth-container .stButton button {{background-color:var(--app-primary-color); color:white; border:none;}}
        .auth-container .stButton button:hover {{background-color:color-mix(in srgb, var(--app-primary-color) 85%, black);}}
        .auth-toggle-link {{font-size:0.9em; text-align:center; margin-top:1rem;}}
        .auth-toggle-link a {{color:var(--app-primary-color); text-decoration:none;}}
        .auth-toggle-link a:hover {{text-decoration:underline;}}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --------------------- Main Application (Authenticated Users) -----------------------
def main_app_content():
    st.set_page_config(page_title="OpenRouter Chat", layout="wide", initial_sidebar_state="expanded")
    load_custom_css()
    db = SessionLocal() # New DB session for this run

    user_id = st.session_state.user_id
    username = st.session_state.username
    user_api_key = st.session_state.user_openrouter_api_key

    if not is_api_key_valid(user_api_key) and not st.session_state.get("settings_panel_open"):
        st.toast("⚠️ Your OpenRouter API Key is not set. Please configure it in Settings.", icon="🔑")
        # Don't block app, but user can't make calls. Settings panel will show warning.

    # Load user's sessions if not already loaded or if user changed
    if "user_sessions" not in st.session_state or st.session_state.get("user_id_changed_flag", False):
        st.session_state.user_sessions = load_user_sessions_from_db(db, user_id)
        st.session_state.user_id_changed_flag = False

    # Initialize current_sid for the user
    if "current_sid" not in st.session_state or st.session_state.current_sid not in st.session_state.user_sessions:
        if st.session_state.user_sessions:
            # Sort by created_at (actual datetime objects)
            sorted_sids = sorted(st.session_state.user_sessions.keys(),
                                 key=lambda s: st.session_state.user_sessions[s].get("created_at", datetime.min),
                                 reverse=True)
            st.session_state.current_sid = sorted_sids[0]
        else:
            new_s = _new_sid_db(db, user_id)
            st.session_state.user_sessions = load_user_sessions_from_db(db, user_id) # Reload
            st.session_state.current_sid = new_s
    current_sid = st.session_state.current_sid

    # Ensure current_sid is valid for the loaded sessions (it should be)
    if current_sid not in st.session_state.user_sessions:
        logging.error(f"SID {current_sid} mismatch for user {user_id}. Resetting SID.")
        # Simplified reset: just create a new one if truly lost.
        st.session_state.current_sid = _new_sid_db(db, user_id)
        st.session_state.user_sessions = load_user_sessions_from_db(db, user_id)
        current_sid = st.session_state.current_sid
        st.rerun()


    # Settings panel state
    if "settings_panel_open" not in st.session_state: st.session_state.settings_panel_open = False
    
    # User-specific credits refresh
    if "user_credits" not in st.session_state: st.session_state.user_credits = {}; st.session_state.user_credits_ts = 0
    if is_api_key_valid(user_api_key): # Only attempt if key looks okay
        credits_stale = time.time() - st.session_state.get("user_credits_ts", 0) > 3600
        if credits_stale or st.session_state.user_credits_ts == 0: # Refresh if stale or never fetched
            logging.info(f"Refreshing credits for user {username}.")
            tot, usd, rem, auth_ok_credits = get_credits_user(user_api_key)
            st.session_state.user_api_key_credits_valid = auth_ok_credits # Track if key worked for /credits
            if auth_ok_credits and tot is not None:
                st.session_state.user_credits = {"total": tot, "used": usd, "remaining": rem}
            else: # Auth failed or other error, keep old credits or clear them
                 st.session_state.user_credits = {"total": 0.0, "used": 0.0, "remaining": 0.0}
            st.session_state.user_credits_ts = time.time()


    # --- Sidebar UI ---
    with st.sidebar:
        st.markdown(f"👤 Logged in: **{username}**")
        if st.button("Logout", key="logout_btn_main_app", use_container_width=True):
            # Clear all user-specific session state before logging out
            keys_to_delete = ["logged_in", "username", "user_id", "user_openrouter_api_key", 
                              "user_sessions", "current_sid", "user_credits", "user_credits_ts",
                              "user_api_key_credits_valid", "user_router_api_key_valid", 
                              "settings_panel_open", "user_id_changed_flag", "user_api_key_setup_required"]
            for key_in_state in keys_to_delete:
                if key_in_state in st.session_state:
                    del st.session_state[key_in_state]
            st.session_state.login_page_selection = "Login" # Ensure auth page router goes to login
            st.rerun()
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

        # Settings Panel Toggle & Content
        settings_label = "⚙️ Close Settings" if st.session_state.settings_panel_open else "⚙️ Settings & API Key"
        if st.button(settings_label, use_container_width=True): st.session_state.settings_panel_open = not st.session_state.settings_panel_open

        if st.session_state.settings_panel_open:
            st.markdown("<div class='settings-panel'>", unsafe_allow_html=True)
            st.subheader("🔑 Your API Key")
            key_status_msg = ""
            if not is_api_key_valid(user_api_key): key_status_msg = st.warning("API Key not set or invalid format.", icon="⚠️")
            elif st.session_state.get("user_api_key_credits_valid") is False: key_status_msg = st.error(f"Key `...{user_api_key[-4:]}` failed /credits validation.", icon="🚫")
            elif st.session_state.get("user_router_api_key_valid") is False: key_status_msg = st.error(f"Key `...{user_api_key[-4:]}` failed router validation.", icon="🚫")
            else: st.caption(f"Current key: `sk-or-...{user_api_key[-4:]}` (Looks OK)")
            
            new_key_input = st.text_input("Update OpenRouter API Key", type="password", placeholder="sk-or-...", key="user_api_key_update_input")
            if st.button("Save & Validate New Key", use_container_width=True, key="user_save_key_btn"):
                if is_api_key_valid(new_key_input):
                    with st.spinner("Validating..."): _, _, _, new_key_auth_ok = get_credits_user(new_key_input)
                    if new_key_auth_ok:
                        db_user_to_update = db.query(User).filter(User.id == user_id).first()
                        db_user_to_update.openrouter_api_key = new_key_input; db.commit()
                        st.session_state.user_openrouter_api_key = new_key_input
                        st.session_state.user_api_key_credits_valid = True # Reset validation flags
                        st.session_state.user_router_api_key_valid = True
                        st.session_state.user_credits_ts = 0 # Force credits refresh
                        st.success("API Key updated and validated!"); time.sleep(0.5); st.rerun()
                    else: st.error("New API Key failed validation. Not saved.")
                elif not new_key_input: st.warning("Field empty. No change.")
                else: st.error("Invalid API Key format.")

            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("📊 Your Model Quotas")
            # Detailed quota display (ensure _ensure_user_quota_data_is_current is called by get_user_quota_usage_and_limits)
            for m_key_loop in sorted(MODEL_MAP.keys()):
                if m_key_loop not in NEW_PLAN_CONFIG: continue
                stats = get_user_quota_usage_and_limits(db, user_id, m_key_loop)
                # (Full display logic for detailed quotas as in previous main_app, using stats)
                model_short_name = MODEL_DESCRIPTIONS.get(m_key_loop, "").split('(')[1].split(')')[0] if '(' in MODEL_DESCRIPTIONS.get(m_key_loop, "") else MODEL_MAP[m_key_loop].split('/')[-1]
                st.markdown(f"{EMOJI.get(m_key_loop, '')} <span class='detailed-quota-modelname'>{m_key_loop} ({model_short_name})</span>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1: st.markdown(f"""<div class="detailed-quota-block"><ul>
                    <li><b>Daily Msgs:</b> {stats.get('used_daily_msg',0)}/{stats.get('limit_daily_msg',0)}</li>
                    <li><b>Daily In Tok:</b> {format_token_count(stats.get('used_daily_in_tokens',0))}/{format_token_count(stats.get('limit_daily_in_tokens',0))}</li>
                    <li><b>Daily Out Tok:</b> {format_token_count(stats.get('used_daily_out_tokens',0))}/{format_token_count(stats.get('limit_daily_out_tokens',0))}</li>
                    </ul></div>""", unsafe_allow_html=True)
                with col2: st.markdown(f"""<div class="detailed-quota-block"><ul>
                    <li><b>Monthly Msgs:</b> {stats.get('used_monthly_msg',0)}/{stats.get('limit_monthly_msg',0)}</li>
                    <li><b>Monthly In Tok:</b> {format_token_count(stats.get('used_monthly_in_tokens',0))}/{format_token_count(stats.get('limit_monthly_in_tokens',0))}</li>
                    <li><b>Monthly Out Tok:</b> {format_token_count(stats.get('used_monthly_out_tokens',0))}/{format_token_count(stats.get('limit_monthly_out_tokens',0))}</li>
                    </ul></div>""", unsafe_allow_html=True)
                if m_key_loop == "A" and stats.get("limit_3hr_msg", float('inf')) != float('inf'):
                    # (Logic for 3-hour cap display as before, fetching UserQuotaDB if needed for timestamps)
                    time_until_next_msg_str = ""
                    if stats.get('used_3hr_msg',0) >= stats.get('limit_3hr_msg', float('inf')):
                        user_quota_rec_for_3h = get_user_quota_record(db, user_id) # get raw record
                        try: active_model_a_calls = json.loads(user_quota_rec_for_3h.model_A_3h_calls_json)
                        except: active_model_a_calls = []
                        if active_model_a_calls:
                            oldest_blocking_call_ts = min(active_model_a_calls) 
                            expiry_time = oldest_blocking_call_ts + NEW_PLAN_CONFIG["A"][7] 
                            time_remaining_seconds = expiry_time - time.time()
                            if time_remaining_seconds > 0:
                                mins, secs = divmod(int(time_remaining_seconds), 60); hrs, mins_rem = divmod(mins, 60)
                                time_until_next_msg_str = f" (Next in {hrs}h {mins_rem}m)" if hrs > 0 else f" (Next in {mins_rem}m {secs}s)"
                    st.markdown(f"""<div class="detailed-quota-block" style="margin-top:-0.5rem; margin-left:0.1rem;"><ul>
                        <li><b>3-Hour Msgs:</b> {stats.get('used_3hr_msg',0)}/{int(stats.get('limit_3hr_msg',0))}{time_until_next_msg_str}</li>
                        </ul></div>""", unsafe_allow_html=True)
                st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True) # End settings-panel

        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
        # Logo, Title
        logo_cols = st.columns([1,4]); logo_cols[0].image("https://avatars.githubusercontent.com/u/130328222?s=200&v=4", width=48); logo_cols[1].title("OpenRouter Chat")
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

        # Compact Quotas (User-specific)
        with st.expander("⚡ YOUR DAILY MODEL QUOTAS", expanded=True):
            # (Compact quota display logic, using get_user_remaining_daily_messages)
            active_m_keys = sorted(MODEL_MAP.keys())
            if not active_m_keys: st.caption("No models for quota tracking.")
            else:
                quota_cols_disp = st.columns(len(active_m_keys))
                for i, mk in enumerate(active_m_keys):
                    with quota_cols_disp[i]:
                        left_d = get_user_remaining_daily_messages(db, user_id, mk)
                        lim_d = NEW_PLAN_CONFIG.get(mk, (0,))[0]
                        pct_f = max(0.0, min(1.0, left_d / lim_d)) if lim_d > 0 else 0.0
                        fill_w = int(pct_f * 100); left_disp = str(left_d)
                        bar_c = "#f44336"; 
                        if pct_f > 0.5: bar_c = "#4caf50"
                        elif pct_f > 0.25: bar_c = "#ffc107"
                        st.markdown(f"""<div class="compact-quota-item"><div class="cq-info">{EMOJI.get(mk,"❔")} <b>{mk}</b></div>
                            <div class="cq-bar-track"><div class="cq-bar-fill" style="width:{fill_w}%; background-color:{bar_c};"></div></div>
                            <div class="cq-value" style="color:{bar_c};">{left_disp}</div></div>""", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
        
        # New Chat button
        current_session_data = st.session_state.user_sessions.get(current_sid, {})
        is_blank_chat = (current_session_data.get("title") == "New chat" and not current_session_data.get("messages"))
        if st.button("➕ New chat", use_container_width=True, disabled=is_blank_chat, key="user_new_chat_btn"):
            new_sid_val = _new_sid_db(db, user_id)
            st.session_state.user_sessions = load_user_sessions_from_db(db, user_id) # Reload
            st.session_state.current_sid = new_sid_val; st.rerun()

        # Chat list (User-specific)
        st.subheader("Your Chats")
        user_sids_sorted_list = sorted(st.session_state.user_sessions.keys(),
                                  key=lambda s: st.session_state.user_sessions[s].get("created_at", datetime.min),
                                  reverse=True)
        for sid_key_loop in user_sids_sorted_list:
            if sid_key_loop not in st.session_state.user_sessions: continue
            title_loop = st.session_state.user_sessions[sid_key_loop].get("title", "Untitled")
            display_title_loop = title_loop[:25] + ("…" if len(title_loop) > 25 else "")
            is_active_loop = current_sid == sid_key_loop
            if st.button(display_title_loop, key=f"sess_btn_{sid_key_loop}_u{user_id}", use_container_width=True, disabled=is_active_loop):
                if not is_active_loop:
                    _delete_unused_blank_sessions_db(db, user_id, keep_sid=sid_key_loop) # Clean before switching
                    st.session_state.current_sid = sid_key_loop; st.rerun()
        
        # Model Routing Map and Credits Expander (User-specific credits)
        # ... (These sections are largely the same as before, but ensure credit refresh uses user's key)
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
        st.subheader("Model-Routing Map")
        st.caption(f"Router: {ROUTER_MODEL_ID}") # Router model is global
        with st.expander("Letters → Models", expanded=False):
             for k_model, model_id_val in MODEL_MAP.items(): st.markdown(f"**{k_model}**: {MODEL_DESCRIPTIONS.get(k_model, model_id_val)} (max_out={MAX_TOKENS.get(k_model,0):,})")
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
        with st.expander("Your Account stats (credits)", expanded=False):
            if not is_api_key_valid(user_api_key): st.caption("API Key not set.")
            else:
                if st.button("Refresh Credits", key="user_refresh_credits_sidebar_btn"):
                    with st.spinner("Refreshing..."): tot_cr, usd_cr, rem_cr, auth_ok_cr_btn = get_credits_user(user_api_key)
                    st.session_state.user_api_key_credits_valid = auth_ok_cr_btn
                    if auth_ok_cr_btn and tot_cr is not None:
                        st.session_state.user_credits = {"total":tot_cr, "used":usd_cr, "remaining":rem_cr}; st.session_state.user_credits_ts = time.time(); st.success("Refreshed!")
                    elif not auth_ok_cr_btn: st.error("API Key auth failed.")
                    else: st.warning("Could not refresh credits.")
                    st.rerun()
                
                tot_disp, usd_disp, rem_disp = st.session_state.user_credits.get("total"), st.session_state.user_credits.get("used"), st.session_state.user_credits.get("remaining")
                if tot_disp is None: st.warning("Could not display credits.")
                else: st.markdown(f"**Remaining:** ${float(rem_disp):.2f} cr\n**Used:** ${float(usd_disp):.2f} cr")
                ts_cred_disp = st.session_state.get("user_credits_ts", 0)
                st.caption(f"Last updated: {datetime.fromtimestamp(ts_cred_disp, TZ).strftime('%-d %b, %H:%M:%S') if ts_cred_disp else 'N/A'}")

    # --- Main Chat Area UI (User-specific) ---
    active_session_data_main = st.session_state.user_sessions.get(current_sid)
    if not active_session_data_main: # Should be caught by now
        st.error("Chat session error."); st.stop()
    chat_history_main = active_session_data_main.get("messages", [])

    for msg_main in chat_history_main: # Display messages
        # (Message display logic as before)
        role_main = msg_main.get("role", "assistant"); avatar_m = "👤" if role_main == "user" else EMOJI.get(msg_main.get("model"), "🤖")
        if msg_main.get("model") == FALLBACK_MODEL_KEY: avatar_m = FALLBACK_MODEL_EMOJI
        with st.chat_message(role_main, avatar=avatar_m): st.markdown(msg_main.get("content", "*empty*"))

    if prompt_main := st.chat_input("Ask anything…", key=f"chat_input_u{user_id}_s{current_sid}"):
        chat_history_main.append({"role": "user", "content": prompt_main})
        active_session_data_main["messages"] = chat_history_main # Update in-memory
        with st.chat_message("user", avatar="👤"): st.markdown(prompt_main)

        if not is_api_key_valid(user_api_key):
            st.error("Your OpenRouter API Key is not configured. Please set it in ⚙️ Settings.")
        else:
            # Quota check and model routing (user-specific)
            # _ensure_user_quota_data_is_current(db, user_id) # Called by is_user_model_available
            allowed_models_user = [k for k in MODEL_MAP if is_user_model_available(db, user_id, k)]
            use_fallback_main, chosen_mkey_main, mid_to_use_main, max_tok_main, avatar_resp_main = (False, None, None, None, "🤖")
            st.session_state.user_router_api_key_valid = True # Assume valid before router call

            if not allowed_models_user:
                # (Set to fallback if no models available for user)
                use_fallback_main, chosen_mkey_main, mid_to_use_main, max_tok_main, avatar_resp_main = (True, FALLBACK_MODEL_KEY, FALLBACK_MODEL_ID, FALLBACK_MODEL_MAX_TOKENS, FALLBACK_MODEL_EMOJI)
                st.info(f"{FALLBACK_MODEL_EMOJI} Your quotas exhausted. Using free fallback.")
            else:
                # (Router logic as before, using route_choice_user and user_api_key)
                routed_key = route_choice_user(user_api_key, prompt_main, allowed_models_user, chat_history_main)
                if st.session_state.get("user_router_api_key_valid") is False:
                    st.error("API Key failed router validation. Check Settings."); use_fallback_main = True
                elif routed_key == FALLBACK_MODEL_KEY: use_fallback_main = True # Router chose fallback
                elif routed_key not in MODEL_MAP or not is_user_model_available(db, user_id, routed_key): use_fallback_main = True # Invalid/unavailable
                else: # Valid model chosen
                    chosen_mkey_main, mid_to_use_main, max_tok_main, avatar_resp_main = routed_key, MODEL_MAP[routed_key], MAX_TOKENS[routed_key], EMOJI.get(routed_key, "🤖")
                
                if use_fallback_main and chosen_mkey_main != FALLBACK_MODEL_KEY : # If fallback was forced by router issue or unavailable choice
                    st.warning(f"{FALLBACK_MODEL_EMOJI} Router issue or chosen model '{routed_key}' unavailable. Using free fallback.")
                    chosen_mkey_main, mid_to_use_main, max_tok_main, avatar_resp_main = (FALLBACK_MODEL_KEY, FALLBACK_MODEL_ID, FALLBACK_MODEL_MAX_TOKENS, FALLBACK_MODEL_EMOJI)


            if not mid_to_use_main and st.session_state.get("user_router_api_key_valid", True): # If no model but router key was okay
                 st.warning(f"{FALLBACK_MODEL_EMOJI} Unexpected model selection issue. Using free fallback.")
                 chosen_mkey_main, mid_to_use_main, max_tok_main, avatar_resp_main = (FALLBACK_MODEL_KEY, FALLBACK_MODEL_ID, FALLBACK_MODEL_MAX_TOKENS, FALLBACK_MODEL_EMOJI)
                 use_fallback_main = True


            if mid_to_use_main: # If a model is chosen (even fallback)
                with st.chat_message("assistant", avatar=avatar_resp_main):
                    # (Streaming logic as before, using streamed_user and user_api_key)
                    resp_placeholder, full_resp_str = st.empty(), ""
                    api_call_ok_main, final_stream_auth_fail = True, False
                    for chunk_str, err_str, stream_auth_flag in streamed_user(user_api_key, mid_to_use_main, chat_history_main, max_tok_main):
                        if stream_auth_flag: full_resp_str = "❗ **API Auth Error**: Check Key."; api_call_ok_main=False; final_stream_auth_fail=True; break
                        if err_str: full_resp_str = f"❗ **API Error**: {err_str}"; api_call_ok_main=False; break
                        if chunk_str: full_resp_str += chunk_str; resp_placeholder.markdown(full_resp_str + "▌")
                    resp_placeholder.markdown(full_resp_str)

                if final_stream_auth_fail: st.session_state.user_api_key_credits_valid = False # Update key status

                # (Token recording and session saving as before, using record_user_use and user_id)
                last_usage_main = st.session_state.pop("last_stream_usage", None)
                p_tok, c_tok = (last_usage_main.get("prompt_tokens",0), last_usage_main.get("completion_tokens",0)) if last_usage_main else (0,0)
                if not last_usage_main and api_call_ok_main: logging.warning("No usage info after stream.")

                chat_history_main.append({"role":"assistant", "content":full_resp_str, "model":chosen_mkey_main if api_call_ok_main else FALLBACK_MODEL_KEY, "prompt_tokens": p_tok, "completion_tokens": c_tok})
                active_session_data_main["messages"] = chat_history_main

                if api_call_ok_main:
                    if not use_fallback_main and chosen_mkey_main in MODEL_MAP: record_user_use(db, user_id, chosen_mkey_main, p_tok, c_tok)
                    if active_session_data_main["title"] == "New chat" and prompt_main: active_session_data_main["title"] = _autoname(prompt_main)
                
                save_user_session_to_db(db, user_id, current_sid, active_session_data_main)
                st.rerun()
            # else: if router API key failed, error already shown.
    db.close() # Close DB session for this run

# --------------------- Authentication Pages -----------------------
def login_page_ui():
    st.set_page_config(page_title="Login - OpenRouter Chat", layout="centered")
    load_custom_css()
    db = SessionLocal()
    st.markdown("<div class='auth-container'>", unsafe_allow_html=True)
    st.title("Welcome Back!"); st.caption("Log in to access your OpenRouter Chat sessions.")
    with st.form("login_form"):
        username_li = st.text_input("Username", key="li_uname")
        password_li = st.text_input("Password", type="password", key="li_pword")
        if st.form_submit_button("Login", use_container_width=True):
            if not username_li or not password_li: st.error("All fields required.")
            else:
                user_db = db_get_user(db, username_li)
                if user_db and check_password(password_li, user_db.hashed_password):
                    st.session_state.logged_in = True
                    st.session_state.username = user_db.username
                    st.session_state.user_id = user_db.id
                    st.session_state.user_openrouter_api_key = user_db.openrouter_api_key
                    st.session_state.user_id_changed_flag = True # To reload sessions
                    st.success("Logged in!"); time.sleep(0.5); st.rerun()
                else: st.error("Invalid username or password.")
    if st.button("Don't have an account? Sign Up", key="goto_signup_btn", use_container_width=True, type="secondary"):
        st.session_state.login_page_selection = "Sign Up"; st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    db.close()

def signup_page_ui():
    st.set_page_config(page_title="Sign Up - OpenRouter Chat", layout="centered")
    load_custom_css()
    db = SessionLocal()
    st.markdown("<div class='auth-container'>", unsafe_allow_html=True)
    st.title("Create Account"); st.caption("Join to start your personalized OpenRouter Chat experience.")
    with st.form("signup_form"):
        username_su = st.text_input("Choose Username", key="su_uname")
        password_su = st.text_input("Create Password", type="password", key="su_pword")
        confirm_su = st.text_input("Confirm Password", type="password", key="su_confirm")
        apikey_su = st.text_input("Your OpenRouter API Key (sk-or-...)", type="password", key="su_apikey")
        if st.form_submit_button("Sign Up", use_container_width=True):
            if not all([username_su, password_su, confirm_su, apikey_su]): st.error("All fields are required.")
            elif password_su != confirm_su: st.error("Passwords do not match.")
            elif len(password_su) < 6: st.error("Password too short (min 6 chars).")
            elif not is_api_key_valid(apikey_su): st.error("Invalid API Key format.")
            else:
                user_created = db_create_user(db, username_su, password_su, apikey_su)
                if user_created:
                    st.success(f"Account '{username_su}' created! Please log in."); time.sleep(1)
                    st.session_state.login_page_selection = "Login"; st.rerun()
                else: st.error(f"Username '{username_su}' taken or DB error.")
    if st.button("Already have an account? Login", key="goto_login_btn", use_container_width=True, type="secondary"):
        st.session_state.login_page_selection = "Login"; st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    db.close()

# --------------------- Page Router -----------------------
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "login_page_selection" not in st.session_state: st.session_state.login_page_selection = "Login"

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", stream=sys.stdout)


if __name__ == "__main__":
    if st.session_state.logged_in:
        main_app_content()
    else:
        if st.session_state.login_page_selection == "Login":
            login_page_ui()
        elif st.session_state.login_page_selection == "Sign Up":
            signup_page_ui()
        else: # Default to login
            login_page_ui()
