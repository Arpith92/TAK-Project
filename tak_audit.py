# tak_audit.py
from __future__ import annotations
import os, uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any

import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

IST = ZoneInfo("Asia/Kolkata")
CAND_KEYS = ["mongo_uri", "MONGO_URI", "mongodb_uri", "MONGODB_URI"]

def _find_uri() -> Optional[str]:
    try:
        import streamlit as st
        for k in CAND_KEYS:
            try: v = st.secrets.get(k)
            except Exception: v = None
            if v: return v
    except Exception:
        pass
    for k in CAND_KEYS:
        v = os.getenv(k)
        if v: return v
    return None

_client: Optional[MongoClient] = None
def _get_db():
    global _client
    if _client is None:
        uri = _find_uri()
        if not uri:
            raise RuntimeError("Mongo URI not configured for audit logs")
        _client = MongoClient(uri, appName="TAK_Audit", serverSelectionTimeoutMS=6000, tz_aware=True)
        _client.admin.command("ping")
    return _client["TAK_DB"]

def now_ist() -> datetime:
    return datetime.now(tz=IST)

def _session_id() -> str:
    import streamlit as st
    if "audit_sid" not in st.session_state:
        st.session_state["audit_sid"] = uuid.uuid4().hex
    return st.session_state["audit_sid"]

def audit_log(action: str, user: str, page: Optional[str] = None, extra: Optional[Dict[str,Any]] = None):
    try:
        db = _get_db()
        now = datetime.utcnow()
        ist = now.astimezone(IST)
        db["audit_logs"].insert_one({
            "action": str(action),
            "user": str(user or "Unknown"),
            "page": str(page or ""),
            "ts_utc": now,
            "ts_ist": ist,
            "ts_ist_str": ist.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "session_id": _session_id(),
            "extra": dict(extra or {})
        })
    except Exception:
        # never break the UI because of logging
        pass

def audit_login(user: str):
    audit_log("login", user, page="LOGIN")

def audit_pageview(user: str, page: str):
    import streamlit as st
    key = f"_audit_last_{page}"
    last = st.session_state.get(key)
    now = now_ist()
    if not last or (now - last) > timedelta(minutes=5):
        audit_log("page_view", user, page=page)
        st.session_state[key] = now

def read_logs(start_ist: Optional[datetime]=None, end_ist: Optional[datetime]=None,
              user: Optional[str]=None, action: Optional[str]=None, limit:int=2000) -> pd.DataFrame:
    db = _get_db()
    q: Dict[str, Any] = {}
    if start_ist or end_ist:
        start_utc = start_ist.astimezone(ZoneInfo("UTC")) if start_ist else None
        end_utc   = end_ist.astimezone(ZoneInfo("UTC"))   if end_ist else None
        q["ts_utc"] = {}
        if start_utc: q["ts_utc"]["$gte"] = start_utc
        if end_utc:   q["ts_utc"]["$lte"] = end_utc
    if user:   q["user"] = user
    if action: q["action"] = action

    cur = db["audit_logs"].find(q).sort("ts_utc", -1).limit(int(limit))
    rows = list(cur)
    if not rows:
        return pd.DataFrame(columns=["ts_ist_str","action","user","page","session_id","extra"])
    for r in rows:
        r.pop("_id", None)
        r["extra"] = r.get("extra", {})
        r["ts_ist_str"] = r.get("ts_ist_str") or r.get("ts_ist")
    return pd.DataFrame(rows)
