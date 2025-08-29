# app.py
from __future__ import annotations

# ---------- Compatibility shim ----------
try:
    import streamlit as st, rich
    from packaging.version import Version
    import sys, subprocess
    if Version(st.__version__) < Version("1.42.0") and Version(rich.__version__) >= Version("14.0.0"):
        subprocess.run([sys.executable, "-m", "pip", "install", "rich==13.9.4"], check=True)
        st.warning("Adjusted rich to 13.9.4 for compatibility. Rerunning…")
        st.experimental_rerun()
except Exception:
    import streamlit as st

# ---------- Std lib / deps ----------
import io, math, os, re, json
import datetime as dt
from collections.abc import Mapping
from zoneinfo import ZoneInfo
import pandas as pd
import requests
from pymongo import MongoClient

IST = ZoneInfo("Asia/Kolkata")

# ---------- App config ----------
st.set_page_config(page_title="TAK – Itinerary Generator", layout="wide")

# ---------- Master URLs ----------
CODE_FILE_URL = "https://raw.githubusercontent.com/Arpith92/TAK-Project/main/Code.xlsx"
BHASMARATHI_TYPE_URL = "https://raw.githubusercontent.com/Arpith92/TAK-Project/main/Bhasmarathi_Type.xlsx"
STAY_CITY_URL = "https://raw.githubusercontent.com/Arpith92/TAK-Project/main/Stay_City.xlsx"

# =========================================================
#                       LOGIN
# =========================================================
def _load_users() -> dict:
    try:
        raw = st.secrets.get("users", {})
        if isinstance(raw, Mapping): return dict(raw)
        if isinstance(raw, dict):    return raw
    except Exception:
        pass
    try:
        try:
            import tomllib
        except Exception:
            import tomli as tomllib
        with open(".streamlit/secrets.toml", "rb") as f:
            data = tomllib.load(f)
        u = data.get("users", {})
        if isinstance(u, Mapping): return dict(u)
        if isinstance(u, dict):    return u
    except Exception:
        pass
    return {}

def _audit_login(user: str):
    try:
        cols["audit_logins"].insert_one({
            "user": str(user),
            "ts_utc": dt.datetime.utcnow(),
            "page": "app.py",
        })
    except Exception:
        pass

def _login() -> str | None:
    with st.sidebar:
        if st.session_state.get("user"):
            st.caption(f"Signed in as **{st.session_state['user']}**")
            if st.button("Log out"):
                st.session_state.clear()
                st.rerun()

    if st.session_state.get("user"):
        return st.session_state["user"]

    users = _load_users()
    if not users:
        st.error("Configure **[users]** in Secrets.")
        st.stop()

    st.markdown("### 🔐 Login")
    c1, c2 = st.columns(2)
    with c1:
        name = st.selectbox("User", list(users.keys()))
    with c2:
        pin = st.text_input("PIN", type="password")

    if st.button("Sign in"):
        if str(users.get(name, "")).strip() == str(pin).strip():
            st.session_state["user"] = name
            _audit_login(name)
            st.success(f"Welcome, {name}!")
            st.rerun()
        else:
            st.error("Invalid PIN"); st.stop()
    return None

# =========================================================
#                    DATABASE (Mongo)
# =========================================================
def _find_uri() -> str | None:
    for k in ("mongo_uri","MONGO_URI","mongodb_uri","MONGODB_URI"):
        try: v = st.secrets.get(k)
        except Exception: v = None
        if v: return v
    for k in ("mongo_uri","MONGO_URI","mongodb_uri","MONGODB_URI"):
        v = os.getenv(k)
        if v: return v
    return None

@st.cache_resource
def _mongo():
    uri = _find_uri()
    if not uri:
        st.error("Mongo URI not configured. Add `mongo_uri` in Secrets.")
        st.stop()
    cli = MongoClient(uri, appName="TAK_App", maxPoolSize=100, serverSelectionTimeoutMS=5000, tz_aware=True)
    cli.admin.command("ping")
    db = cli["TAK_DB"]
    return {
        "itineraries": db["itineraries"],
        "audit_logins": db["audit_logins"]
    }

cols = _mongo()
col_it = cols["itineraries"]

# =========================================================
#                    HELPERS & MASTERS
# =========================================================
@st.cache_data(ttl=900)
def read_excel_from_url(url, sheet=None):
    r = requests.get(url, timeout=25); r.raise_for_status()
    return pd.read_excel(io.BytesIO(r.content), sheet_name=sheet)

try:
    stay_city_df = read_excel_from_url(STAY_CITY_URL, sheet="Stay_City")
    code_df      = read_excel_from_url(CODE_FILE_URL,      sheet="Code")
    bhas_df      = read_excel_from_url(BHASMARATHI_TYPE_URL, sheet="Bhasmarathi_Type")
except Exception as e:
    st.error(f"Could not load master sheets: {e}")
    st.stop()

def is_valid_mobile(s: str) -> bool:
    d = "".join(ch for ch in str(s or "") if ch.isdigit())
    return len(d) == 10

def in_locale(n: int) -> str: return f"{int(n):,}"
def ceil_to_999(n: float) -> int: return (math.ceil(n/1000)*1000 - 1) if n>0 else 0

def _time_list(step=15):
    base = dt.datetime(2000,1,1,0,0)
    return [(base + dt.timedelta(minutes=i)).time().strftime("%I:%M %p") for i in range(0,24*60,step)]

def _code_to_desc(code) -> str:
    s = str(code or "").strip()
    if not s or s.lower() in ("none","nan"): return "No code provided"
    try:
        m = code_df.loc[code_df["Code"].astype(str) == s, "Particulars"]
        return str(m.iloc[0]) if not m.empty else f"No description for code {s}"
    except Exception:
        return f"No description for code {s}"

def _code_to_route(code) -> str | None:
    s = str(code or "").strip()
    if not s or s.lower() in ("none","nan"): return None
    try:
        m = code_df.loc[code_df["Code"].astype(str) == s, "Route"]
        return str(m.iloc[0]) if not m.empty else None
    except Exception:
        return None

# Build option lists
stay_city_options = sorted(stay_city_df["Stay City"].dropna().astype(str).unique().tolist()) if "Stay City" in stay_city_df.columns else []
code_options  = code_df["Code"].dropna().astype(str).unique().tolist() if not code_df.empty else []
base_cars     = ["Sedan","Ertiga","Innova","Tempo Traveller"]
car_options   = [f"{ac} {c}" for c in base_cars for ac in ("AC","Non AC")]
hotel_options = ["AC Standard AC","Non-AC Standard AC","3Star AC Hotel room","4Star AC Hotel room","5Star AC Hotel room"]
room_options  = [f"{occ} occupancy {i} room" for occ in ["Double","Triple","Quad","Quint"] for i in range(1,5)]
time_options  = _time_list(15)

# =========================================================
#            LOGIN GATE + TODAY COUNTER
# =========================================================
user = _login()
if not user: st.stop()

today_ist = dt.datetime.now(IST).date()
start_today = dt.datetime.combine(today_ist, dt.time.min).astimezone(IST).astimezone(dt.timezone.utc)
end_today   = dt.datetime.combine(today_ist, dt.time.max).astimezone(IST).astimezone(dt.timezone.utc)
try:
    made_today = col_it.count_documents({"upload_date": {"$gte": start_today, "$lte": end_today}})
except Exception:
    made_today = 0

top_l, top_r = st.columns([3,1])
with top_l:
    st.title("🧭 TAK – Itinerary Generator")
with top_r:
    st.metric("Packages today", int(made_today))

# =========================================================
#               MODE SWITCH (Create / Search)
# =========================================================
mode = st.radio("Choose action", ["Create new itinerary", "Search itinerary"], horizontal=True)

# A stable token to avoid accidental table re-seeding within the same run
st.session_state.setdefault("_active_token", None)

# =============== SEARCH ===============
def _suggest(prefix: str) -> list[str]:
    """Type-ahead suggestions on name or mobile (prefix from first char)."""
    if not prefix: return []
    rx = f"^{re.escape(prefix)}"
    try:
        cur = col_it.aggregate([
            {"$match": {"$or":[
                {"client_name": {"$regex": rx, "$options": "i"}},
                {"client_mobile": {"$regex": rx}}
            ]}},
            {"$group": {"_id": {"n":"$client_name","m":"$client_mobile"}}},
            {"$project": {"_id":0, "name":"$_id.n", "mobile":"$_id.m"}},
            {"$limit": 50}
        ])
        out = []
        for x in cur:
            n = (x.get("name") or "").strip()
            m = (x.get("mobile") or "").strip()
            if n or m: out.append(f"{n} — {m}" if n and m else n or m)
        # unique preserving alpha sort
        out = sorted(set(out), key=lambda s: s.lower())
        return out
    except Exception:
        return []

def _safe_int(x, default=0):
    try: return int(float(str(x).replace(",","")))
    except Exception: return default

# ---------------------------------------------------------
#     STATE INITIALIZATION HELPERS (no first-edit loss)
# ---------------------------------------------------------
TARGET_COLS = ["Date","Time","Code","Car Type","Hotel Type","Stay City","Room Type",
               "Pkg-Car Cost","Pkg-Hotel Cost","Act-Car Cost","Act-Hotel Cost"]

def _blank_df(n_rows: int, start: dt.date) -> pd.DataFrame:
    return pd.DataFrame({
        "Date": [start + dt.timedelta(days=i) for i in range(n_rows)],
        "Time": ["" for _ in range(n_rows)],
        "Code": ["" for _ in range(n_rows)],
        "Car Type": ["" for _ in range(n_rows)],
        "Hotel Type": ["" for _ in range(n_rows)],
        "Stay City": ["" for _ in range(n_rows)],
        "Room Type": ["" for _ in range(n_rows)],
        "Pkg-Car Cost": [0.0 for _ in range(n_rows)],
        "Pkg-Hotel Cost": [0.0 for _ in range(n_rows)],
        "Act-Car Cost": [0.0 for _ in range(n_rows)],
        "Act-Hotel Cost": [0.0 for _ in range(n_rows)],
    })

def _seed_form_rows(n_rows: int, start: dt.date, force=False):
    """Create form_rows only if empty or token changed."""
    token = f"{str(start)}|{n_rows}"
    if force or st.session_state.get("_active_token") != token or "form_rows" not in st.session_state:
        st.session_state["_active_token"] = token
        st.session_state["form_rows"] = _blank_df(n_rows, start)

def _rows_from_doc(doc: dict, start: dt.date) -> pd.DataFrame:
    rows = doc.get("rows") or []
    df = pd.DataFrame(rows)
    for c in TARGET_COLS:
        if c not in df.columns:
            df[c] = 0 if "Cost" in c else ""
    # numeric fields
    for c in ["Pkg-Car Cost","Pkg-Hotel Cost","Act-Car Cost","Act-Hotel Cost"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    try:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    except Exception:
        df["Date"] = df["Date"].astype(str)
    # enforce sequential dates from start
    df = df.reset_index(drop=True)
    for i in range(len(df)):
        df.at[i, "Date"] = start + dt.timedelta(days=i)
    return df[TARGET_COLS]

# =========================================================
#                     CREATE MODE
# =========================================================
if mode == "Create new itinerary":
    # fresh start button
    if st.button("🧹 Clear & start new", type="secondary"):
        keep_user = st.session_state.get("user")
        st.session_state.clear()
        st.session_state["user"] = keep_user
        st.rerun()

    # header fields (not saved until you press generate)
    st.markdown("### Client & trip")
    c1,c2,c3,c4 = st.columns([1.6,1,1,1])
    client_name = c1.text_input("Client name*")
    client_mobile_raw = c2.text_input("Client mobile (10 digits)*")
    rep = c3.selectbox("Representative*", ["-- Select --","Arpith","Reena","Kuldeep","Teena"])
    total_pax = c4.number_input("Total Pax*", min_value=1, step=1, value=2)

    if rep == "-- Select --": st.warning("Select representative.")
    if client_name.strip()=="" or not is_valid_mobile(client_mobile_raw):
        st.info("Enter client name and valid 10-digit mobile to continue.")

    # dates
    h1, h2 = st.columns(2)
    start_date = h1.date_input("Start date", value=dt.date.today())
    days = h2.number_input("No. of days", min_value=1, step=1, value=2)

    # bhas out-of-table
    st.markdown("### Bhasmarathi")
    b1,b2,b3 = st.columns(3)
    bhas_required = b1.selectbox("Bhasmarathi required?", ["No","Yes"], index=0)
    bhas_type = b2.selectbox("Bhasmarathi Type", ["V-BH","P-BH","BH"], index=0)
    bhas_persons = b3.number_input("Persons for Bhasmarathi", min_value=0, step=1, value=0, disabled=(bhas_required=="No"))
    c_pkg, c_act = st.columns(2)
    unit_pkg = c_pkg.number_input("Unit Package Cost (₹) — Bhasmarathi", min_value=0, step=100, value=0, disabled=(bhas_required=="No"))
    unit_act = c_act.number_input("Unit Actual Cost (₹) — Bhasmarathi",  min_value=0, step=100, value=0, disabled=(bhas_required=="No"))

    # referral
    st.markdown("### Referral")
    def _ref_options():
        try:
            cur = cols["itineraries"].aggregate([
                {"$group": {"_id": {"n":"$client_name","m":"$client_mobile"}}},
                {"$project":{"_id":0,"name":"$_id.n","mobile":"$_id.m"}},
            ])
            out = []
            for r in cur:
                n = (r.get("name") or "").strip(); m = (r.get("mobile") or "").strip()
                if n or m: out.append(f"{n} — {m}" if n and m else n or m)
            return ["-- None --"] + sorted(set(out), key=lambda s:s.lower())
        except Exception:
            return ["-- None --"]
    referred_sel = st.selectbox("Referred By (applies 10% discount)", _ref_options())

    # ---------- line items table ----------
    st.markdown("### Fill line items")

    # seed df once for this (start_date, days) pair
    _seed_form_rows(int(days), start_date)

    col_cfg = {
        "Date": st.column_config.DateColumn("Date", disabled=True),
        "Time": st.column_config.SelectboxColumn("Time", options=time_options),
        "Code": st.column_config.SelectboxColumn("Code", options=code_options),
        "Car Type": st.column_config.SelectboxColumn("Car Type", options=car_options),
        "Hotel Type": st.column_config.SelectboxColumn("Hotel Type", options=hotel_options),
        "Stay City": st.column_config.SelectboxColumn("Stay City", options=stay_city_options),
        "Room Type": st.column_config.SelectboxColumn("Room Type", options=room_options),
        "Pkg-Car Cost": st.column_config.NumberColumn("Pkg-Car Cost", min_value=0.0, step=100.0),
        "Pkg-Hotel Cost": st.column_config.NumberColumn("Pkg-Hotel Cost", min_value=0.0, step=100.0),
        "Act-Car Cost": st.column_config.NumberColumn("Act-Car Cost", min_value=0.0, step=100.0),
        "Act-Hotel Cost": st.column_config.NumberColumn("Act-Hotel Cost", min_value=0.0, step=100.0),
    }

    # keep dates in sync if user changes start/days
    wanted_len = int(days)
    have_len = len(st.session_state.form_rows)
    if have_len != wanted_len:
        _seed_form_rows(wanted_len, start_date, force=True)
    # enforce dates every run
    for i in range(len(st.session_state.form_rows)):
        st.session_state.form_rows.at[i, "Date"] = start_date + dt.timedelta(days=i)

    edited = st.data_editor(
        st.session_state.form_rows,
        key="editor_rows",
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        column_config=col_cfg,
    )
    # persist only if actually changed (prevents first-edit vanishing)
    if not edited.equals(st.session_state.form_rows):
        st.session_state.form_rows = edited

    # ---------- Generate & Save (under table) ----------
    st.markdown(" ")
    gen_left, gen_right = st.columns([1,3])
    with gen_right:
        go = st.button("✅ Generate itinerary & save (rev 1)", type="primary")
    if go:
        # validations
        if not client_name.strip() or not is_valid_mobile(client_mobile_raw) or rep == "-- Select --":
            st.error("Please fill client name, valid mobile and representative.")
        else:
            df_rows = st.session_state.form_rows.copy()
            # totals
            pkg_car   = pd.to_numeric(df_rows["Pkg-Car Cost"], errors="coerce").fillna(0).sum()
            pkg_hotel = pd.to_numeric(df_rows["Pkg-Hotel Cost"], errors="coerce").fillna(0).sum()
            act_car   = pd.to_numeric(df_rows["Act-Car Cost"], errors="coerce").fillna(0).sum()
            act_hotel = pd.to_numeric(df_rows["Act-Hotel Cost"], errors="coerce").fillna(0).sum()
            bh_pkg    = (unit_pkg * bhas_persons) if bhas_required=="Yes" else 0
            bh_act    = (unit_act * bhas_persons) if bhas_required=="Yes" else 0

            package_cost_rows = float(pkg_car + pkg_hotel)
            actual_cost_rows  = float(act_car + act_hotel)
            total_package     = ceil_to_999(package_cost_rows + bh_pkg)
            total_actual      = actual_cost_rows + bh_act
            profit_total      = int(total_package - total_actual)
            has_ref           = referred_sel != "-- None --"
            after_ref         = int(round(total_package * 0.9)) if has_ref else total_package

            # meta for text
            car_types   = "-".join(pd.Series(df_rows["Car Type"]).dropna().astype(str).replace("","").unique().tolist()).strip("-")
            hotel_types = "-".join(pd.Series(df_rows["Hotel Type"]).dropna().astype(str).replace("","").unique().tolist()).strip("-")
            # route
            route_parts = [r for r in ( _code_to_route(c) for c in df_rows["Code"] ) if r]
            route_raw = "-".join(route_parts).replace(" -","-").replace("- ","-")
            route_list = [x for x in route_raw.split("-") if x]
            final_route = "-".join([route_list[i] for i in range(len(route_list)) if i == 0 or route_list[i] != route_list[i-1]])

            # bhas desc
            bhas_desc = ""
            if bhas_required=="Yes":
                m = bhas_df.loc[bhas_df["Bhasmarathi Type"].astype(str)==str(bhas_type), "Description"]
                if not m.empty: bhas_desc = str(m.iloc[0])

            # text
            dates_series = pd.to_datetime(df_rows["Date"], errors="coerce")
            total_days_calc = len(df_rows)
            total_nights = max(total_days_calc-1, 0)
            night_txt  = "Night" if total_nights==1 else "Nights"
            person_txt = "Person" if total_pax==1 else "Persons"

            greet = f"Greetings from TravelAajkal,\n\n*Client Name: {client_name}*\n\n"
            plan  = f"*Plan:- {total_days_calc}Days and {total_nights}{night_txt} {final_route} for {total_pax} {person_txt}*"
            grouped = {}
            for _, rr in df_rows.iterrows():
                dstr = pd.to_datetime(rr["Date"]).strftime("%d-%b-%Y")
                tp = f"{(rr.get('Time') or '')}: " if str(rr.get('Time') or '').strip() else ""
                grouped.setdefault(dstr, []).append(f"{tp}{_code_to_desc(rr.get('Code'))}")

            itinerary_text = greet + plan + "\n\n*Itinerary:*\n"
            for i, (d, evs) in enumerate(grouped.items(), 1):
                itinerary_text += f"\n*Day{i}:{d}*\n" + "\n".join(evs) + "\n"
            bits = [x for x in [car_types or None, hotel_types or None, bhas_desc or None] if x]
            detail_line = "(" + ",".join(bits) + ")" if bits else ""
            itinerary_text += f"\n*Package cost: ₹{in_locale(total_package)}/-*\n"
            if has_ref:
                itinerary_text += f"*Package cost (after referral 10%): ₹{in_locale(after_ref)}/-*\n"
            itinerary_text += detail_line

            # serialize rows
            rows_serialized = df_rows.copy()
            rows_serialized["Date"] = pd.to_datetime(rows_serialized["Date"]).dt.strftime("%Y-%m-%d")

            # build record (rev starts from 1)
            client_mobile = "".join(ch for ch in client_mobile_raw if ch.isdigit())
            record = {
                "client_name": client_name,
                "client_mobile": client_mobile,
                "representative": rep,
                "upload_date": dt.datetime.utcnow(),
                "start_date": str(start_date),
                "end_date": str((start_date + dt.timedelta(days=int(days)-1))),
                "total_days": int(days),
                "total_pax": int(total_pax),
                "final_route": final_route,
                "car_types": car_types,
                "hotel_types": hotel_types,
                "bhasmarathi_required": (bhas_required=="Yes"),
                "bhasmarathi_type": bhas_type if bhas_required=="Yes" else None,
                "bhasmarathi_persons": int(bhas_persons) if bhas_required=="Yes" else 0,
                "bhasmarathi_unit_pkg": int(unit_pkg) if bhas_required=="Yes" else 0,
                "bhasmarathi_unit_actual": int(unit_act) if bhas_required=="Yes" else 0,
                "bhasmarathi_pkg_total": int(bh_pkg),
                "bhasmarathi_actual_total": int(bh_act),
                "package_total": int(total_package),
                "package_after_referral": int(after_ref),
                "actual_total": int(total_actual),
                "profit_total": int(profit_total),
                "referred_by": referred_sel if has_ref else None,
                "referral_discount_pct": 10 if has_ref else 0,
                "rows": rows_serialized.to_dict(orient="records"),
                "package_cost": int(total_package),
                "bhasmarathi_types": bhas_desc,
                "itinerary_text": itinerary_text,
                "revision_num": 1,                # first save is rev 1
                "is_revision": True,
                "revision_of": None,
                "revision_notes": "initial save",
            }

            try:
                # if exactly same key already exists, next revision = max+1
                existing = list(col_it.find(
                    {"client_mobile": client_mobile, "start_date": str(start_date)},
                    {"revision_num":1}
                ))
                if existing:
                    mx = max(_safe_int(e.get("revision_num"), 0) for e in existing)
                    record["revision_num"] = int(mx) + 1

                col_it.insert_one(record)
                st.success(f"Saved package (rev {record['revision_num']}).")
                st.session_state["last_itinerary_text"] = itinerary_text
            except Exception as e:
                st.error(f"Save failed: {e}")

    # preview / download if generated in this session
    if txt := st.session_state.get("last_itinerary_text"):
        st.divider()
        st.markdown("### Preview")
        c1, c2 = st.columns(2)
        c1.text_area("Itinerary (copy)", txt, height=360)
        c2.download_button(
            "⬇️ Download itinerary (.txt)",
            data=txt,
            file_name="itinerary.txt",
            mime="text/plain",
            use_container_width=True
        )

# =========================================================
#                     SEARCH MODE
# =========================================================
else:
    st.markdown("### Search by client or mobile")
    q = st.text_input("Type to search (suggestions appear from first letter/number)")
    suggestions = _suggest(q.strip())
    pick = st.selectbox("Suggestions", ["--"] + suggestions, index=0)
    loaded_doc = None
    if pick != "--":
        # parse selection
        if "—" in pick:
            nm, mb = [p.strip() for p in pick.split("—", 1)]
        else:
            nm, mb = (pick, "")
        # fetch all packages for this mobile
        docs = list(col_it.find({"client_mobile": mb}, {"itinerary_text":0}).sort([("start_date",-1),("revision_num",-1)]))
        if docs:
            labels = [f"{d.get('client_name','')} — {d.get('client_mobile','')} • start:{d.get('start_date','?')} • rev:{int(d.get('revision_num',0))}" for d in docs]
            idx = st.selectbox("Pick a package", list(range(len(docs))), format_func=lambda i: labels[i])
            if st.button("Load this package"):
                loaded_doc = docs[idx]

    if loaded_doc:
        st.success("Package loaded. You can edit and press **Update itinerary** to save a new revision.")
        # fill header
        st.markdown("### Client & trip")
        c1,c2,c3,c4 = st.columns([1.6,1,1,1])
        client_name = c1.text_input("Client name*", value=loaded_doc.get("client_name",""))
        client_mobile_raw = c2.text_input("Client mobile (10 digits)*", value=loaded_doc.get("client_mobile",""))
        rep = c3.selectbox("Representative*", ["-- Select --","Arpith","Reena","Kuldeep","Teena"], index=0)
        # select default to loaded rep
        if loaded_doc.get("representative") in ["Arpith","Reena","Kuldeep","Teena"]:
            rep = c3.selectbox("Representative*", ["-- Select --","Arpith","Reena","Kuldeep","Teena"], index=["-- Select --","Arpith","Reena","Kuldeep","Teena"].index(loaded_doc.get("representative")))
        total_pax = c4.number_input("Total Pax*", min_value=1, step=1, value=_safe_int(loaded_doc.get("total_pax",2)))

        h1,h2 = st.columns(2)
        try:
            sdt = dt.date.fromisoformat(str(loaded_doc.get("start_date")))
        except Exception:
            sdt = dt.date.today()
        start_date = h1.date_input("Start date", value=sdt)
        days = h2.number_input("No. of days", min_value=1, step=1, value=_safe_int(loaded_doc.get("total_days",2)))

        # bhas section
        st.markdown("### Bhasmarathi")
        b_req = "Yes" if loaded_doc.get("bhasmarathi_required") else "No"
        b1,b2,b3 = st.columns(3)
        bhas_required = b1.selectbox("Bhasmarathi required?", ["No","Yes"], index=1 if b_req=="Yes" else 0)
        bhas_type = b2.selectbox("Bhasmarathi Type", ["V-BH","P-BH","BH"], index=["V-BH","P-BH","BH"].index(loaded_doc.get("bhasmarathi_type","V-BH")))
        bhas_persons = b3.number_input("Persons for Bhasmarathi", min_value=0, step=1, value=_safe_int(loaded_doc.get("bhasmarathi_persons",0)), disabled=(bhas_required=="No"))
        c_pkg, c_act = st.columns(2)
        unit_pkg = c_pkg.number_input("Unit Package Cost (₹) — Bhasmarathi", min_value=0, step=100, value=_safe_int(loaded_doc.get("bhasmarathi_unit_pkg",0)), disabled=(bhas_required=="No"))
        unit_act = c_act.number_input("Unit Actual Cost (₹) — Bhasmarathi",  min_value=0, step=100, value=_safe_int(loaded_doc.get("bhasmarathi_unit_actual",0)), disabled=(bhas_required=="No"))

        # referral (just display loaded)
        referred_sel = loaded_doc.get("referred_by") or "-- None --"
        st.selectbox("Referred By (applies 10% discount)", [referred_sel], index=0, disabled=True)

        # table prefill (single-pass, no vanishing)
        st.markdown("### Fill line items")
        st.session_state["_active_token"] = None  # force re-seed once for this loaded doc
        st.session_state["form_rows"] = _rows_from_doc(loaded_doc, start_date)

        col_cfg = {
            "Date": st.column_config.DateColumn("Date", disabled=True),
            "Time": st.column_config.SelectboxColumn("Time", options=time_options),
            "Code": st.column_config.SelectboxColumn("Code", options=code_options),
            "Car Type": st.column_config.SelectboxColumn("Car Type", options=car_options),
            "Hotel Type": st.column_config.SelectboxColumn("Hotel Type", options=hotel_options),
            "Stay City": st.column_config.SelectboxColumn("Stay City", options=stay_city_options),
            "Room Type": st.column_config.SelectboxColumn("Room Type", options=room_options),
            "Pkg-Car Cost": st.column_config.NumberColumn("Pkg-Car Cost", min_value=0.0, step=100.0),
            "Pkg-Hotel Cost": st.column_config.NumberColumn("Pkg-Hotel Cost", min_value=0.0, step=100.0),
            "Act-Car Cost": st.column_config.NumberColumn("Act-Car Cost", min_value=0.0, step=100.0),
            "Act-Hotel Cost": st.column_config.NumberColumn("Act-Hotel Cost", min_value=0.0, step=100.0),
        }

        edited = st.data_editor(
            st.session_state.form_rows,
            key="editor_rows_loaded",
            use_container_width=True,
            num_rows="fixed",
            hide_index=True,
            column_config=col_cfg,
        )
        if not edited.equals(st.session_state.form_rows):
            st.session_state.form_rows = edited

        if st.button("🔁 Update itinerary (save as next revision)", type="primary"):
            df_rows = st.session_state.form_rows.copy()
            # compute totals & build record very similar to create-mode
            pkg_car   = pd.to_numeric(df_rows["Pkg-Car Cost"], errors="coerce").fillna(0).sum()
            pkg_hotel = pd.to_numeric(df_rows["Pkg-Hotel Cost"], errors="coerce").fillna(0).sum()
            act_car   = pd.to_numeric(df_rows["Act-Car Cost"], errors="coerce").fillna(0).sum()
            act_hotel = pd.to_numeric(df_rows["Act-Hotel Cost"], errors="coerce").fillna(0).sum()
            bh_pkg    = (unit_pkg * bhas_persons) if bhas_required=="Yes" else 0
            bh_act    = (unit_act * bhas_persons) if bhas_required=="Yes" else 0

            package_cost_rows = float(pkg_car + pkg_hotel)
            actual_cost_rows  = float(act_car + act_hotel)
            total_package     = ceil_to_999(package_cost_rows + bh_pkg)
            total_actual      = actual_cost_rows + bh_act
            profit_total      = int(total_package - total_actual)

            car_types = "-".join(pd.Series(df_rows["Car Type"]).dropna().astype(str).replace("","").unique().tolist()).strip("-")
            hotel_types = "-".join(pd.Series(df_rows["Hotel Type"]).dropna().astype(str).replace("","").unique().tolist()).strip("-")
            route_parts = [r for r in (_code_to_route(c) for c in df_rows["Code"]) if r]
            route_raw = "-".join(route_parts).replace(" -","-").replace("- ","-")
            route_list = [x for x in route_raw.split("-") if x]
            final_route = "-".join([route_list[i] for i in range(len(route_list)) if i==0 or route_list[i]!=route_list[i-1]])

            # serialize
            rows_serialized = df_rows.copy()
            rows_serialized["Date"] = pd.to_datetime(rows_serialized["Date"]).dt.strftime("%Y-%m-%d")

            client_mobile = "".join(ch for ch in client_mobile_raw if ch.isdigit())

            rec = {
                "client_name": client_name,
                "client_mobile": client_mobile,
                "representative": rep,
                "upload_date": dt.datetime.utcnow(),
                "start_date": str(start_date),
                "end_date": str((start_date + dt.timedelta(days=int(days)-1))),
                "total_days": int(days),
                "total_pax": int(total_pax),
                "final_route": final_route,
                "car_types": car_types,
                "hotel_types": hotel_types,
                "bhasmarathi_required": (bhas_required=="Yes"),
                "bhasmarathi_type": bhas_type if bhas_required=="Yes" else None,
                "bhasmarathi_persons": int(bhas_persons) if bhas_required=="Yes" else 0,
                "bhasmarathi_unit_pkg": int(unit_pkg) if bhas_required=="Yes" else 0,
                "bhasmarathi_unit_actual": int(unit_act) if bhas_required=="Yes" else 0,
                "bhasmarathi_pkg_total": int(bh_pkg),
                "bhasmarathi_actual_total": int(bh_act),
                "package_total": int(total_package),
                "actual_total": int(total_actual),
                "profit_total": int(profit_total),
                "rows": rows_serialized.to_dict(orient="records"),
                "package_cost": int(total_package),
                "itinerary_text": loaded_doc.get("itinerary_text",""),  # optional reuse
                "is_revision": True,
            }
            # next revision #
            try:
                existing = list(col_it.find({"client_mobile": client_mobile, "start_date": str(start_date)}, {"revision_num":1}))
                mx = max(_safe_int(e.get("revision_num"), 0) for e in existing) if existing else 0
                rec["revision_num"] = int(mx) + 1
                col_it.insert_one(rec)
                st.success(f"Saved new revision (rev {rec['revision_num']}).")
            except Exception as e:
                st.error(f"Save failed: {e}")
