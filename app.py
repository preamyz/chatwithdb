# app.py
import streamlit as st
import pandas as pd
import sqlite3
import io

import google.generativeai as genai
from dsyp_core import call_router_llm, build_params_for_template

def format_answer(template_key: str, df: pd.DataFrame) -> str:
    """
    แปลงผลลัพธ์ DataFrame -> ข้อความตอบแบบคน
    ปรับ mapping ตาม template_key ของคุณ
    """
    if df is None or df.empty:
        return "ไม่พบข้อมูลจากคำถามนี้"

    row = df.iloc[0].to_dict()

    # ✅ ตัวอย่าง mapping (คุณเพิ่มได้เรื่อยๆ)
    if template_key in ["SALES_TOTAL_CURR", "SALES_TOTAL_CURR_VS_PREV", "SALES_TOTAL_CURR_VS_PREV2"]:
        # กรณี template นี้คืนค่าเป็น total_value
        if "total_value" in row:
            return f"ยอดขายเดือนนี้ {row['total_value']:,.0f} บาท"
        # กรณีคุณทำอีก template ที่คืน contract_cnt
        if "contract_cnt" in row:
            return f"ยอดขายเดือนนี้ {row['contract_cnt']:,.0f} สัญญา"

    if template_key in ["CREDIT_CONTRACT_CNT"]:
        if "contract_cnt" in row:
            return f"เดือนนี้มีสัญญาทั้งหมด {row['contract_cnt']:,.0f} สัญญา"

    if template_key in ["SALES_CONTRACT_CNT_VS_PREV", "CREDIT_CONTRACT_CNT_VS_PREV"]:
        # คาดว่าคืน cur_cnt / prev_cnt / diff_cnt / diff_pct
        cur_cnt = row.get("cur_cnt")
        prev_cnt = row.get("prev_cnt")
        diff_cnt = row.get("diff_cnt")
        diff_pct = row.get("diff_pct")

        if diff_pct is not None and diff_cnt is not None:
            direction = "เพิ่มขึ้น" if diff_cnt >= 0 else "ลดลง"
            return f"ขาย{direction}เทียบกับเดือนที่แล้ว {abs(diff_pct):.0f}% หรือ {abs(diff_cnt):,.0f} สัญญา"

    # fallback (ถ้าไม่เข้า mapping)
    # เอาคอลัมน์แรกมาโชว์แบบง่าย
    first_col = df.columns[0]
    val = df.iloc[0][first_col]
    if isinstance(val, (int, float)):
        return f"{first_col}: {val:,.0f}"
    return f"{first_col}: {val}"


APP_VERSION = "v2025-12-31-clean4"


def load_csv_to_sqlite(conn, table_name: str, file_bytes: bytes, if_exists: str = "replace"):
    """โหลด CSV (bytes) -> pandas -> write ลง SQLite เป็น table_name"""
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(file_bytes), encoding="cp874")

    df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    conn.commit()
    return df.shape


def sqlite_schema_doc(conn) -> str:
    q = """
    SELECT name
    FROM sqlite_master
    WHERE type='table' AND name NOT LIKE 'sqlite_%'
    ORDER BY name;
    """
    tables = [r[0] for r in conn.execute(q).fetchall()]

    lines = []
    for t in tables:
        cols = conn.execute(f"PRAGMA table_info('{t}')").fetchall()
        col_lines = [f"- {c[1]} ({c[2]})" for c in cols]
        lines.append(f"TABLE: {t}\n" + "\n".join(col_lines))

    return "\n\n".join(lines)


def read_xlsx(uploaded) -> pd.DataFrame:
    # อ่าน sheet แรกเสมอ เพื่อให้ได้ DataFrame ไม่ใช่ dict
    return pd.read_excel(uploaded, sheet_name=0)


# ---------------------------
# UI
# ---------------------------

# =========================
# 4) UI (Streamlit Community Cloud ready)
#   - No file upload for BU
#   - Load data/question_bank/templates from /assets
#   - API key from Streamlit Secrets
# =========================

st.set_page_config(page_title="DSYP Chat-to-SQL (Gemini + SQLite)", layout="wide")
st.title("DSYP Chat-to-SQL (Gemini + SQLite)")
st.sidebar.caption(f"APP_VERSION: {APP_VERSION}")

# --- Paths in repo ---
ASSETS_DIR = Path(__file__).parent / "assets"
SALES_CSV_PATH  = ASSETS_DIR / "sales_master_enhanced_2024_2025.csv"
CREDIT_CSV_PATH = ASSETS_DIR / "credit_contract_enhanced_2024_2025.csv"
QB_XLSX_PATH    = ASSETS_DIR / "question_bank.xlsx"
TPL_XLSX_PATH   = ASSETS_DIR / "sql_templates_with_placeholder.xlsx"

# --- DB ---
DB_PATH = "/tmp/app.db"
if "conn" not in st.session_state:
    st.session_state.conn = sqlite3.connect(DB_PATH, check_same_thread=False)

def load_assets_to_sqlite(conn):
    df_sales = pd.read_csv(SALES_CSV_PATH)
    df_sales.to_sql("SALES_MASTER", conn, if_exists="replace", index=False)

    df_credit = pd.read_csv(CREDIT_CSV_PATH)
    df_credit.to_sql("CREDIT_CONTRACT", conn, if_exists="replace", index=False)

    conn.commit()
    return df_sales.shape, df_credit.shape

def ensure_loaded_assets(conn):
    """Load assets only once per session by default."""
    if st.session_state.get("db_loaded"):
        return
    if not (SALES_CSV_PATH.exists() and CREDIT_CSV_PATH.exists()):
        st.error("Missing CSV files in /assets (SALES/CREDIT). Please commit them to GitHub.")
        st.stop()
    (rs, cs), (rc, cc) = load_assets_to_sqlite(conn)
    st.session_state.db_loaded = True
    st.session_state.load_status = f"Loaded SALES_MASTER ({rs} rows) and CREDIT_CONTRACT ({rc} rows)"

def load_artifacts_from_assets():
    if not (QB_XLSX_PATH.exists() and TPL_XLSX_PATH.exists()):
        st.error("Missing question_bank.xlsx or sql_templates_with_placeholder.xlsx in /assets.")
        st.stop()
    qb = pd.read_excel(QB_XLSX_PATH)
    tpl = pd.read_excel(TPL_XLSX_PATH)
    return qb, tpl

# --- Settings (Secrets first) ---
with st.sidebar:
    st.header("Settings")

    api_key = st.secrets.get("GEMINI_API_KEY", "")
    model_name = st.secrets.get("GEMINI_MODEL", "gemini-2.0-flash")

    # Optional fallback for local dev only
    if not api_key:
        st.warning("No GEMINI_API_KEY in Streamlit Secrets. (Local dev only) Please input API key below.")
        api_key = st.text_input("Google Gemini API Key", type="password")

    model_name = st.text_input("Model name", value=model_name)

    st.divider()
    if st.button("Reload data from assets"):
        try:
            (rs, cs), (rc, cc) = load_assets_to_sqlite(st.session_state.conn)
            st.session_state.db_loaded = True
            st.session_state.load_status = f"Reloaded SALES_MASTER ({rs} rows) and CREDIT_CONTRACT ({rc} rows)"
            st.success(st.session_state.load_status)
        except Exception as e:
            st.error(f"Reload failed: {e}")

# --- Auto load data + artifacts ---
ensure_loaded_assets(st.session_state.conn)
question_bank_df, templates_df = load_artifacts_from_assets()

# --- Schema doc ---
if "schema_doc" not in st.session_state:
    st.session_state.schema_doc = ""

colA, colB = st.columns([1, 2])
with colA:
    if st.button("Generate schema from SQLite"):
        st.session_state.schema_doc = sqlite_schema_doc(st.session_state.conn)
with colB:
    st.caption(st.session_state.get("load_status", "Data ready from /assets"))

schema_doc = st.text_area("Schema doc", value=st.session_state.schema_doc, height=180)

st.divider()

# --- Report / Data Type selector ---
st.subheader("Report / Data Type")

DATA_SCOPE_OPTIONS = ["All", "Sales", "Credit", "Marketing", "Accounting", "Risk"]

data_scope = st.selectbox(
    "Choose data scope",
    DATA_SCOPE_OPTIONS,
    index=0,
    help="All = ใช้คำถามทุกโดเมน (ไม่จำกัดข้อมูล). เลือกโดเมนอื่นเพื่อจำกัดคำถาม/เทมเพลตเฉพาะหมวดนั้น"
)

selected_domain = None
if data_scope == "Sales":
    selected_domain = "sales"
elif data_scope == "Credit":
    selected_domain = "credit"
elif data_scope == "Marketing":
    selected_domain = "marketing"
elif data_scope == "Accounting":
    selected_domain = "accounting"
elif data_scope == "Risk":
    selected_domain = "risk"
else:
    selected_domain = None  # All = no filter

def guess_domain_rule(q: str) -> str:
    """Heuristic domain suggestion (used only for UI hint when All mode)."""
    q = (q or "").lower()

    kw = {
        "credit": ["เครดิต", "อนุมัติ", "reject", "ปฏิเสธ", "leadtime", "npl", "dpd", "loan", "bureau", "underwriting", "cancellation", "ยกเลิก", "สัญญา", "สินเชื่อ"],
        "sales": ["ยอดขาย", "แคมเปญ", "campaign", "สินค้า", "product", "sales", "รุ่น", "model", "dealer", "สาขา", "ยอด", "ราคา"],
        "marketing": ["การตลาด", "marketing", "campaign roi", "segment", "segmentation", "conversion", "lead", "funnel", "ctr", "cpc"],
        "accounting": ["บัญชี", "accounting", "gl", "ledger", "invoice", "ap", "ar", "vat", "ภาษี", "กำไรขาดทุน", "งบการเงิน"],
        "risk": ["ความเสี่ยง", "risk", "pd", "lgd", "ead", "default", "early warning", "delinquent", "ผิดนัด", "kri", "kpi risk"],
    }

    for d, words in kw.items():
        if any(w in q for w in words):
            return d
    return "sales"


user_question = st.text_input("Ask a question", value="ยอดขายเดือนนี้เท่าไร")
run_btn = st.button("Run", type="primary")

# Hint only (does not filter) when All mode is selected
if data_scope == "All":
    hint_domain = guess_domain_rule(user_question)
    st.caption(f"All mode: system will search across all domains. (Hint: looks like **{hint_domain}**)")

if run_btn:
    if not api_key:
        st.error("Missing Gemini API key (set GEMINI_API_KEY in Streamlit Secrets).")
        st.stop()
    if question_bank_df is None or templates_df is None:
        st.error("Missing artifacts in /assets.")
        st.stop()
    if not schema_doc.strip():
        st.error("กรุณา Generate schema หรือใส่ schema_doc ก่อน")
        st.stop()

    try:
        genai.configure(api_key=api_key)

        qb_use = question_bank_df.copy()
        if selected_domain:
            qb_use = qb_use[qb_use["domain"].astype(str).str.lower() == selected_domain].copy()

        router_out = call_router_llm(
            user_question=user_question,
            question_bank_df=qb_use,
            schema_doc=schema_doc,
            model_name=model_name,
        )

        template_key = router_out.get("sql_template_key", "")

        allowed = set(qb_use["sql_template_key"].dropna().astype(str).unique().tolist())
        if template_key not in allowed:
            st.error("คำถามนี้อยู่นอกขอบเขต question_bank ของ domain ที่เลือก (ถูก block เพื่อป้องกัน hallucination)")
            st.json({"router_out": router_out, "domain": selected_domain, "allowed_template_keys": sorted(list(allowed))})
            st.stop()

        final_sql, params = build_params_for_template(
            router_out=router_out,
            question_bank_df=qb_use,
            templates_df=templates_df,
        )

        final_sql = (
            final_sql
            .replace("—", "--")
            .replace("≥", ">=")
            .replace("≤", "<=")
        )

        st.session_state["last_user_question"] = user_question

        # display vs execute
        display_sql = final_sql
        sql_exec = strip_sql_comments(final_sql)
        sql_exec = override_sql_dates_by_question(sql_exec, template_key, user_question)

        ok, msg = is_safe_readonly_sql(sql_exec, st.session_state.conn)
        if not ok:
            st.error(f"SQL ถูก block: {msg}")
            with st.expander("ดู SQL ที่ถูก block (optional)"):
                st.code(display_sql, language="sql")
            st.stop()

        df = pd.read_sql_query(sql_exec, st.session_state.conn)

        answer_text = hybrid_answer(
            model_name=model_name,
            user_question=user_question,
            template_key=template_key,
            df=df,
            question_bank_df=qb_use,
        )

        st.markdown(f"**Domain:** {selected_domain}")
        st.markdown(f"**คำถาม:** {user_question}")
        st.markdown(f"**คำตอบ:** {answer_text}")

        st.divider()

        meta = {"rows": int(df.shape[0]), "columns": df.columns.tolist()}

        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Router output")
            st.json(router_out)

        with st.expander("ดู SQL ที่รัน (optional)"):
            st.code(display_sql, language="sql")

        with c2:
            st.subheader("Result")
            st.write(meta)
            if df.empty:
                st.warning("ไม่พบข้อมูลจากคำถามนี้")
            else:
                st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(str(e))
