# app.py
import streamlit as st
import pandas as pd
import sqlite3
import io

import google.generativeai as genai
from dsyp_core import call_router_llm, build_params_for_template


APP_VERSION = "v2025-12-31-clean2"


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
st.set_page_config(page_title="DSYP Chat-to-SQL (Gemini + SQLite)", layout="wide")
st.title("DSYP Chat-to-SQL (Gemini + SQLite)")
st.sidebar.caption(f"APP_VERSION: {APP_VERSION}")
st.caption("อัปโหลดไฟล์ → ตั้งค่า API key → ถามคำถาม → ได้ SQL + ผลลัพธ์ตาราง")

# ---------------------------
# Sidebar: Settings
# ---------------------------
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Google Gemini API Key", type="password")
    model_name = st.text_input("Model name", value="gemini-2.0-flash")

    st.divider()
    st.subheader("Upload config")
    table_name = st.text_input("CSV table name in SQLite", value="SALES_MASTER")

# ---------------------------
# Main: Upload files
# ---------------------------
col1, col2, col3 = st.columns(3)
with col1:
    qb_file = st.file_uploader("Upload question_bank.xlsx", type=["xlsx"])
with col2:
    tpl_file = st.file_uploader("Upload sql_templates_with_placeholder.xlsx", type=["xlsx"])
with col3:
    csv_file = st.file_uploader("Upload data CSV", type=["csv"])

# ---------------------------
# Initialize DB once per session
# ---------------------------
DB_PATH = "/tmp/app.db"

if "conn" not in st.session_state:
    st.session_state.conn = sqlite3.connect(DB_PATH, check_same_thread=False)

# ---------------------------
# Load Excel files
# ---------------------------
question_bank_df = read_xlsx(qb_file) if qb_file is not None else None
templates_df = read_xlsx(tpl_file) if tpl_file is not None else None

# ---------------------------
# Load CSV into sqlite
# ---------------------------
if csv_file is not None:
    try:
        rows, cols = load_csv_to_sqlite(
            st.session_state.conn,
            table_name=table_name,
            file_bytes=csv_file.getvalue(),
        )
        st.success(f"Loaded CSV into SQLite table: {table_name} ({rows} rows, {cols} cols)")
        st.info(f"หมายเหตุ: SQL template ต้องอ้าง table ชื่อเดียวกัน เช่น FROM {table_name}")
    except Exception as e:
        st.error(f"Load CSV failed: {e}")

st.divider()

# ---------------------------
# Schema doc (auto-generate)
# ---------------------------
if "schema_doc" not in st.session_state:
    st.session_state.schema_doc = ""

cA, cB = st.columns([1, 1])
with cA:
    if st.button("Generate schema from SQLite"):
        st.session_state.schema_doc = sqlite_schema_doc(st.session_state.conn)
with cB:
    st.caption("กดปุ่มเพื่อดึง schema จาก SQLite (ตารางที่โหลดจาก CSV)")

schema_doc = st.text_area("Schema doc", value=st.session_state.schema_doc, height=180)

st.divider()

# ---------------------------
# Ask question
# ---------------------------
user_question = st.text_input("Ask a question", value="ยอดขายเดือนนี้เท่าไร")
run_btn = st.button("Run", type="primary")

if run_btn:
    if not api_key:
        st.error("กรุณาใส่ Gemini API key ก่อน")
        st.stop()
    if question_bank_df is None or templates_df is None:
        st.error("กรุณาอัปโหลด question_bank.xlsx และ sql_templates_with_placeholder.xlsx")
        st.stop()
    if not schema_doc.strip():
        st.error("กรุณา Generate schema หรือใส่ schema_doc ก่อน")
        st.stop()

    try:
        # 1) init gemini (configure key)
        genai.configure(api_key=api_key)

        # 2) router เลือก template_key
        router_out = call_router_llm(
            user_question=user_question,
            question_bank_df=question_bank_df,
            schema_doc=schema_doc,
            model_name=model_name,
        )

        # 3) render SQL จาก template + params
        final_sql, params = build_params_for_template(
            router_out=router_out,
            question_bank_df=question_bank_df,
            templates_df=templates_df,
        )

        # DEBUG
        st.subheader("DEBUG: build_params_for_template output")
        st.write("Type(final_sql):", type(final_sql))
        st.write("final_sql:", final_sql)

        # Guard: final_sql ต้องเป็น string
        if not isinstance(final_sql, str):
            st.error("❌ Generated SQL is not a string (cannot execute)")
            st.stop()

        # 4) run sql
        df = pd.read_sql_query(final_sql, st.session_state.conn)

        # Guard: df ต้องเป็น DataFrame
        if not isinstance(df, pd.DataFrame):
            st.error("❌ Query result is not a DataFrame")
            st.write("Type(df):", type(df))
            st.write(df)
            st.stop()

        meta = {"rows": int(df.shape[0]), "columns": df.columns.tolist()}

        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Router output")
            st.json(router_out)

            st.subheader("Rendered SQL")
            st.code(final_sql, language="sql")

        with c2:
            st.subheader("Result")
            st.write(meta)
            if df.empty:
                st.warning("ไม่พบข้อมูลจากคำถามนี้")
            else:
                st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(str(e))
