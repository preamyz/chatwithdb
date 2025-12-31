# app.py (LLM + Rule Hybrid Answer, anti-hallucination)
import streamlit as st
import pandas as pd
import sqlite3
import io
import re
import json
from typing import Dict, Any, Optional, Tuple, List

import google.generativeai as genai
from dsyp_core import call_router_llm, build_params_for_template


# =========================
# 1) Utilities: Safe SQL
# =========================
DANGEROUS_SQL_TOKENS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|VACUUM|PRAGMA)\b",
    re.IGNORECASE
)

def strip_sql_comments(sql: str) -> str:
    # remove -- comments
    sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    # remove /* */ comments
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql.strip()

def extract_table_names(sql: str) -> List[str]:
    """
    Best-effort table extraction for SQLite (FROM/JOIN).
    NOTE: May include CTE aliases (e.g., FROM cur). We will filter those later.
    """
    cleaned = strip_sql_comments(sql)
    cleaned = re.sub(r"\s+", " ", cleaned)
    hits = re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)\b", cleaned, flags=re.IGNORECASE)

    seen = set()
    out = []
    for h in hits:
        if h not in seen:
            out.append(h)
            seen.add(h)
    return out

def extract_cte_names(sql: str) -> List[str]:
    """
    Extract CTE names from WITH clause.
    Example:
      WITH cur AS (...), prev AS (...)
      SELECT ... FROM cur JOIN prev ...
    Return: ["cur", "prev"]
    """
    cleaned = strip_sql_comments(sql)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Must start with WITH to have CTEs
    if not re.match(r"^WITH\b", cleaned, flags=re.IGNORECASE):
        return []

    # Grab text after WITH and before main SELECT (best-effort)
    m = re.match(r"^WITH\s+(.*)\s+SELECT\b", cleaned, flags=re.IGNORECASE)
    if not m:
        return []

    cte_part = m.group(1)

    # Split CTE definitions safely at commas that begin a new "<name> AS ("
    parts = re.split(r",\s*(?=[A-Za-z_][A-Za-z0-9_]*\s+AS\s*\()", cte_part, flags=re.IGNORECASE)

    names = []
    for p in parts:
        m2 = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(", p, flags=re.IGNORECASE)
        if m2:
            names.append(m2.group(1))
    return names

def existing_tables(conn) -> set:
    q = """
    SELECT name
    FROM sqlite_master
    WHERE type='table' AND name NOT LIKE 'sqlite_%';
    """
    return {r[0] for r in conn.execute(q).fetchall()}

def is_safe_readonly_sql(sql: str, conn) -> Tuple[bool, str]:
    """
    Guardrail: allow only SELECT/WITH queries; block multi-statement & DDL/DML; ensure tables exist.
    Fix: ignore CTE aliases when validating table existence.
    """
    if not isinstance(sql, str) or not sql.strip():
        return False, "SQL ว่าง"

    s = sql.strip()

    # Block multiple statements (very strict)
    # Allow semicolons only at the very end (optional)
    semis = [m.start() for m in re.finditer(";", s)]
    if len(semis) > 1 or (len(semis) == 1 and semis[0] != len(s) - 1):
        return False, "SQL มีหลาย statement (ถูก block เพื่อความปลอดภัย)"

    core = strip_sql_comments(s)
    if not re.match(r"^(SELECT|WITH)\b", core, flags=re.IGNORECASE):
        return False, "อนุญาตเฉพาะ SELECT/WITH เท่านั้น"

    if DANGEROUS_SQL_TOKENS.search(core):
        return False, "ตรวจพบคำสั่งที่ไม่ปลอดภัย (DDL/DML) จึงถูก block"

    # Ensure referenced tables exist (ignore CTE alias names)
    tables = extract_table_names(core)
    cte_names = set(extract_cte_names(core))  # ✅ new
    if tables:
        exist = existing_tables(conn)
        missing = [t for t in tables if (t not in exist) and (t not in cte_names)]
        if missing:
            return False, f"SQL อ้างอิงตารางที่ไม่มีใน DB: {missing}"

    return True, "OK"


# =========================
# 2) Answer layer (Hybrid)
# =========================
def _fmt_money(x) -> str:
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return str(x)

def _fmt_num(x) -> str:
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return str(x)

def _fmt_pct(x) -> str:
    try:
        return f"{float(x):,.1f}%"
    except Exception:
        return str(x)

def _first_existing(row: Dict[str, Any], keys: List[str]):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None

def rule_based_answer(template_key: str, df: pd.DataFrame, qb_row: Optional[pd.Series] = None) -> Optional[str]:
    """
    Rule-based summary for known templates. Return None if cannot confidently format.
    """
    if df is None or df.empty:
        return "ไม่พบข้อมูลจากคำถามนี้"

    # Most single-row metrics
    if df.shape[0] == 1:
        row = df.iloc[0].to_dict()

        # SALES_TOTAL_CURR
        if template_key == "SALES_TOTAL_CURR":
            v = _first_existing(row, ["total_value", "total_sales", "sales_value", "sum_value"])
            if v is not None:
                return f"ยอดขายเดือนนี้ {_fmt_money(v)} บาท"

        # SALES_TOTAL_CURR_VS_PREV
        if template_key == "SALES_TOTAL_CURR_VS_PREV":
            cur = _first_existing(row, ["cur_value", "cur_total", "cur_sales", "cur"])
            prev = _first_existing(row, ["prev_value", "prev_total", "prev_sales", "prev"])
            diff = _first_existing(row, ["diff_value", "diff", "delta_value"])
            diff_pct = _first_existing(row, ["diff_pct", "pct_change", "delta_pct"])
            if cur is not None and prev is not None:
                direction = None
                try:
                    d = float(cur) - float(prev)
                    direction = "เพิ่มขึ้น" if d >= 0 else "ลดลง"
                except Exception:
                    direction = "เปลี่ยนแปลง"
                parts = [f"ยอดขายเดือนนี้ {_fmt_money(cur)} บาท (เดือนที่แล้ว {_fmt_money(prev)} บาท)"]
                if diff is not None:
                    parts.append(f"{direction} {_fmt_money(abs(diff))} บาท")
                if diff_pct is not None:
                    parts.append(f"({direction} {_fmt_pct(abs(diff_pct))})")
                return " ".join(parts)

        # CREDIT_CONTRACT_CNT
        if template_key == "CREDIT_CONTRACT_CNT":
            v = _first_existing(row, ["contract_cnt", "cnt", "total_cnt"])
            if v is not None:
                return f"เดือนนี้มีสัญญาทั้งหมด {_fmt_num(v)} สัญญา"

        # CREDIT_APPROVAL_RATE_VS_PREV / CREDIT_CANCELLATION_RATE_VS_PREV
        if template_key in ["CREDIT_APPROVAL_RATE_VS_PREV", "CREDIT_CANCELLATION_RATE_VS_PREV"]:
            cur = _first_existing(row, ["cur_rate", "cur_pct", "cur_value", "cur"])
            prev = _first_existing(row, ["prev_rate", "prev_pct", "prev_value", "prev"])
            diff = _first_existing(row, ["diff_pp", "diff", "delta_pp", "delta"])
            if cur is not None and prev is not None:
                try:
                    d = float(cur) - float(prev)
                    direction = "เพิ่มขึ้น" if d >= 0 else "ลดลง"
                except Exception:
                    direction = "เปลี่ยนแปลง"
                msg = f"อัตรา{('อนุมัติ' if template_key=='CREDIT_APPROVAL_RATE_VS_PREV' else 'ยกเลิก')}เดือนนี้ {_fmt_pct(cur)} (เดือนที่แล้ว {_fmt_pct(prev)})"
                if diff is not None:
                    msg += f" {direction} {abs(float(diff)):.1f} จุด"
                return msg

        # CREDIT_LEADTIME_AVG
        if template_key == "CREDIT_LEADTIME_AVG":
            v = _first_existing(row, ["avg_leadtime", "leadtime_avg", "avg_days", "avg_hour"])
            if v is not None:
                # unit unknown → keep generic
                return f"ค่าเฉลี่ย Leadtime เดือนนี้ {_fmt_num(v)}"

    # Ranking / multi-row
    if template_key in ["SALES_BY_PRODUCT_TOP", "SALES_BY_CAMPAIGN_TOP", "CREDIT_REJECT_REASON_TOP"]:
        cols = list(df.columns)
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        label_cols = [c for c in cols if c not in num_cols]
        if not num_cols or not label_cols:
            return None
        val_col = num_cols[0]
        label_col = label_cols[0]
        topn = int(qb_row["top_n"]) if qb_row is not None and pd.notna(qb_row.get("top_n")) else min(3, len(df))
        lines = []
        for i, r in df.head(topn).iterrows():
            lines.append(f"- {r[label_col]}: {_fmt_money(r[val_col])}")
        title = "Top" if topn else "ผลลัพธ์"
        if template_key == "SALES_BY_PRODUCT_TOP":
            return f"{title} สินค้าขายดี:\n" + "\n".join(lines)
        if template_key == "SALES_BY_CAMPAIGN_TOP":
            return f"{title} แคมเปญที่ทำยอดขายสูง:\n" + "\n".join(lines)
        if template_key == "CREDIT_REJECT_REASON_TOP":
            return f"{title} เหตุผลที่ถูกปฏิเสธ:\n" + "\n".join(lines)

    # Diagnostic drops (branch/dim)
    if template_key in ["SALES_BY_BRANCH_DIFF_VS_PREV", "SALES_BY_DIM_DIFF_VS_PREV"]:
        cols = list(df.columns)
        label_candidates = ["branch_code", "branch", "dim_value", "dimension", "name", "code"]
        label_col = next((c for c in label_candidates if c in cols), None)
        if label_col is None:
            num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
            label_cols = [c for c in cols if c not in num_cols]
            label_col = label_cols[0] if label_cols else None

        diff_candidates = ["diff_value", "diff", "delta_value", "drop_value", "change_value"]
        diff_col = next((c for c in diff_candidates if c in cols), None)
        pct_candidates = ["diff_pct", "pct_change", "delta_pct", "change_pct"]
        pct_col = next((c for c in pct_candidates if c in cols), None)

        if label_col and (diff_col or pct_col) and len(df) >= 1:
            r0 = df.iloc[0]
            label = r0[label_col]
            if diff_col and pd.notna(r0[diff_col]):
                direction = "เพิ่มขึ้น" if float(r0[diff_col]) >= 0 else "ลดลง"
                msg = f"มิติที่เปลี่ยนแปลงมากสุดคือ {label} โดย{direction} {_fmt_money(abs(r0[diff_col]))}"
                if pct_col and pd.notna(r0[pct_col]):
                    msg += f" ({_fmt_pct(abs(r0[pct_col]))})"
                return msg
            if pct_col and pd.notna(r0[pct_col]):
                direction = "เพิ่มขึ้น" if float(r0[pct_col]) >= 0 else "ลดลง"
                return f"มิติที่เปลี่ยนแปลงมากสุดคือ {label} โดย{direction} {_fmt_pct(abs(r0[pct_col]))}"

    return None


def df_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "(empty)"
    return df.head(max_rows).to_markdown(index=False)

def llm_grounded_answer(
    model_name: str,
    user_question: str,
    template_key: str,
    df: pd.DataFrame,
    qb_row: Optional[pd.Series],
) -> Optional[str]:
    """
    LLM summary with strict grounding.
    Returns answer text if JSON is valid AND values match df; otherwise None.
    """
    qb_info = {}
    if qb_row is not None:
        qb_info = {
            "intent_type": qb_row.get("intent_type"),
            "compare_type": qb_row.get("compare_type"),
            "dimension": qb_row.get("dimension"),
            "metric_expression": qb_row.get("metric_expression"),
            "top_n": qb_row.get("top_n"),
        }

    table_md = df_to_markdown(df, max_rows=20)
    meta = {"rows": int(df.shape[0]), "columns": df.columns.tolist()}

    prompt = f"""
คุณคือผู้ช่วยสรุปผลจากฐานข้อมูลแบบ "ห้ามเดา" และ "ห้ามแต่งข้อมูล"

เงื่อนไขสำคัญ:
- ตอบได้เฉพาะสิ่งที่พิสูจน์ได้จาก TABLE RESULT เท่านั้น
- ห้ามใช้ความรู้ภายนอก ห้ามคาดเดาว่าช่วงเวลาเป็น "เดือนนี้/ทั้งปี" ถ้าไม่มีระบุในผลลัพธ์
- ถ้าข้อมูลไม่พอให้ตอบว่า: "ข้อมูลไม่เพียงพอจากฐานข้อมูลเพื่อสรุปคำตอบนี้"
- ต้องอ้างอิงค่าจากตารางจริง (ตัวเลขต้องตรงกับตาราง)

ส่งออกเป็น JSON เท่านั้นตาม schema:
{{
  "answer_th": "คำตอบภาษาไทยแบบสั้น ชัดเจน (1-3 ประโยค)",
  "used_columns": ["colA","colB"],
  "used_values": {{"colA":"<value from table>", "colB":"<value from table>"}}
}}

Context:
- question: {user_question}
- template_key: {template_key}
- question_bank_meta: {json.dumps(qb_info, ensure_ascii=False)}
- result_meta: {json.dumps(meta, ensure_ascii=False)}
- TABLE RESULT (head):
{table_md}
""".strip()

    try:
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        raw = (resp.text or "").strip()

        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return None
        obj = json.loads(m.group(0))

        answer = obj.get("answer_th")
        used_cols = obj.get("used_columns", [])
        used_vals = obj.get("used_values", {})

        if not isinstance(answer, str) or not answer.strip():
            return None
        if not isinstance(used_cols, list) or not isinstance(used_vals, dict):
            return None

        for c in used_cols:
            if c not in df.columns:
                return None

        df_str = df.astype(str)
        for c, v in used_vals.items():
            if c not in df.columns:
                return None
            v_str = str(v)
            if (df_str[c] == v_str).any():
                continue
            try:
                v_num = float(v_str.replace(",", ""))
                col_num = pd.to_numeric(df[c], errors="coerce")
                if (col_num - v_num).abs().min() <= 1e-6:
                    continue
            except Exception:
                pass
            return None

        return answer.strip()
    except Exception:
        return None


def hybrid_answer(
    model_name: str,
    user_question: str,
    template_key: str,
    df: pd.DataFrame,
    question_bank_df: pd.DataFrame
) -> str:
    """
    Rule-first. If rule cannot answer confidently, use grounded LLM with validation.
    If still not safe, return a safe fallback (no hallucination).
    """
    qb_row = None
    try:
        qb_row = question_bank_df.loc[question_bank_df["sql_template_key"] == template_key].iloc[0]
    except Exception:
        qb_row = None

    rb = rule_based_answer(template_key, df, qb_row=qb_row)
    if rb:
        return rb

    llm = llm_grounded_answer(
        model_name=model_name,
        user_question=user_question,
        template_key=template_key,
        df=df,
        qb_row=qb_row
    )
    if llm:
        return llm

    if df is None or df.empty:
        return "ไม่พบข้อมูลจากคำถามนี้"
    return f"ได้ผลลัพธ์จากฐานข้อมูล {df.shape[0]} แถว {df.shape[1]} คอลัมน์ (โปรดดูตาราง Result เพื่อรายละเอียด)"


# =========================
# 3) Existing helpers
# =========================
APP_VERSION = "v2025-12-31-hybrid1-cte-fix"

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
    return pd.read_excel(uploaded, sheet_name=0)


# =========================
# 4) UI
# =========================
st.set_page_config(page_title="DSYP Chat-to-SQL (Gemini + SQLite)", layout="wide")
st.title("DSYP Chat-to-SQL (Gemini + SQLite)")
st.sidebar.caption(f"APP_VERSION: {APP_VERSION}")
st.caption("อัปโหลดไฟล์ → ตั้งค่า API key → ถามคำถาม → ได้ SQL + ผลลัพธ์ตาราง + คำตอบแบบ Hybrid (Rule + LLM)")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Google Gemini API Key", type="password")
    model_name = st.text_input("Model name", value="gemini-2.0-flash")

    st.divider()
    st.subheader("Upload config")
    table_name = st.text_input("CSV table name in SQLite", value="SALES_MASTER")

col1, col2, col3 = st.columns(3)
with col1:
    qb_file = st.file_uploader("Upload question_bank.xlsx", type=["xlsx"])
with col2:
    tpl_file = st.file_uploader("Upload sql_templates_with_placeholder.xlsx", type=["xlsx"])
with col3:
    csv_file = st.file_uploader("Upload data CSV", type=["csv"])

DB_PATH = "/tmp/app.db"
if "conn" not in st.session_state:
    st.session_state.conn = sqlite3.connect(DB_PATH, check_same_thread=False)

question_bank_df = read_xlsx(qb_file) if qb_file is not None else None
templates_df = read_xlsx(tpl_file) if tpl_file is not None else None

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
        genai.configure(api_key=api_key)

        router_out = call_router_llm(
            user_question=user_question,
            question_bank_df=question_bank_df,
            schema_doc=schema_doc,
            model_name=model_name,
        )

        template_key = router_out.get("sql_template_key", "")

        allowed = set(question_bank_df["sql_template_key"].dropna().astype(str).unique().tolist())
        if template_key not in allowed:
            st.error("คำถามนี้อยู่นอกขอบเขต question_bank (ถูก block เพื่อป้องกัน hallucination)")
            st.json({"router_out": router_out, "allowed_template_keys": sorted(list(allowed))})
            st.stop()

        final_sql, params = build_params_for_template(
            router_out=router_out,
            question_bank_df=question_bank_df,
            templates_df=templates_df,
        )

        final_sql = (
            final_sql
            .replace("—", "--")
            .replace("≥", ">=")
            .replace("≤", "<=")
        )

        ok, msg = is_safe_readonly_sql(final_sql, st.session_state.conn)
        if not ok:
            st.error(f"SQL ถูก block: {msg}")
            with st.expander("ดู SQL ที่ถูก block (optional)"):
                st.code(final_sql, language="sql")
            st.stop()

        df = pd.read_sql_query(final_sql, st.session_state.conn)

        answer_text = hybrid_answer(
            model_name=model_name,
            user_question=user_question,
            template_key=template_key,
            df=df,
            question_bank_df=question_bank_df,
        )

        st.markdown(f"**คำถาม:** {user_question}")
        st.markdown(f"**คำตอบ:** {answer_text}")

        st.divider()

        meta = {"rows": int(df.shape[0]), "columns": df.columns.tolist()}

        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Router output")
            st.json(router_out)

        with st.expander("ดู SQL ที่รัน (optional)"):
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
