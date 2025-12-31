# app.py (LLM + Rule Hybrid Answer, anti-hallucination) - patched
import streamlit as st
import pandas as pd
import sqlite3
import io
import re
import json
from typing import Dict, Any, Optional, Tuple, List
from datetime import date
from dateutil.relativedelta import relativedelta

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
    sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql.strip()

def extract_table_names(sql: str) -> List[str]:
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
    Extract CTE names from WITH clause, e.g.
      WITH cur AS (...), prev AS (...) SELECT ...
    """
    cleaned = strip_sql_comments(sql)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not re.match(r"^WITH\b", cleaned, flags=re.IGNORECASE):
        return []

    m = re.match(r"^WITH\s+(.*)\s+SELECT\b", cleaned, flags=re.IGNORECASE)
    if not m:
        return []

    cte_part = m.group(1)
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
    if not isinstance(sql, str) or not sql.strip():
        return False, "SQL ว่าง"

    s = sql.strip()

    semis = [m.start() for m in re.finditer(";", s)]
    if len(semis) > 1 or (len(semis) == 1 and semis[0] != len(s) - 1):
        return False, "SQL มีหลาย statement (ถูก block เพื่อความปลอดภัย)"

    core = strip_sql_comments(s)
    if not re.match(r"^(SELECT|WITH)\b", core, flags=re.IGNORECASE):
        return False, "อนุญาตเฉพาะ SELECT/WITH เท่านั้น"

    if DANGEROUS_SQL_TOKENS.search(core):
        return False, "ตรวจพบคำสั่งที่ไม่ปลอดภัย (DDL/DML) จึงถูก block"

    # Ensure referenced tables exist (ignore CTE aliases)
    tables = extract_table_names(core)
    ctes = set(extract_cte_names(core))
    if tables:
        exist = existing_tables(conn)
        missing = [t for t in tables if (t not in exist) and (t not in ctes)]
        if missing:
            return False, f"SQL อ้างอิงตารางที่ไม่มีใน DB: {missing}"

    return True, "OK"


# =========================
# 1.5) Question parsing: month/year override
# =========================
def parse_month_year_from_th_question(q: str) -> Optional[Tuple[int, int]]:
    """
    Support patterns like:
      - 'ยอดขายเดือน 7 ปี 2025'
      - 'ยอดขายเดือน 07 ปี 2025'
      - 'ยอดขายเดือน 7/2025'
    Return (year, month) or None.
    """
    if not q:
        return None
    q = q.strip()

    m = re.search(r"เดือน\s*(\d{1,2})\s*ปี\s*(\d{4})", q)
    if m:
        month = int(m.group(1))
        year = int(m.group(2))
        if 1 <= month <= 12:
            return (year, month)

    m = re.search(r"เดือน\s*(\d{1,2})\s*/\s*(\d{4})", q)
    if m:
        month = int(m.group(1))
        year = int(m.group(2))
        if 1 <= month <= 12:
            return (year, month)

    return None

def month_range(year: int, month: int) -> Tuple[str, str]:
    start = date(year, month, 1)
    end = start + relativedelta(months=1)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

def override_sql_dates_by_question(sql: str, template_key: str, user_question: str) -> str:
    """
    V2: Replace ONLY date filters in conditions like:
      <date_field> >= 'YYYY-MM-DD'
      <date_field> <  'YYYY-MM-DD'
    This avoids accidentally replacing dates in other parts of SQL (CTE/CASE/metadata strings).

    Supports:
      - SALES_TOTAL_CURR : replace first (>=) and first (<)
      - *_VS_PREV       : replace 2 pairs (cur then prev)
    """
    parsed = parse_month_year_from_th_question(user_question)
    if not parsed:
        return sql

    year, month = parsed
    cur_start, cur_end = month_range(year, month)

    prev_dt = date(year, month, 1) - relativedelta(months=1)
    prev_start, prev_end = month_range(prev_dt.year, prev_dt.month)

    # Pattern: <field> >= 'YYYY-MM-DD'
    ge_pat = re.compile(r"(\b[A-Za-z_][A-Za-z0-9_]*\b\s*>=\s*)'(\d{4}-\d{2}-\d{2})'", re.IGNORECASE)
    # Pattern: <field> < 'YYYY-MM-DD'
    lt_pat = re.compile(r"(\b[A-Za-z_][A-Za-z0-9_]*\b\s*<\s*)'(\d{4}-\d{2}-\d{2})'", re.IGNORECASE)

    out = sql

    if template_key == "SALES_TOTAL_CURR":
        out = ge_pat.sub(rf"\1'{cur_start}'", out, count=1)
        out = lt_pat.sub(rf"\1'{cur_end}'", out, count=1)
        return out

    if template_key.endswith("_VS_PREV"):
        out = ge_pat.sub(rf"\1'{cur_start}'", out, count=1)
        out = lt_pat.sub(rf"\1'{cur_end}'", out, count=1)

        out = ge_pat.sub(rf"\1'{prev_start}'", out, count=1)
        out = lt_pat.sub(rf"\1'{prev_end}'", out, count=1)
        return out

    return out


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

def choose_pretty_label_column(df: pd.DataFrame, preferred: List[str]) -> Optional[str]:
    """
    Prefer *_name columns if exist; else fall back to preferred codes.
    """
    cols = set(df.columns)
    for p in preferred:
        if p in cols:
            name_candidate = None
            if p.endswith("_code"):
                name_candidate = p.replace("_code", "_name")
            elif p.endswith("_id"):
                name_candidate = p.replace("_id", "_name")
            if name_candidate and name_candidate in cols:
                return name_candidate
        # direct preferred label
    for p in preferred:
        if p in cols:
            return p
    # final: first non-numeric
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    label_cols = [c for c in df.columns if c not in num_cols]
    return label_cols[0] if label_cols else None

def rule_based_answer(template_key: str, df: pd.DataFrame, qb_row: Optional[pd.Series] = None) -> Optional[str]:
    if df is None or df.empty:
        return "ไม่พบข้อมูลจากคำถามนี้"

    if df.shape[0] == 1:
        row = df.iloc[0].to_dict()

        if template_key == "SALES_TOTAL_CURR":
            v = _first_existing(row, ["total_value", "total_sales", "sales_value", "sum_value"])
            if v is not None:
                # ✅ ใช้เดือน/ปีจากคำถามถ้ามี
                parsed = parse_month_year_from_th_question(st.session_state.get("last_user_question", ""))
                if parsed:
                    y, m = parsed
                    return f"ยอดขายเดือน {m} ปี {y} {_fmt_money(v)} บาท"
                return f"ยอดขายเดือนนี้ {_fmt_money(v)} บาท"


        if template_key == "SALES_TOTAL_CURR_VS_PREV":
            cur = _first_existing(row, ["cur_value", "cur_total", "cur_sales", "cur"])
            prev = _first_existing(row, ["prev_value", "prev_total", "prev_sales", "prev"])
            diff = _first_existing(row, ["diff_value", "diff", "delta_value"])
            diff_pct = _first_existing(row, ["diff_pct", "pct_change", "delta_pct"])
            if cur is not None and prev is not None:
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

    # TOP lists (prefer *_name)
    if template_key in ["SALES_BY_PRODUCT_TOP", "SALES_BY_CAMPAIGN_TOP", "CREDIT_REJECT_REASON_TOP"]:
        cols = list(df.columns)
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            return None
        val_col = num_cols[0]

        if template_key == "SALES_BY_CAMPAIGN_TOP":
            label_col = choose_pretty_label_column(df, ["campaign_type", "campaign_code", "campaign"])
            title = "Top แคมเปญที่ทำยอดขายสูง"
        elif template_key == "SALES_BY_PRODUCT_TOP":
            label_col = choose_pretty_label_column(df, ["product_name", "product_code", "product"])
            title = "Top สินค้าขายดี"
        else:
            label_col = choose_pretty_label_column(df, ["reject_reason_name", "reject_reason_code", "reason"])
            title = "Top เหตุผลที่ถูกปฏิเสธ"

        if not label_col:
            return None

        topn = int(qb_row["top_n"]) if qb_row is not None and pd.notna(qb_row.get("top_n")) else min(3, len(df))
        lines = []
        for _, r in df.head(topn).iterrows():
            lines.append(f"- {r[label_col]}: {_fmt_money(r[val_col])}")

        return f"{title}:\n" + "\n".join(lines)

    # Diagnostic drops (branch/dim) - wording aligned
    if template_key in ["SALES_BY_BRANCH_DIFF_VS_PREV", "SALES_BY_DIM_DIFF_VS_PREV"]:
        cols = list(df.columns)
        if template_key == "SALES_BY_BRANCH_DIFF_VS_PREV":
            label_col = choose_pretty_label_column(df, ["branch_name", "branch_code", "branch"])
            subject_word = "สาขา"
        else:
            label_col = choose_pretty_label_column(df, ["dim_name", "dim_value", "dimension", "name", "code"])
            subject_word = "มิติ"

        diff_candidates = ["diff_value", "diff", "delta_value", "drop_value", "change_value"]
        diff_col = next((c for c in diff_candidates if c in cols), None)
        pct_candidates = ["diff_pct", "pct_change", "delta_pct", "change_pct"]
        pct_col = next((c for c in pct_candidates if c in cols), None)

        if label_col and (diff_col or pct_col) and len(df) >= 1:
            r0 = df.iloc[0]
            label = r0[label_col]
            if diff_col and pd.notna(r0[diff_col]):
                direction = "เพิ่มขึ้น" if float(r0[diff_col]) >= 0 else "ลดลง"
                msg = f"{subject_word}ที่เปลี่ยนแปลงมากสุดคือ {label} โดย{direction} {_fmt_money(abs(r0[diff_col]))}"
                if pct_col and pd.notna(r0[pct_col]):
                    msg += f" ({_fmt_pct(abs(r0[pct_col]))})"
                return msg
            if pct_col and pd.notna(r0[pct_col]):
                direction = "เพิ่มขึ้น" if float(r0[pct_col]) >= 0 else "ลดลง"
                return f"{subject_word}ที่เปลี่ยนแปลงมากสุดคือ {label} โดย{direction} {_fmt_pct(abs(r0[pct_col]))}"

    return None


def df_to_markdown_safe(df: pd.DataFrame, max_rows: int = 20) -> str:
    """
    Avoid dependency on `tabulate` by using a small fixed-width text table.
    """
    if df is None or df.empty:
        return "(empty)"
    show = df.head(max_rows).copy()
    # convert to string for safety
    show = show.astype(str)

    cols = list(show.columns)
    widths = {c: max(len(c), show[c].map(len).max()) for c in cols}
    widths = {c: min(40, w) for c, w in widths.items()}  # cap width

    def clip(s: str, w: int) -> str:
        return s if len(s) <= w else (s[:w-1] + "…")

    header = " | ".join(clip(c, widths[c]).ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    rows = []
    for _, r in show.iterrows():
        rows.append(" | ".join(clip(r[c], widths[c]).ljust(widths[c]) for c in cols))
    return "\n".join([header, sep] + rows)

def llm_grounded_answer(
    model_name: str,
    user_question: str,
    template_key: str,
    df: pd.DataFrame,
    qb_row: Optional[pd.Series],
) -> Optional[str]:
    qb_info = {}
    if qb_row is not None:
        qb_info = {
            "intent_type": qb_row.get("intent_type"),
            "compare_type": qb_row.get("compare_type"),
            "dimension": qb_row.get("dimension"),
            "metric_expression": qb_row.get("metric_expression"),
            "top_n": qb_row.get("top_n"),
        }

    table_text = df_to_markdown_safe(df, max_rows=20)
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
- TABLE RESULT (head as text):
{table_text}
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
APP_VERSION = "v2026-01-01-hybrid2"

def load_csv_to_sqlite(conn, table_name: str, file_bytes: bytes, if_exists: str = "replace"):
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
    table_name = st.text_input("CSV table name in SQLite", value="sales_data")

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

        # ✅ เก็บคำถามล่าสุดไว้ให้ rule-based ใช้ทำ wording (ถ้าคุณใช้ข้อ A ด้วย)
        st.session_state["last_user_question"] = user_question

        # ✅ แยก SQL สำหรับ "แสดงผล" กับ "รันจริง"
        display_sql = final_sql

        # ✅ IMPORTANT: ตัด comment ออกก่อน แล้วค่อย override date
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
