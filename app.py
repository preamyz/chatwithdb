# app.py (LLM + Rule Hybrid Answer, anti-hallucination) - patched
import streamlit as st
import pandas as pd
import sqlite3
import io
import re
import json
import difflib
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



# -------- Conversational helpers (Thai) --------
_TH_MONTHS = {
    1: "ม.ค.", 2: "ก.พ.", 3: "มี.ค.", 4: "เม.ย.", 5: "พ.ค.", 6: "มิ.ย.",
    7: "ก.ค.", 8: "ส.ค.", 9: "ก.ย.", 10: "ต.ค.", 11: "พ.ย.", 12: "ธ.ค."
}

def _month_label_from_range(start_iso: Optional[str]) -> Optional[str]:
    """Return Thai month label like 'ธ.ค. 2025' from YYYY-MM-DD."""
    if not start_iso:
        return None
    try:
        y, m, _ = start_iso.split("-")
        y = int(y); m = int(m)
        if 1 <= m <= 12:
            return f"{_TH_MONTHS[m]} {y}"
    except Exception:
        return None
    return None

def _month_label(user_question: str, params: Optional[Dict[str, Any]] = None) -> str:
    """Prefer month/year parsed from Thai question; else fallback to params['cur_start'] or today."""
    parsed = parse_month_year_from_th_question(user_question or "")
    if parsed:
        y, m = parsed
        return f"{_TH_MONTHS.get(m, str(m))} {y}"
    if params:
        label = _month_label_from_range(params.get("cur_start"))
        if label:
            return label
    # fallback today
    t = date.today()
    return f"{_TH_MONTHS.get(t.month, str(t.month))} {t.year}"

def _infer_unit(template_key: str, qb_row: Optional[pd.Series] = None, df: Optional[pd.DataFrame] = None) -> str:
    """Infer unit for conversational answer."""
    metric_expr = ""
    if qb_row is not None:
        try:
            metric_expr = str(qb_row.get("metric_expression") or "").upper()
        except Exception:
            metric_expr = ""

    # explicit by template
    if template_key in {"CREDIT_LEADTIME_AVG"}:
        return "วัน"
    if "RATE" in template_key or template_key in {"CREDIT_APPROVAL_RATE_VS_PREV", "CREDIT_CANCELLATION_RATE_VS_PREV"}:
        return "%"

    # infer from metric expression
    if any(k in metric_expr for k in ["COUNT", "DISTINCTCOUNT"]):
        # domain default: contract counts
        return "สัญญา"
    if any(k in metric_expr for k in ["SUM", "AMOUNT", "VALUE", "PRICE", "REVENUE"]):
        return "บาท"

    # infer from df column names
    if df is not None and not df.empty:
        cols = [c.lower() for c in df.columns]
        if any("rate" in c or "pct" in c for c in cols):
            return "%"
        if any("leadtime" in c or "days" in c for c in cols):
            return "วัน"
        if any("count" in c or c.endswith("_cnt") for c in cols):
            return "สัญญา"

    # safe default
    return "รายการ"

def _fmt_value_by_unit(v: Any, unit: str) -> str:
    if v is None:
        return "-"
    if unit == "%":
        # accept 0-1 or 0-100
        try:
            fv = float(v)
            if 0 <= fv <= 1:
                fv *= 100.0
            return _fmt_pct(fv).replace("%", "")  # return number only
        except Exception:
            return str(v)
    if unit == "บาท":
        return _fmt_money(v)
    # counts/days
    return _fmt_num(v)

def _normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[,\.\!\?\(\)\[\]\{\}\:\;\"\'\-_/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _local_best_template(user_question: str, question_bank_df: pd.DataFrame) -> Tuple[Optional[str], float, Optional[str]]:
    """Local fuzzy match against question_bank (fast, avoids LLM when confident)."""
    uq = _normalize_text(user_question)
    best_key, best_score, best_q = None, 0.0, None
    if question_bank_df is None or question_bank_df.empty:
        return None, 0.0, None
    for _, r in question_bank_df.iterrows():
        q = str(r.get("question_text_th") or "")
        if not q:
            continue
        score = difflib.SequenceMatcher(None, uq, _normalize_text(q)).ratio()
        if score > best_score:
            best_score = score
            best_key = str(r.get("sql_template_key") or "").strip() or None
            best_q = q
    return best_key, float(best_score), best_q
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
    """Fast, deterministic Thai conversational answers for known templates.

    Return None to allow LLM grounded fallback.
    """
    if df is None or df.empty:
        return "ไม่พบข้อมูลจากคำถามนี้"

    user_q = st.session_state.get("last_user_question", "")
    params = st.session_state.get("last_params", {}) or {}
    month_label = _month_label(user_q, params)

    unit = _infer_unit(template_key, qb_row=qb_row, df=df)

    # ---------- Single row (aggregates / comparisons) ----------
    if df.shape[0] == 1:
        row = df.iloc[0].to_dict()

        # A) totals / counts
        if template_key in {"SALES_TOTAL_CURR", "CREDIT_CONTRACT_CNT"}:
            v = _first_existing(row, ["total_contracts", "contract_cnt", "contracts", "cnt", "total_value", "total_sales", "sales_value", "sum_value"])
            if v is None:
                return None
            val = _fmt_value_by_unit(v, unit)
            if unit == "%":
                return f"ผลลัพธ์เดือน {month_label} คือ {val}%"
            return f"ยอดขายเดือน {month_label} คือ {val} {unit}"

        if template_key == "CREDIT_LEADTIME_AVG":
            v = _first_existing(row, ["avg_leadtime_days", "avg_days", "leadtime_avg", "avg_leadtime"])
            if v is None:
                return None
            val = _fmt_value_by_unit(v, unit)
            return f"ระยะเวลาอนุมัติเดือน {month_label} เฉลี่ย {val} {unit}"

        # B) vs prev (need cur & prev)
        if template_key in {"SALES_TOTAL_CURR_VS_PREV", "CREDIT_APPROVAL_RATE_VS_PREV", "CREDIT_CANCELLATION_RATE_VS_PREV"}:
            cur = _first_existing(row, ["cur_value", "current_value", "cur", "cur_cnt", "cur_rate"])
            prev = _first_existing(row, ["prev_value", "previous_value", "prev", "prev_cnt", "prev_rate"])
            diff = _first_existing(row, ["diff_value", "delta_value", "diff", "delta"])
            diff_pct = _first_existing(row, ["diff_pct", "pct_change", "delta_pct", "diff_percent"])
            # if pct missing but cur/prev exist, compute
            try:
                if diff_pct is None and cur is not None and prev not in (None, 0, 0.0):
                    diff_pct = (float(cur) - float(prev)) / float(prev) * 100.0
            except Exception:
                pass
            # if diff missing but cur/prev exist, compute
            try:
                if diff is None and cur is not None and prev is not None:
                    diff = float(cur) - float(prev)
            except Exception:
                pass

            if cur is None and diff is None and diff_pct is None:
                return None

            # Determine up/down wording
            updown = None
            try:
                if diff is not None:
                    updown = "ดีขึ้น" if float(diff) > 0 else ("แย่ลง" if float(diff) < 0 else "ทรงตัว")
                elif diff_pct is not None:
                    updown = "ดีขึ้น" if float(diff_pct) > 0 else ("แย่ลง" if float(diff_pct) < 0 else "ทรงตัว")
            except Exception:
                updown = None

            # Format pct and abs
            pct_txt = None
            if diff_pct is not None:
                try:
                    pct_txt = f"{abs(float(diff_pct)):.1f}%"
                except Exception:
                    pct_txt = str(diff_pct)

            abs_txt = None
            if diff is not None:
                try:
                    abs_txt = _fmt_value_by_unit(abs(float(diff)), unit)
                except Exception:
                    abs_txt = str(diff)

            # rate templates should end with %
            if unit == "%":
                # diff is in percentage points if provided; else use pct_txt (relative)
                if abs_txt is not None and diff is not None:
                    abs_txt = f"{abs(float(diff)):.1f}"  # pp
                if updown is None:
                    updown = "เปลี่ยนแปลง"
                if pct_txt and abs_txt:
                    return f"เดือน {month_label} {updown}กว่าเดือนที่แล้วประมาณ {pct_txt} (เปลี่ยน {abs_txt} จุด)"
                if pct_txt:
                    return f"เดือน {month_label} {updown}กว่าเดือนที่แล้วประมาณ {pct_txt}"
                if abs_txt:
                    return f"เดือน {month_label} {updown}กว่าเดือนที่แล้วประมาณ {abs_txt} จุด"
                return None

            # count / money templates
            if updown is None:
                updown = "เปลี่ยนแปลง"
            if pct_txt and abs_txt:
                return f"ยอดขายเดือน {month_label} {updown}กว่าเดือนที่แล้วประมาณ {pct_txt} หรือ {abs_txt} {unit}"
            if pct_txt:
                return f"ยอดขายเดือน {month_label} {updown}กว่าเดือนที่แล้วประมาณ {pct_txt}"
            if abs_txt:
                return f"ยอดขายเดือน {month_label} {updown}กว่าเดือนที่แล้วประมาณ {abs_txt} {unit}"
            return None

    # ---------- Multi-row (rankings / breakdowns) ----------
    # Ranking top N
    if template_key in {"SALES_BY_PRODUCT_TOP", "SALES_BY_CAMPAIGN_TOP", "CREDIT_REJECT_REASON_TOP"}:
        label_col = guess_label_col(df, preferred=["product_name", "product", "campaign_name", "campaign", "reject_reason", "reason"])
        value_col = guess_value_col(df, preferred=["total_value", "total_sales", "sales_value", "cnt", "contract_cnt", "value"])
        if not label_col or not value_col:
            return None
        top_n = 5
        try:
            if qb_row is not None and pd.notna(qb_row.get("top_n")):
                top_n = int(qb_row.get("top_n"))
        except Exception:
            pass
        top_n = max(1, min(top_n, 10))
        head = df.head(top_n).copy()
        lines = []
        for i, r in enumerate(head.itertuples(index=False), start=1):
            label = getattr(r, label_col)
            value = getattr(r, value_col)
            vtxt = _fmt_value_by_unit(value, unit)
            suffix = "%" if unit == "%" else f" {unit}"
            lines.append(f"{i}) {label}: {vtxt}{suffix}")
        title = "Top รายการ"
        if template_key == "SALES_BY_PRODUCT_TOP":
            title = "Top สินค้า"
        elif template_key == "SALES_BY_CAMPAIGN_TOP":
            title = "Top แคมเปญ"
        elif template_key == "CREDIT_REJECT_REASON_TOP":
            title = "Top เหตุผลที่ Reject"
        return f"{title} เดือน {month_label}:\n" + "\n".join(lines)

    # Largest change by branch/dimension vs prev
    if template_key in {"SALES_BY_BRANCH_DIFF_VS_PREV", "SALES_BY_DIM_DIFF_VS_PREV"}:
        # expect columns like label, cur_value, prev_value, diff_value, diff_pct
        label_col = guess_label_col(df, preferred=["branch_name", "branch", "dealer_name", "dealer", "dim_value", "dimension", "name"])
        diff_col = guess_value_col(df, preferred=["diff_value", "delta_value", "diff", "delta"])
        pct_col = _first_existing({c:c for c in df.columns}, ["diff_pct","pct_change","delta_pct","diff_percent"])
        # If pct_col is a string key returned above:
        if isinstance(pct_col, str) and pct_col in df.columns:
            pct_col_name = pct_col
        else:
            pct_col_name = None

        if not label_col:
            label_col = df.columns[0]

        # pick row with minimum diff (largest drop) if diff exists; else use first row
        pick = df.iloc[0]
        if diff_col and diff_col in df.columns and pd.api.types.is_numeric_dtype(df[diff_col]):
            pick = df.loc[df[diff_col].astype(float).idxmin()]
        label = pick.get(label_col)
        diff = pick.get(diff_col) if diff_col and diff_col in df.columns else None
        pct = pick.get(pct_col_name) if pct_col_name else None

        if diff is None and pct is None:
            return None
        try:
            diff_f = float(diff) if diff is not None else None
        except Exception:
            diff_f = None
        direction = "ลดลง" if (diff_f is not None and diff_f < 0) else ("เพิ่มขึ้น" if (diff_f is not None and diff_f > 0) else "เปลี่ยนแปลง")
        abs_txt = _fmt_value_by_unit(abs(diff_f), unit) if diff_f is not None else None
        pct_txt = None
        if pct is not None:
            try:
                pct_txt = f"{abs(float(pct)):.1f}%"
            except Exception:
                pct_txt = str(pct)
        if pct_txt and abs_txt:
            return f"ตัวที่เปลี่ยนแปลงมากสุดคือ {label} โดย{direction} {abs_txt} {unit} ({pct_txt})"
        if abs_txt:
            return f"ตัวที่เปลี่ยนแปลงมากสุดคือ {label} โดย{direction} {abs_txt} {unit}"
        if pct_txt:
            return f"ตัวที่เปลี่ยนแปลงมากสุดคือ {label} โดย{direction} {pct_txt}"
        return None

    return None



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
    month_label = _month_label(user_question, st.session_state.get("last_params", {}) or {})

    prompt = f"""
คุณคือผู้ช่วยสรุปผลลัพธ์จาก SQL ให้เป็น "ภาษาคน" แบบคุยกัน แต่ยังคงต้อง *อ้างอิงจากตารางจริงเท่านั้น*

กติกา (สำคัญมาก):
- ตอบได้เฉพาะสิ่งที่พิสูจน์ได้จาก TABLE RESULT เท่านั้น (ห้ามเดา/ห้ามแต่งตัวเลข)
- ถ้าข้อมูลไม่พอ ให้ตอบว่า: "ข้อมูลไม่เพียงพอจากฐานข้อมูลเพื่อสรุปคำตอบนี้"
- สไตล์คำตอบ: ภาษาไทยธรรมชาติ เหมือนคุยกับคน, 1–2 ประโยค (สั้น ชัด)
- ถ้ามีค่า month_label ให้ระบุช่วงเวลาเป็น: เดือน {month_label}

ส่งออกเป็น JSON เท่านั้นตาม schema:
{{
  "answer_th": "คำตอบภาษาไทยแบบสนทนา (1–2 ประโยค)",
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

        # --- Fast local match first (helps paraphrase + reduces LLM calls) ---
        local_key, local_score, local_q = _local_best_template(user_question, question_bank_df)
        router_out = {}
        if local_key and local_score >= 0.72:
            router_out = {"sql_template_key": local_key, "router_mode": "local_fuzzy", "score": round(local_score, 3), "matched_question": local_q}
        else:
            router_out = call_router_llm(
                user_question=user_question,
                question_bank_df=question_bank_df,
                schema_doc=schema_doc,
                model_name=model_name,
            )

        template_key = str(router_out.get("sql_template_key", "") or "").strip()


        allowed = set(question_bank_df["sql_template_key"].dropna().astype(str).unique().tolist())
        if template_key not in allowed:
            # fallback: try local fuzzy match even if router output is unexpected
            fb_key, fb_score, fb_q = _local_best_template(user_question, question_bank_df)
            if fb_key and fb_score >= 0.55:
                template_key = fb_key
                router_out = {"sql_template_key": fb_key, "router_mode": "fallback_local_fuzzy", "score": round(fb_score, 3), "matched_question": fb_q}
            else:
                st.error("คำถามนี้อยู่นอกขอบเขต question_bank (ถูก block เพื่อป้องกัน hallucination)")
                st.json({"router_out": router_out, "fallback_score": round(fb_score or 0, 3)})
                st.stop()


        final_sql, params = build_params_for_template(
            router_out=router_out,
            question_bank_df=question_bank_df,
            templates_df=templates_df,
        )
        st.session_state["last_params"] = params


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
