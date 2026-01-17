# app.py (LLM + Rule Hybrid Answer, anti-hallucination) - patched
import streamlit as st
import pandas as pd
import sqlite3

# =========================
# SQLite session init (CRITICAL)
# =========================
if "conn" not in st.session_state:
    st.session_state.conn = sqlite3.connect(":memory:", check_same_thread=False)

import io
import re
import json
import difflib
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
# =========================
# Assets paths (repo) - auto load (no user upload)
# =========================
ASSETS_DIR = Path(__file__).parent / "assets"
QB_PATH = ASSETS_DIR / ("question_bank_v3.xlsx" if (ASSETS_DIR / "question_bank_v3.xlsx").exists() else "question_bank.xlsx")
TPL_PATH = ASSETS_DIR / ("sql_templates_with_placeholder_v3.xlsx" if (ASSETS_DIR / "sql_templates_with_placeholder_v3.xlsx").exists() else "sql_templates_with_placeholder.xlsx")

# raw data CSV in /assets
SALES_CSV_PATH  = ASSETS_DIR / "sales_master_enhanced_2024_2025.csv"
CREDIT_CSV_PATH = ASSETS_DIR / "credit_contract_enhanced_2024_2025.csv"

def _load_csv_path_to_table(conn: sqlite3.Connection, csv_path: Path, table_name: str) -> int:
    """Load a CSV file into SQLite table. Returns number of rows loaded."""
    df = pd.read_csv(csv_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    return int(df.shape[0])

from datetime import date
from dateutil.relativedelta import relativedelta

import google.generativeai as genai
from dsyp_core import call_router_llm, build_params_for_template



# -----------------------------
# Safe casting helpers
# -----------------------------

def _safe_int(v, default: int = 0) -> int:
    """Convert Excel/DF values to int safely (handles NaN)."""
    try:
        import pandas as _pd
        if v is None or (_pd.isna(v)):  # type: ignore
            return int(default)
    except Exception:
        if v is None:
            return int(default)
    try:
        # handle numeric strings / floats
        if isinstance(v, str):
            s = v.strip()
            if s == '':
                return int(default)
            return int(float(s))
        return int(float(v))
    except Exception:
        return int(default)

# =========================
# 1) Utilities: Safe SQL
# =========================

def canonical_compare_df(template_key: str, conn: sqlite3.Connection, params: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """For selected templates, run a canonical cur vs prev query to avoid template-specific ambiguity (e.g., AVG vs SUM).
    Returns a df with columns: cur_value, prev_value (and optional pct_change, abs_change)."""
    if template_key not in {"SALES_TOTAL_CURR_VS_PREV"}:
        return None

    table = params.get("table_name")
    date_field = params.get("date_field") or "order_datetime"
    metric_expr = params.get("metric_expr") or "price_final"
    cur_start = params.get("cur_start")
    cur_end = params.get("cur_end")
    prev_start = params.get("prev_start")
    prev_end = params.get("prev_end")
    if not all([table, date_field, metric_expr, cur_start, cur_end, prev_start, prev_end]):
        return None

    # Canonical: SUM for sales value
    sql = f"""WITH
cur AS (
  SELECT SUM({metric_expr}) AS cur_value
  FROM {table}
  WHERE {date_field} >= '{cur_start}' AND {date_field} < '{cur_end}'
),
prev AS (
  SELECT SUM({metric_expr}) AS prev_value
  FROM {table}
  WHERE {date_field} >= '{prev_start}' AND {date_field} < '{prev_end}'
)
SELECT
  cur.cur_value AS cur_value,
  prev.prev_value AS prev_value,
  (cur.cur_value - prev.prev_value) AS abs_change,
  CASE WHEN prev.prev_value IS NULL OR prev.prev_value = 0 THEN NULL
       -- return percentage (not fraction) to avoid UI showing 0.1% for 11.7%
       ELSE (cur.cur_value - prev.prev_value) * 100.0 / prev.prev_value
  END AS pct_change
FROM cur CROSS JOIN prev;
"""
    try:
        return pd.read_sql_query(sql, conn)
    except Exception:
        return None

DANGEROUS_SQL_TOKENS = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|ATTACH|DETACH|VACUUM|PRAGMA)\b",
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


def df_to_markdown_safe(df: pd.DataFrame, max_rows: int = 20, max_cols: int = 12) -> str:
    """Convert a DataFrame to a compact pipe-table string for prompting.
    Avoids pandas.to_markdown() (tabulate dependency) to keep Streamlit Cloud light.
    """
    if df is None:
        return "(no result)"
    try:
        if hasattr(df, "empty") and df.empty:
            return "(no result)"
        view = df.copy()
        # limit columns and rows for speed / token budget
        if view.shape[1] > max_cols:
            view = view.iloc[:, :max_cols]
        view = view.head(max_rows)

        cols = [str(c) for c in view.columns.tolist()]
        # header
        out = []
        out.append("| " + " | ".join(cols) + " |")
        out.append("| " + " | ".join(["---"] * len(cols)) + " |")

        for _, row in view.iterrows():
            vals = []
            for c in cols:
                v = row[c]
                if pd.isna(v):
                    vals.append("")
                else:
                    s = str(v)
                    # keep it short and one-line
                    s = s.replace("\n", " ").replace("|", " ")
                    if len(s) > 80:
                        s = s[:77] + "..."
                    vals.append(s)
            out.append("| " + " | ".join(vals) + " |")
        return "\n".join(out)
    except Exception:
        # fallback: simple text table
        try:
            return df.head(max_rows).to_string(index=False)
        except Exception:
            return "(no result)"

# -----------------------------
# Optional chart rendering (lightweight, no extra LLM)
# -----------------------------
def _try_import_altair():
    try:
        import altair as alt  # type: ignore
        return alt
    except Exception:
        return None

def _render_sparkline(values, labels, color=None):
    """Render a tiny trend line. Falls back to st.line_chart if Altair isn't available."""
    if values is None or len(values) == 0:
        return
    alt = _try_import_altair()
    if alt is None:
        # Streamlit fallback (no custom color)
        st.line_chart(values, height=260)
        return

    import pandas as _pd
    dfc = _pd.DataFrame({"period": labels, "value": values})
    line_color = color or "#4C78A8"
    chart = (
        alt.Chart(dfc)
        .mark_line(point=alt.OverlayMarkDef(size=60), color=line_color)
        .encode(
            x=alt.X("period:N", title=None, axis=alt.Axis(labelAngle=0, labelFontSize=12)),
            y=alt.Y("value:Q", title=None, axis=alt.Axis(labelFontSize=12)),
            tooltip=["period:N", "value:Q"],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)

def _render_bar(df, label_col, value_col, top_n=10, color=None):
    """Render Top-N bar chart. Uses Altair if available; otherwise st.bar_chart."""
    if df is None or df.empty:
        return
    if label_col not in df.columns or value_col not in df.columns:
        return

    d = df[[label_col, value_col]].copy().head(top_n)

    alt = _try_import_altair()
    if alt is None:
        st.bar_chart(d.set_index(label_col), height=520)
        return

    chart_color = color or "#4C78A8"
    chart = (
        alt.Chart(d)
        .mark_bar(color=chart_color)
        .encode(
            y=alt.Y(f"{label_col}:N", sort="-x", title=None),
            x=alt.X(f"{value_col}:Q", title=None),
            tooltip=[alt.Tooltip(label_col, type="nominal"), alt.Tooltip(value_col, type="quantitative")],
        )
        .properties(height=min(760, 46 * len(d) + 120))
    )
    st.altair_chart(chart, use_container_width=True)

def _safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        # strings like '1,234'
        s = str(x).replace(",", "").strip()
        if s == "" or s.lower() == "none" or s.lower() == "nan":
            return None
        return float(s)
    except Exception:
        return None

def render_optional_visuals(template_key: str, df: pd.DataFrame, user_question: str, params: Optional[Dict[str, Any]], show_chart: bool):
    """Show optional charts per template_key using the SQL result df."""
    if not show_chart:
        return
    if df is None or df.empty:
        return

    # === Time-series templates (auto line charts) ===
    try:
        if template_key in ("SALES_TREND_LAST_N_MONTHS", "SALES_TREND_BY_YEAR") and df.shape[1] >= 2:
            label_col, value_col = df.columns[0], df.columns[1]
            labels = df[label_col].astype(str).tolist()
            values = [(_safe_float(x) or 0.0) for x in df[value_col].tolist()]
            _render_sparkline(values, labels)
            return

        if template_key in ("SALES_YOY_YTD_BY_MONTH",) and set(["month_no", "cur_value", "prev_value"]).issubset(set(df.columns)):
            # Build month labels
            month_map = {
                '01':'Jan','02':'Feb','03':'Mar','04':'Apr','05':'May','06':'Jun',
                '07':'Jul','08':'Aug','09':'Sep','10':'Oct','11':'Nov','12':'Dec'
            }
            labels = [month_map.get(str(m).zfill(2), str(m)) for m in df['month_no'].astype(str).tolist()]

            alt = _try_import_altair()
            if alt is None:
                # fallback: Streamlit line chart
                d2 = df[['month_no','cur_value','prev_value']].copy()
                d2['month_no'] = labels
                st.line_chart(d2.set_index('month_no'), height=300)
                return

            import pandas as _pd
            dlong = _pd.DataFrame({
                'month': labels + labels,
                'series': ['current']*len(labels) + ['previous']*len(labels),
                'value': [(_safe_float(x) or 0.0) for x in df['cur_value'].tolist()] + [(_safe_float(x) or 0.0) for x in df['prev_value'].tolist()],
            })
            chart = (
                alt.Chart(dlong)
                .mark_line(point=alt.OverlayMarkDef(size=60))
                .encode(
                    x=alt.X('month:N', title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('value:Q', title=None),
                    color=alt.Color('series:N', title=None),
                    tooltip=['month:N','series:N','value:Q'],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
            return
    except Exception:
        pass

    # Normalize columns
    cols = list(df.columns)

    # A) TOTAL / COUNT / AVG: show KPI card (fast, clean)
    if template_key in ("SALES_TOTAL_CURR", "CREDIT_CONTRACT_CNT", "CREDIT_LEADTIME_AVG"):
        val = _safe_float(df.iloc[0, 0]) if len(cols) >= 1 else None
        if val is None:
            return
        label = "Value"
        if template_key == "SALES_TOTAL_CURR":
            label = "Total (current period)"
        elif template_key == "CREDIT_CONTRACT_CNT":
            label = "Contract count (current period)"
        elif template_key == "CREDIT_LEADTIME_AVG":
            label = "Avg lead time (days)"
        st.metric(label=label, value=f"{val:,.0f}" if abs(val) >= 1 else f"{val:,.4f}")
        return

    # B) VS PREV: metric + sparkline (prev → cur → forecast)
    if template_key.endswith("_VS_PREV") or template_key in ("CREDIT_APPROVAL_RATE_VS_PREV", "CREDIT_CANCELLATION_RATE_VS_PREV"):
        # try common column names
        cur = None
        prev = None
        # prioritize explicit column names
        for c in ["cur_value", "current_value", "cur", "current", "value_cur"]:
            if c in df.columns:
                cur = _safe_float(df.iloc[0][c])
                break
        for c in ["prev_value", "previous_value", "prev", "previous", "value_prev"]:
            if c in df.columns:
                prev = _safe_float(df.iloc[0][c])
                break
        # fallback: 2 columns numeric
        if cur is None or prev is None:
            num_cols = [c for c in df.columns if _safe_float(df.iloc[0][c]) is not None]
            if len(num_cols) >= 2:
                c0, c1 = num_cols[0], num_cols[1]
                n0 = _safe_float(df.iloc[0][c0])
                n1 = _safe_float(df.iloc[0][c1])

                # If column names hint at prev/current, respect that first.
                name0 = str(c0).lower()
                name1 = str(c1).lower()
                if ("prev" in name0 or "previous" in name0 or "last" in name0) and not ("prev" in name1 or "previous" in name1 or "last" in name1):
                    prev, cur = n0, n1
                elif ("prev" in name1 or "previous" in name1 or "last" in name1) and not ("prev" in name0 or "previous" in name0 or "last" in name0):
                    cur, prev = n0, n1
                elif ("cur" in name0 or "current" in name0) and not ("cur" in name1 or "current" in name1):
                    cur, prev = n0, n1
                elif ("cur" in name1 or "current" in name1) and not ("cur" in name0 or "current" in name0):
                    prev, cur = n0, n1
                else:
                    # Default DSYP convention: first numeric = cur, second = prev
                    cur, prev = n0, n1

        if cur is None or prev is None:
            return

        delta = cur - prev
        st.metric(label="Current vs Previous", value=f"{cur:,.4f}" if abs(cur) < 1 else f"{cur:,.0f}", delta=f"{delta:,.4f}" if abs(delta) < 1 else f"{delta:,.0f}")

        # Choose line color green/red based on delta
        line_color = "#22c55e" if delta >= 0 else "#ef4444"

        values = [prev, cur]
        labels = _labels_prev_cur(params)

        _render_sparkline(values, labels, color=line_color)
        return

    # C) TOP-N: bar chart (first col = label, second col = value)
    if "TOP" in template_key and len(cols) >= 2:
        label_col, value_col = cols[0], cols[1]
        # ensure value is numeric-like
        if _safe_float(df.iloc[0][value_col]) is not None:
            _render_bar(df, label_col, value_col, top_n=10)
        return

    # D) DIFF templates: show bar of diff by label (try detect diff columns)
    if "DIFF" in template_key and len(cols) >= 2:
        label_col = cols[0]
        # prefer diff column names
        diff_col = None
        for c in ["diff_value", "diff", "delta", "change_value", "change"]:
            if c in df.columns:
                diff_col = c
                break
        if diff_col is None:
            # fallback to second column
            diff_col = cols[1]

        # If diff is numeric, color by sign using Altair (optional)
        if _safe_float(df.iloc[0][diff_col]) is None:
            return

        alt = _try_import_altair()
        if alt is None:
            _render_bar(df, label_col, diff_col, top_n=10)
            return

        d = df[[label_col, diff_col]].copy().head(10)
        d["sign"] = d[diff_col].apply(lambda x: "up" if (_safe_float(x) or 0) >= 0 else "down")
        chart = (
            alt.Chart(d)
            .mark_bar()
            .encode(
                y=alt.Y(f"{label_col}:N", sort="-x", title=None),
                x=alt.X(f"{diff_col}:Q", title=None),
                color=alt.Color("sign:N", scale=alt.Scale(domain=["up","down"], range=["#22c55e","#ef4444"]), legend=None),
                tooltip=[label_col, diff_col],
            )
            .properties(height=min(760, 46 * len(d) + 120))
        )
        st.altair_chart(chart, use_container_width=True)
        return



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
    Parse month/year from Thai questions.

    Supports patterns like:
      - 'ยอดขายเดือน 7 ปี 2025'
      - 'ยอดขายเดือน 07 ปี 2025'
      - 'ยอดขายเดือน 7/2025'
      - 'ยอดขายเดือน มกราคม ปี 2025'
      - 'ยอดขายเดือน ม.ค. 2025'
      - 'มกราคม 2025'

    Return (year, month) or None.
    """
    if not q:
        return None
    q = q.strip()

    # 1) Numeric month patterns
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

    # 2) Thai month name patterns
    # Accept both full and abbreviated Thai month names.
    thai_month_map = {
        "มกราคม": 1, "ม.ค.": 1, "มค": 1, "ม.ค": 1, "มกรา": 1,
        # Add common colloquial forms (users often type these)
        "กุมภาพันธ์": 2, "ก.พ.": 2, "กพ": 2, "ก.พ": 2, "กุมภา": 2,
        "มีนาคม": 3, "มี.ค.": 3, "มีค": 3, "มี.ค": 3, "มีนา": 3,
        "เมษายน": 4, "เม.ย.": 4, "เมย": 4, "เม.ย": 4,
        "พฤษภาคม": 5, "พ.ค.": 5, "พค": 5, "พ.ค": 5,
        "มิถุนายน": 6, "มิ.ย.": 6, "มิย": 6, "มิ.ย": 6,
        "กรกฎาคม": 7, "ก.ค.": 7, "กค": 7, "ก.ค": 7,
        "สิงหาคม": 8, "ส.ค.": 8, "สค": 8, "ส.ค": 8,
        "กันยายน": 9, "ก.ย.": 9, "กย": 9, "ก.ย": 9,
        "ตุลาคม": 10, "ต.ค.": 10, "ตค": 10, "ต.ค": 10,
        "พฤศจิกายน": 11, "พ.ย.": 11, "พย": 11, "พ.ย": 11,
        "ธันวาคม": 12, "ธ.ค.": 12, "ธค": 12, "ธ.ค": 12, "ธันวา": 12,
    }

    # Normalize: remove extra spaces
    q_norm = re.sub(r"\s+", " ", q)

    def _norm_year(y: str) -> int:
        """Normalize year tokens. Support 2-digit years like '25' -> 2025."""
        yy = int(y)
        if yy < 100:
            return 2000 + yy
        return yy

    # Pattern: 'เดือน <month_name> ปี <yyyy>'
    m = re.search(r"เดือน\s*([ก-๙\.]{2,12})\s*ปี\s*(\d{2,4})", q_norm)
    if m:
        mn = m.group(1).strip()
        year = _norm_year(m.group(2))
        month = thai_month_map.get(mn)
        if month:
            return (year, month)

    # Pattern: 'เดือน <month_name> <yyyy>' (without 'ปี')
    m = re.search(r"เดือน\s*([ก-๙\.]{2,12})\s*(\d{2,4})", q_norm)
    if m:
        mn = m.group(1).strip()
        year = _norm_year(m.group(2))
        month = thai_month_map.get(mn)
        if month:
            return (year, month)

    # Pattern: '<month_name> ปี <yyyy>' (without 'เดือน')
    # NOTE: Thai word boundaries (\b) are unreliable; avoid them.
    m = re.search(r"([ก-๙\.]{2,12})\s*ปี\s*(\d{2,4})", q_norm)
    if m:
        mn = m.group(1).strip()
        year = _norm_year(m.group(2))
        month = thai_month_map.get(mn)
        if month:
            return (year, month)

    # Pattern: '<month_name> <yyyy>' (fallback)
    m = re.search(r"([ก-๙\.]{2,12})\s*(\d{2,4})", q_norm)
    if m:
        mn = m.group(1).strip()
        year = _norm_year(m.group(2))
        month = thai_month_map.get(mn)
        if month:
            return (year, month)

    return None


# --- Month/Year parsing (TH + EN) ---
EN_MONTH_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

def parse_month_year_from_question(q: str) -> Optional[Tuple[int, int]]:
    """Parse month/year from Thai OR English (and numeric forms). Return (year, month) or None."""
    if not q:
        return None

    # 1) Thai (includes numeric patterns too)
    th = parse_month_year_from_th_question(q)
    if th:
        return th

    q_norm = re.sub(r"\s+", " ", q.strip()).lower()

    # 2) Numeric patterns like 01/2025, 1/2025
    m = re.search(r"\b(\d{1,2})\s*/\s*(\d{4})\b", q_norm)
    if m:
        month = int(m.group(1))
        year = int(m.group(2))
        if 1 <= month <= 12:
            return (year, month)

    # 3) English month name patterns
    #    Examples:
    #      - Jan 2025
    #      - January 2025
    #      - Jan, 2025
    #      - Jan of 2025
    m = re.search(r"\b([a-z]{3,9})\b\s*(?:,\s*)?(?:of\s+)?(\d{4})\b", q_norm)
    if m:
        mn = m.group(1).strip().strip(".")
        year = int(m.group(2))
        month = EN_MONTH_MAP.get(mn)
        if month:
            return (year, month)

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


def forced_params_from_question(user_question: str) -> Optional[Dict[str, str]]:
    """If user_question contains an explicit month/year, return a dict with cur_start/cur_end/prev_start/prev_end (YYYY-MM-DD)."""
    parsed = parse_month_year_from_question(user_question)
    if not parsed:
        return None
    year, month = parsed
    cur_start, cur_end = month_range(year, month)
    prev_dt = date(year, month, 1) - relativedelta(months=1)
    prev_start, prev_end = month_range(prev_dt.year, prev_dt.month)
    return {"cur_start": cur_start, "cur_end": cur_end, "prev_start": prev_start, "prev_end": prev_end}


def forced_today_from_question(user_question: str) -> Optional[date]:
    """Return a 'today' date anchored inside the asked month, so dsyp_core computes cur/prev correctly."""
    parsed = parse_month_year_from_question(user_question)
    if not parsed:
        return None
    y, m = parsed
    # use mid-month to avoid edge cases
    return date(int(y), int(m), 15)

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
    parsed = parse_month_year_from_question(user_question)
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



def _extract_routing_features(user_question: str) -> Dict[str, Any]:
    q = _normalize_text(user_question)
    feats = {
        "q_norm": q,
        "want_compare": bool(_COMPARE_HINTS.search(q)),
        "hint_top": bool(_TOP_HINTS.search(q)),
        "hint_sales": bool(_SALES_HINTS.search(q)),
        "hint_credit": bool(_CREDIT_HINTS.search(q)),
        "hint_count": bool(_COUNT_HINTS.search(q)),
        "hint_branch": bool(_BRANCH_HINTS.search(q)),
        "hint_product": bool(_PRODUCT_HINTS.search(q)),
        "hint_campaign": bool(_CAMPAIGN_HINTS.search(q)),
        "hint_reject": bool(_REJECT_HINTS.search(q)),
        "hint_approval": bool(_APPROVAL_HINTS.search(q)),
        "hint_cancel": bool(_CANCEL_HINTS.search(q)),
        "hint_leadtime": bool(_LEADTIME_HINTS.search(q)),
    }
    # ambiguous wording patterns
    feats["has_ambiguous_count_word"] = bool(re.search(r"\bจำนวน\b", q)) and (not feats["hint_credit"]) and (not re.search(r"สัญญา|contract", q, flags=re.IGNORECASE))
    return feats


def _score_template_row(
    template_key: str,
    qb_row: Dict[str, Any],
    feats: Dict[str, Any],
    fuzzy_score: float,
) -> float:
    """Score how well a template matches the question, using multiple signals.

    This is intentionally conservative to avoid wrong-template answers.
    """
    key_u = str(template_key or "").upper()
    score = 0.0

    # Base fuzzy (low weight; helps only when close)
    score += 0.35 * float(fuzzy_score or 0.0)

    # Compare intent
    if feats.get("want_compare"):
        if "_VS_PREV" in key_u:
            score += 1.2
        else:
            score -= 0.4

    # Domain/metric intent hints
    if feats.get("hint_sales") and key_u.startswith("SALES_"):
        score += 0.8
    if feats.get("hint_credit") and key_u.startswith("CREDIT_"):
        score += 0.8

    # Count intent (credits typically)
    if feats.get("hint_count"):
        if "_CNT" in key_u or "COUNT" in key_u or "CONTRACT" in key_u:
            score += 0.7
        # count word with sales total is often wrong; be conservative
        if key_u.startswith("SALES_") and "TOTAL" in key_u and feats.get("hint_credit"):
            score -= 0.3

    # Specific intents
    # Generic TOP/ranking intent
    if feats.get("hint_top") and "_TOP" in key_u:
        score += 0.55

    if feats.get("hint_branch") and "BRANCH" in key_u:
        score += 0.9
    if feats.get("hint_product") and "PRODUCT" in key_u:
        score += 0.9
    if feats.get("hint_campaign") and "CAMPAIGN" in key_u:
        score += 0.9
    if feats.get("hint_reject") and "REJECT" in key_u:
        score += 1.0
    if feats.get("hint_approval") and "APPROVAL" in key_u:
        score += 1.0
    if feats.get("hint_cancel") and "CANCELLATION" in key_u:
        score += 1.0
    if feats.get("hint_leadtime") and "LEADTIME" in key_u:
        score += 1.0

    # Metadata nudges (if present)
    compare_type = str(qb_row.get("compare_type") or "").upper()
    intent_type = str(qb_row.get("intent_type") or "").upper()
    dimension = str(qb_row.get("dimension") or "").upper()
    main_table = str(qb_row.get("main_table") or "").upper()

    if feats.get("want_compare") and ("VS" in compare_type or "PREV" in compare_type):
        score += 0.4
    if feats.get("hint_sales") and ("SALES" in intent_type or "SALES" in main_table):
        score += 0.2
    if feats.get("hint_credit") and ("CREDIT" in intent_type or "CREDIT" in main_table):
        score += 0.2
    if feats.get("hint_branch") and "BRANCH" in dimension:
        score += 0.2
    if feats.get("hint_product") and "PRODUCT" in dimension:
        score += 0.2
    if feats.get("hint_campaign") and "CAMPAIGN" in dimension:
        score += 0.2

    return float(score)


def smart_local_route(user_question: str, question_bank_df: pd.DataFrame) -> Dict[str, Any]:
    """Return routing result dict.

    May return:
      - {"mode": "clarify", "question": ..., "options": [...]}
      - {"mode": "ok", "sql_template_key": ..., "router_mode": ..., ...}
      - {"mode": "low_conf"}  (caller may fall back to LLM)
    """
    if question_bank_df is None or question_bank_df.empty:
        return {"mode": "low_conf"}

    feats = _extract_routing_features(user_question)

    # Pre-compute fuzzy similarity vs QB questions
    uq = feats["q_norm"]
    scored = []
    for _, r in question_bank_df.iterrows():
        key = str(r.get("sql_template_key") or "").strip()
        if not key:
            continue
        q_th = str(r.get("question_text_th") or "")
        q_en = str(r.get("question_text_en") or "")
        cand = q_th if q_th else q_en
        fuzzy = difflib.SequenceMatcher(None, uq, _normalize_text(cand)).ratio() if cand else 0.0
        s = _score_template_row(key, r.to_dict(), feats, fuzzy)
        scored.append((s, key, fuzzy, cand))

    if not scored:
        return {"mode": "low_conf"}

    scored.sort(key=lambda x: x[0], reverse=True)
    best_s, best_key, best_fuzzy, best_q = scored[0]
    second_s = scored[1][0] if len(scored) > 1 else -999

    # Handle common ambiguity: "จำนวน" without clear credit/sales context
    if feats.get("has_ambiguous_count_word"):
        # Offer two safest defaults: sales total vs credit contract count (if they exist)
        keys = set(question_bank_df["sql_template_key"].dropna().astype(str).unique().tolist())
        opt = []
        # If user clearly wants comparison, prefer VS_PREV variants where available.
        if feats.get("want_compare"):
            if "SALES_TOTAL_CURR_VS_PREV" in keys:
                opt.append({"label": "ยอดขาย (เทียบเดือนก่อน)", "template_key": "SALES_TOTAL_CURR_VS_PREV"})
            if "CREDIT_CONTRACT_CNT_VS_PREV" in keys:
                opt.append({"label": "จำนวนสัญญา (เทียบเดือนก่อน)", "template_key": "CREDIT_CONTRACT_CNT_VS_PREV"})
        # Fallback to current-period variants
        if "SALES_TOTAL_CURR" in keys:
            opt.append({"label": "ยอดขาย (บาท)", "template_key": "SALES_TOTAL_CURR"})
        if "CREDIT_CONTRACT_CNT" in keys:
            opt.append({"label": "จำนวนสัญญา (สัญญา)", "template_key": "CREDIT_CONTRACT_CNT"})
        if len(opt) >= 2:
            return {
                "mode": "clarify",
                "question": user_question,
                "message": "คำว่า ‘จำนวน’ ยังไม่ชัดเจน: ต้องการดูยอดขาย หรือจำนวนสัญญา?",
                "options": opt[:2],
            }

    # Confidence rule (conservative)
    # - best score must be reasonably high
    # - and must separate from second
    if best_s >= 1.35 and (best_s - second_s) >= 0.25:
        return {
            "mode": "ok",
            "sql_template_key": best_key,
            "router_mode": "local_scoring",
            "score": round(float(best_s), 3),
            "fuzzy": round(float(best_fuzzy), 3),
            "matched_question": best_q,
        }

    # If compare intent is strong, try to restrict to *_VS_PREV keys then re-pick
    if feats.get("want_compare"):
        vs_scored = [x for x in scored if "_VS_PREV" in str(x[1]).upper()]
        if vs_scored:
            vs_scored.sort(key=lambda x: x[0], reverse=True)
            vs_best_s, vs_best_key, vs_best_fuzzy, vs_best_q = vs_scored[0]
            if vs_best_s >= 1.15:
                return {
                    "mode": "ok",
                    "sql_template_key": vs_best_key,
                    "router_mode": "local_scoring_vs_prev",
                    "score": round(float(vs_best_s), 3),
                    "fuzzy": round(float(vs_best_fuzzy), 3),
                    "matched_question": vs_best_q,
                }

    return {"mode": "low_conf", "best_guess": best_key, "best_score": round(float(best_s), 3)}



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


def _month_abbr_label_from_iso(start_iso: Optional[str]) -> Optional[str]:
    """Return label like 'Jan-25' from YYYY-MM-DD."""
    if not start_iso:
        return None
    try:
        y, m, _ = start_iso.split("-")
        y = int(y); m = int(m)
        if 1 <= m <= 12:
            _EN_ABBR = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            return f"{_EN_ABBR[m-1]}-{y%100:02d}"
    except Exception:
        return None
    return None

def _labels_prev_cur(params: Optional[Dict[str, Any]]) -> List[str]:
    """Labels for sparkline axis: prev month then current month."""
    if not params:
        return ["Prev", "Cur"]
    prev = _month_abbr_label_from_iso(params.get("prev_start"))
    cur  = _month_abbr_label_from_iso(params.get("cur_start"))
    if prev and cur:
        return [prev, cur]
    return ["Prev", "Cur"]

def _month_label(user_question: str, params: Optional[Dict[str, Any]] = None) -> str:
    """Prefer month/year parsed from Thai question; else fallback to params['cur_start'] or today."""
    parsed = parse_month_year_from_question(user_question or "")
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


def _norm_colname(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def guess_label_col(df: pd.DataFrame, preferred: List[str]) -> Optional[str]:
    """Pick a label/dimension column from a dataframe."""
    if df is None or df.empty:
        return None

    cols = list(df.columns)
    norm_lookup = {_norm_colname(c): c for c in cols}

    # 1) preferred exact / normalized match
    for p in preferred or []:
        p_norm = _norm_colname(p)
        if p in cols:
            return p
        if p_norm in norm_lookup:
            return norm_lookup[p_norm]

    # 2) any non-numeric column
    for c in cols:
        try:
            if not pd.api.types.is_numeric_dtype(df[c]):
                return c
        except Exception:
            continue
    return cols[0] if cols else None

def guess_value_col(df: pd.DataFrame, preferred: List[str]) -> Optional[str]:
    """Pick a numeric/value column from a dataframe."""
    if df is None or df.empty:
        return None

    cols = list(df.columns)
    norm_lookup = {_norm_colname(c): c for c in cols}

    # 1) preferred exact / normalized match
    for p in preferred or []:
        p_norm = _norm_colname(p)
        if p in cols:
            return p
        if p_norm in norm_lookup:
            return norm_lookup[p_norm]

    # 2) first numeric column
    for c in cols:
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
        except Exception:
            continue
    return None

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

            # Some sources may return pct as a fraction (e.g., 0.117 instead of 11.7)
            try:
                if diff_pct is not None and abs(float(diff_pct)) <= 1.0:
                    diff_pct = float(diff_pct) * 100.0
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
                top_n = _safe_int(qb_row.get("top_n"), 10)
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
# 4) UI  (Query to Insight - AI Assistant)
# =========================
APP_VERSION = "v2026-01-09-q2i"

st.set_page_config(page_title="Query to Insight - AI Assistant", layout="wide")

# --- Minimal CSS to mimic "AI assistant" landing ---
st.markdown(
    """
    <style>
      /* tighten default padding */
      .block-container { padding-top: 2.0rem; padding-bottom: 3rem; max-width: 980px; margin-left: 2.25rem; margin-right: auto; }
      /* hide hamburger footer spacing a bit */
      footer {visibility: hidden;}
      /* make buttons look like pills for suggested questions */
      div.stButton > button {
        border-radius: 999px !important;
        padding: 0.35rem 0.85rem !important;
        font-size: 0.95rem !important;
      }
      /* input + arrow button alignment */
      .q2i-input label { display:none !important; }
      /* make input text a bit larger */
      div[data-testid="stTextInput"] input { font-size: 1.05rem; padding-top: 0.6rem; padding-bottom: 0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Sidebar: keep only admin settings (optional) ----
with st.sidebar:
    st.caption(f"APP_VERSION: {APP_VERSION}")
    st.divider()
    with st.expander("Admin settings", expanded=False):
        api_key = st.text_input("Google Gemini API Key", type="password")
        model_name = st.text_input("Model name", value="gemini-2.0-flash")
        show_debug = st.checkbox("Show debug (router/sql/result)", value=False)
        show_chart = st.checkbox("Show chart (optional)", value=True)
    st.caption("Tip: End users don't need to open this sidebar.")

# ---- Ensure assets + data are loaded (no user upload) ----
def ensure_assets_data_loaded() -> None:
    if st.session_state.get("assets_data_loaded"):
        return

    loaded = []
    if SALES_CSV_PATH.exists():
        rows = _load_csv_path_to_table(st.session_state.conn, SALES_CSV_PATH, "SALES_MASTER")
        loaded.append(f"SALES_MASTER ({rows} rows)")
    if CREDIT_CSV_PATH.exists():
        rows = _load_csv_path_to_table(st.session_state.conn, CREDIT_CSV_PATH, "CREDIT_CONTRACT")
        loaded.append(f"CREDIT_CONTRACT ({rows} rows)")

    if not loaded:
        st.error(
            "❌ No raw CSV data found in /assets. Expected at least one of: "
            f"{SALES_CSV_PATH.name}, {CREDIT_CSV_PATH.name}"
        )
        st.stop()

    st.session_state.assets_data_loaded = True
    st.session_state.loaded_tables = loaded

# Validate required asset files exist
missing_assets = [str(p) for p in [QB_PATH, TPL_PATH] if not Path(p).exists()]
if missing_assets:
    st.error("❌ Missing required asset files in /assets: " + ", ".join(missing_assets))
    st.info("Please ensure your GitHub repo contains assets/question_bank.xlsx and assets/sql_templates_with_placeholder.xlsx")
    st.stop()

ensure_assets_data_loaded()

# Auto schema (no user action needed)
schema_doc = sqlite_schema_doc(st.session_state.conn)

# Load KB + templates
question_bank_df = read_xlsx(QB_PATH)
templates_df = read_xlsx(TPL_PATH)

# ---- UI state ----
if "asof_date" not in st.session_state:
    st.session_state.asof_date = date.today() - relativedelta(days=1)

if "last_result" not in st.session_state:
    st.session_state.last_result = None  # dict with answer/sql/df/template_key/router_out

# ---- Header (centered) ----
st.markdown(
    """
    <div style="text-align:left; margin-top: 0.25rem;">
      <div style="font-size: 2.05rem; font-weight: 750; letter-spacing: -0.01em;">🐱 Query to Insight - AI Assistant</div>
      <div style="color: #6b7280; margin-top: 0.25rem; font-size: 1.0rem;">
        Ask a question and get an answer grounded in your database.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ---- Data update (read-only) ----
st.caption(f"Data update: **{st.session_state.asof_date}**")

st.write("")

# ---- Suggested questions (shortcut buttons) ----
SUGGESTED = [
    ("ยอดขายเดือนนี้เท่าไร", "Sales MTD"),
    ("ยอดขายเดือนนี้เทียบเดือนก่อน", "Sales vs Prev"),
    ("จำนวนสัญญาเครดิตเดือนนี้เท่าไร", "Credit count"),
]

st.markdown("<div style='text-align:left; color:#6b7280; margin-bottom: 0.35rem;'>Shortcuts</div>", unsafe_allow_html=True)

btn_cols = st.columns(len(SUGGESTED))
for i, (q, _tag) in enumerate(SUGGESTED):
    with btn_cols[i]:
        if st.button(q, key=f"suggest_{i}", use_container_width=True):
            # Important: text_input with a fixed key will ignore the "value="
            # parameter once the key exists in session_state. Therefore, for
            # shortcuts we must write directly to the widget's key.
            st.session_state["q2i_question"] = q
            st.session_state["run_now"] = True
            st.rerun()

st.write("")

# ---- Ask a question input (center style) ----
# Use a 2-column row: wide input + arrow button
qrow1, qrow2 = st.columns([10, 1])
with qrow1:
    user_question = st.text_input(
        "Ask a question…",
        value=st.session_state.get("q2i_question", ""),
        key="q2i_question",
        label_visibility="collapsed",
        placeholder="Ask a question…",
    )
with qrow2:
    run_clicked = st.button("➤", type="primary", use_container_width=True)

# run triggers
run_now = bool(st.session_state.get("run_now", False)) or run_clicked
st.session_state["run_now"] = False



# =========================
# 5) Time-intent engine (latest month / last N months / YoY YTD)  [Local, no-LLM]
# =========================

_PH_RE = re.compile(r":([a-zA-Z_][a-zA-Z0-9_]*)")

def extract_placeholders(sql_template: str) -> set:
    return set(_PH_RE.findall(sql_template or ""))

def pick_template_sql_column(templates_df: pd.DataFrame) -> str:
    for c in ["template_sql", "sql_template", "template", "sql", "Template SQL"]:
        if c in templates_df.columns:
            return c
    raise ValueError("No SQL template column found")

def get_template_sql(templates_df: pd.DataFrame, key: str) -> str:
    m = templates_df[templates_df["sql_template_key"].astype(str) == str(key)]
    if m.empty:
        raise KeyError(f"Template key not found: {key}")
    col = pick_template_sql_column(templates_df)
    return str(m.iloc[0][col])

def get_qb_row(question_bank_df: pd.DataFrame, key: str) -> pd.Series:
    m = question_bank_df[question_bank_df["sql_template_key"].astype(str) == str(key)]
    if m.empty:
        raise KeyError(f"Question bank row not found for key: {key}")
    return m.iloc[0]

_LOGICAL_TO_PHYSICAL_TABLE = {
    "SALES_MASTER": "SALES_MASTER",
    "CREDIT_CONTRACT": "CREDIT_CONTRACT",
}

def _month_start_end(y: int, m: int) -> tuple[str, str]:
    return month_range(int(y), int(m))

def _shift_month(y: int, m: int, delta_months: int) -> tuple[int, int]:
    d = date(int(y), int(m), 1) + relativedelta(months=int(delta_months))
    return d.year, d.month

def _latest_month_range(conn: sqlite3.Connection, table: str, date_field: str) -> tuple[str, str]:
    # Cache per table+field to keep UI snappy
    cache = st.session_state.setdefault("_latest_month_cache", {})
    key = f"{table}::{date_field}"
    if key in cache:
        return cache[key]

    # Handle datetime text by taking first 10 chars and normalizing / to -
    sql = f"""
    SELECT MAX(
      DATE(
        CASE
          WHEN substr({date_field},5,1)='-' THEN substr({date_field},1,10)
          WHEN substr({date_field},5,1)='/' THEN replace(substr({date_field},1,10), '/', '-')
          WHEN substr({date_field},3,1)='/' THEN substr({date_field},7,4)||'-'||substr({date_field},4,2)||'-'||substr({date_field},1,2)
          WHEN substr({date_field},3,1)='-' THEN substr({date_field},7,4)||'-'||substr({date_field},4,2)||'-'||substr({date_field},1,2)
          WHEN LENGTH({date_field})>=8 AND {date_field} GLOB '[0-9]*' THEN substr({date_field},1,4)||'-'||substr({date_field},5,2)||'-'||substr({date_field},7,2)
          ELSE NULL
        END
      )
    ) AS max_dt
    FROM {table}
    WHERE {date_field} IS NOT NULL AND TRIM({date_field}) <> ''
    """
    try:
        r = conn.execute(sql).fetchone()
        max_dt = (r[0] if r else None)
    except Exception:
        max_dt = None

    if not max_dt:
        # fallback to "today" month
        t = date.today()
        start, end = _month_start_end(t.year, t.month)
        cache[key] = (start, end)
        return start, end

    try:
        y, m, _ = str(max_dt).split("-")
        start, end = _month_start_end(int(y), int(m))
    except Exception:
        t = date.today()
        start, end = _month_start_end(t.year, t.month)

    cache[key] = (start, end)
    return start, end

def parse_rolling_n_months(q: str) -> int | None:
    if not q:
        return None
    s = q.lower()
    m = re.search(r"ย้อนหลัง\s*(\d{1,2})\s*เดือน", s)
    if m:
        try:
            n = int(m.group(1))
            return n if 1 <= n <= 24 else None
        except Exception:
            return None
    m = re.search(r"(?:last|past|previous|trailing)\s*(\d{1,2})\s*months", s)
    if m:
        try:
            n = int(m.group(1))
            return n if 1 <= n <= 24 else None
        except Exception:
            return None
    return None

def parse_month_name_with_last_year(q: str) -> tuple[int,int] | None:
    # e.g., "ธันวาปีที่แล้ว" / "Dec last year"
    if not q:
        return None
    qn = q.strip().lower()

    # Thai month names
    th_map = {
        "มกราคม":1,"ม.ค":1,"ม.ค.":1,"มกรา":1,
        "กุมภาพันธ์":2,"ก.พ":2,"ก.พ.":2,"กุมภา":2,
        "มีนาคม":3,"มี.ค":3,"มี.ค.":3,"มีนา":3,
        "เมษายน":4,"เม.ย":4,"เม.ย.":4,
        "พฤษภาคม":5,"พ.ค":5,"พ.ค.":5,
        "มิถุนายน":6,"มิ.ย":6,"มิ.ย.":6,
        "กรกฎาคม":7,"ก.ค":7,"ก.ค.":7,
        "สิงหาคม":8,"ส.ค":8,"ส.ค.":8,
        "กันยายน":9,"ก.ย":9,"ก.ย.":9,
        "ตุลาคม":10,"ต.ค":10,"ต.ค.":10,
        "พฤศจิกายน":11,"พ.ย":11,"พ.ย.":11,
        "ธันวาคม":12,"ธ.ค":12,"ธ.ค.":12,"ธันวา":12,
    }

    if "ปีที่แล้ว" in qn:
        # try find a month token
        for k, m in th_map.items():
            if k in qn:
                y = (st.session_state.asof_date.year if isinstance(st.session_state.get('asof_date'), date) else date.today().year) - 1
                return y, m
        # English month token
        for k, m in EN_MONTH_MAP.items():
            if k in qn:
                y = (st.session_state.asof_date.year if isinstance(st.session_state.get('asof_date'), date) else date.today().year) - 1
                return y, m
    return None

def detect_time_flags(q: str) -> dict:
    s = (q or "").lower()
    return {
        "this_month": any(x in s for x in ["เดือนนี้","เดือนปัจจุบัน","this month","current month"]),
        "latest": any(x in s for x in ["เดือนล่าสุด","ล่าสุด","most recent","latest month","recent month"]),
        "last_year": any(x in s for x in ["ปีที่แล้ว","last year","previous year"]),
        "yoy": any(x in s for x in ["yoy","ปีนี้เทียบปีที่แล้ว","เทียบปีที่แล้ว","compare last year"]),
    }

def compute_time_params(user_question: str, qb_row: pd.Series, template_key: str, conn: sqlite3.Connection) -> dict:
    """Return a dict of date params (cur/prev/range) grounded in DB, using default anchor=latest."""
    # Domain table/date_field
    main_table = str(qb_row.get("main_table") or "").strip()
    table = _LOGICAL_TO_PHYSICAL_TABLE.get(main_table, main_table) or main_table
    date_field = str(qb_row.get("date_field") or "").strip() or ("order_datetime" if "SALES" in template_key else "application_date")

    flags = detect_time_flags(user_question)

    # 1) explicit month/year in question
    parsed = parse_month_year_from_question(user_question)
    if not parsed:
        parsed = parse_month_name_with_last_year(user_question)

    # Determine anchor month start/end
    if parsed:
        ay, am = parsed
        anchor_start, anchor_end = _month_start_end(ay, am)
        anchor_mode = "explicit"
    elif flags.get("this_month"):
        t = st.session_state.asof_date if isinstance(st.session_state.get('asof_date'), date) else date.today()
        anchor_start, anchor_end = _month_start_end(t.year, t.month)
        anchor_mode = "current"
    else:
        # default latest month in data
        anchor_start, anchor_end = _latest_month_range(conn, table, date_field)
        anchor_mode = "latest"

    # prev month
    py, pm = _shift_month(int(anchor_start[:4]), int(anchor_start[5:7]), -1)
    prev_start, prev_end = _month_start_end(py, pm)

    # rolling N
    n = parse_rolling_n_months(user_question)
    if not n:
        try:
            n = int(qb_row.get("window_n_default") or 0) or None
        except Exception:
            n = None

    # Time intent from QB (preferred)
    time_intent = str(qb_row.get("time_intent") or "").upper().strip()

    out = {
        "anchor_mode": anchor_mode,
        "anchor_start": anchor_start,
        "anchor_end": anchor_end,
        "cur_start": anchor_start,
        "cur_end": anchor_end,
        "prev_start": prev_start,
        "prev_end": prev_end,
    }

    # Rolling window params
    if time_intent in {"ROLLING_N_MONTHS", "ROLLING", "TREND_LAST_N_MONTHS"} or (n is not None and "TREND" in template_key):
        if n is None:
            n = 6
        # inclusive N months ending at anchor month: start = anchor_start - (n-1) months
        sy, sm = _shift_month(int(anchor_start[:4]), int(anchor_start[5:7]), -(n-1))
        range_start, _ = _month_start_end(sy, sm)
        range_end = anchor_end
        out.update({"window_n": n, "range_start": range_start, "range_end": range_end})

    # YoY YTD
    if time_intent in {"YOY_YTD", "YOY_YTD_BY_MONTH", "YOY"} or flags.get("yoy"):
        # YTD end aligned to anchor_end
        cur_end = anchor_end
        cur_year = int(anchor_start[:4])
        cur_start = f"{cur_year}-01-01"
        prev_year = cur_year - 1
        prev_start = f"{prev_year}-01-01"
        # prev_end aligned by shifting cur_end - 1 year
        try:
            ce = date.fromisoformat(cur_end)
            pe = ce.replace(year=ce.year-1)
            prev_end = pe.isoformat()
        except Exception:
            prev_end = f"{prev_year}-{anchor_end[5:7]}-{anchor_end[8:10]}"
        out.update({"cur_start": cur_start, "cur_end": cur_end, "prev_start": prev_start, "prev_end": prev_end})

    # Yearly trend (default last 5 years up to anchor year)
    if time_intent in {"YEARLY_TREND", "TREND_BY_YEAR"} or template_key == "SALES_TREND_BY_YEAR":
        end_year = int(anchor_start[:4])
        start_year = end_year - 4
        out.update({
            "range_start": f"{start_year}-01-01",
            "range_end": f"{end_year+1}-01-01",
        })

    # latest-only shortcut
    if time_intent in {"LATEST_AVAILABLE", "LATEST_MONTH"} or template_key == "SALES_TOTAL_LATEST":
        out.update({"cur_start": anchor_start, "cur_end": anchor_end})

    return out


def build_sql_from_key(template_key: str, user_question: str, question_bank_df: pd.DataFrame, templates_df: pd.DataFrame, conn: sqlite3.Connection) -> tuple[str, dict, pd.Series]:
    qb_row = get_qb_row(question_bank_df, template_key)
    tpl_sql = get_template_sql(templates_df, template_key)

    # base params
    main_table = str(qb_row.get("main_table") or "").strip()
    table_name = _LOGICAL_TO_PHYSICAL_TABLE.get(main_table, main_table) or main_table

    params = {
        "table_name": table_name,
        "date_field": qb_row.get("date_field", None),
        "metric_expr": qb_row.get("metric_expression", None),
        "dimension": qb_row.get("dimension", None),
        "dim_field": qb_row.get("dimension", None),
        "group_by_field": qb_row.get("dimension", None),
        "top_n": _safe_int(qb_row.get("top_n"), 10),
        "filter_clause": qb_row.get("filter_hint", "AND 1=1"),
    }

    # compute time params (grounded)
    tparams = compute_time_params(user_question, qb_row, template_key, conn)
    params.update(tparams)

    # ensure filter_clause shape
    if not str(params.get("filter_clause") or "").strip():
        params["filter_clause"] = "AND 1=1"

    # Render template using dsyp_core.render_template (keeps guardrails)
    from dsyp_core import render_template
    sql = render_template(tpl_sql, params)
    return sql, params, qb_row


def _run_one_question(user_question: str):
    if not user_question or not user_question.strip():
        st.warning("Please type a question.")
        return

    # If UI forced a template choice (clarify flow), bypass routing
    forced_key = st.session_state.pop('force_template_key', None)
    if forced_key:
        template_key = str(forced_key).strip()
        router_out = {'sql_template_key': template_key, 'router_mode': 'forced_by_user'}
        # Build SQL with time-intent (latest/rolling/yoy) grounded in DB
        try:
            final_sql, params, qb_row = build_sql_from_key(
                template_key=template_key,
                user_question=user_question,
                question_bank_df=question_bank_df,
                templates_df=templates_df,
                conn=st.session_state.conn,
            )
        except Exception as e:
            st.session_state['last_result'] = {
                'question': user_question,
                'answer': f'SQL build error: {e}',
                'template_key': template_key,
                'router_out': router_out,
                'sql': '',
                'df': pd.DataFrame(),
                'params': {},
            }
            st.error(f'SQL build error: {e}')
            return

        params = params or {}
        st.session_state['last_params'] = params
        st.session_state['last_user_question'] = user_question

        display_sql = final_sql
        sql_exec = strip_sql_comments(final_sql)

        ok, msg = is_safe_readonly_sql(sql_exec, st.session_state.conn)
        if not ok:
            st.session_state['last_result'] = {
                'question': user_question,
                'answer': f'SQL blocked: {msg}',
                'template_key': template_key,
                'router_out': router_out,
                'sql': display_sql,
                'df': pd.DataFrame(),
                'params': params or {},
                'blocked': True,
                'blocked_reason': msg,
            }
            st.error(f'SQL blocked: {msg}')
            if show_debug:
                st.code(display_sql, language='sql')
            return

        df = pd.read_sql_query(sql_exec, st.session_state.conn)
        df_canon = canonical_compare_df(template_key, st.session_state.conn, params)
        if df_canon is not None and not df_canon.empty:
            df = df_canon

        answer_text = hybrid_answer(
            model_name=model_name,
            user_question=user_question,
            template_key=template_key,
            df=df,
            question_bank_df=question_bank_df,
        )

        st.session_state.last_result = {
            'question': user_question,
            'answer': answer_text,
            'template_key': template_key,
            'router_out': router_out,
            'sql': display_sql,
            'df': df,
            'params': params,
        }
        return

    # Configure Gemini only if key provided
    if api_key:
        try:
            genai.configure(api_key=api_key)
        except Exception:
            pass
    # Route: smart local scoring first (more flexible than fuzzy)
    router_out = {}
    try:
        smart = smart_local_route(user_question, question_bank_df)
    except Exception:
        smart = {"mode": "low_conf"}

    if smart.get("mode") == "clarify":
        # Persist clarify state to UI
        st.session_state.last_result = {
            "question": user_question,
            "answer": smart.get("message") or "Need clarification",
            "template_key": "__CLARIFY__",
            "router_out": smart,
            "sql": "",
            "df": pd.DataFrame(),
            "params": {},
        }
        return

    if smart.get("mode") == "ok":
        router_out = smart
    else:
        # fallback: local fuzzy
        local_key, local_score, local_q = _local_best_template(user_question, question_bank_df)
        if local_key and local_score >= 0.72:
            router_out = {"sql_template_key": local_key, "router_mode": "local_fuzzy", "score": round(local_score, 3), "matched_question": local_q}
        else:
            if not api_key:
                st.error("This question needs LLM routing, but API key is missing. Please open sidebar > Admin settings and add the key.")
                return
            router_out = call_router_llm(
                user_question=user_question,
                question_bank_df=question_bank_df,
                schema_doc=schema_doc,
                model_name=model_name,
            )

    template_key = str(router_out.get("sql_template_key", "") or "").strip()
    allowed = set(question_bank_df["sql_template_key"].dropna().astype(str).unique().tolist())
    if template_key not in allowed:
        fb_key, fb_score, fb_q = _local_best_template(user_question, question_bank_df)
        if fb_key and fb_score >= 0.55:
            template_key = fb_key
            router_out = {"sql_template_key": fb_key, "router_mode": "fallback_local_fuzzy", "score": round(fb_score, 3), "matched_question": fb_q}
        else:
            st.error("Question is out of scope (question_bank).")
            if show_debug:
                st.json({"router_out": router_out, "fallback_score": round(fb_score or 0, 3)})
            return
    # Build SQL with time-intent (latest/rolling/yoy) grounded in DB
    try:
        final_sql, params, qb_row = build_sql_from_key(
            template_key=template_key,
            user_question=user_question,
            question_bank_df=question_bank_df,
            templates_df=templates_df,
            conn=st.session_state.conn,
        )
    except Exception as e:
        st.session_state["last_result"] = {
            "question": user_question,
            "answer": f"SQL build error: {e}",
            "template_key": template_key,
            "router_out": router_out,
            "sql": "",
            "df": pd.DataFrame(),
            "params": {},
        }
        st.error(f"SQL build error: {e}")
        return

    params = params or {}
    st.session_state["last_params"] = params
    st.session_state["last_user_question"] = user_question


    display_sql = final_sql
    sql_exec = strip_sql_comments(final_sql)

    ok, msg = is_safe_readonly_sql(sql_exec, st.session_state.conn)
    if not ok:
        # Still persist debug evidence so Show debug can display the SQL and reason
        st.session_state["last_result"] = {
            "question": user_question,
            "answer": f"SQL blocked: {msg}",
            "template_key": template_key,
            "router_out": router_out,
            "sql": display_sql,
            "df": pd.DataFrame(),
            "params": params or {},
            "blocked": True,
            "blocked_reason": msg,
        }
        st.error(f"SQL blocked: {msg}")
        if show_debug:
            st.code(display_sql, language="sql")
        return

    df = pd.read_sql_query(sql_exec, st.session_state.conn)

    # Canonical compare override (ensure correct cur/prev metric definition when needed)
    df_canon = canonical_compare_df(template_key, st.session_state.conn, params)
    if df_canon is not None and not df_canon.empty:
        df = df_canon

    answer_text = hybrid_answer(
        model_name=model_name,
        user_question=user_question,
        template_key=template_key,
        df=df,
        question_bank_df=question_bank_df,
    )

    st.session_state.last_result = {
        "question": user_question,
        "answer": answer_text,
        "template_key": template_key,
        "router_out": router_out,
        "sql": display_sql,
        "df": df,
        "params": params,
    }

if run_now:
    _run_one_question(st.session_state.get("q2i_question", ""))

# ---- Result area ----
res = st.session_state.get("last_result")
if res:
    st.write("")
    st.markdown("### Answer")
    st.markdown(f"**Q:** {res['question']}")
    st.markdown(f"**A:** {res['answer']}")

    # If router asked for clarification, show choice buttons
    if res.get('template_key') == '__CLARIFY__':
        opts = (res.get('router_out') or {}).get('options') or []
        if opts:
            st.info('Please select one option:')
            cols = st.columns(min(3, len(opts)))
            for i, opt in enumerate(opts):
                with cols[i % len(cols)]:
                    if st.button(str(opt.get('label') or opt.get('template_key')), key=f'clarify_opt_{i}', use_container_width=True):
                        st.session_state['force_template_key'] = opt.get('template_key')
                        st.session_state['run_now'] = True
                        st.rerun()

    # Optional visuals (keep lightweight)
    try:
        render_optional_visuals(res["template_key"], res["df"], user_question=res["question"], params=res.get("params"), show_chart=show_chart)
    except Exception:
        pass

    if show_debug:
        st.write("")
        with st.expander("Evidence (SQL / Result)", expanded=False):
            st.caption(f"template_key: {res['template_key']}")
            st.code(res["sql"], language="sql")
            st.dataframe(res["df"].head(200), use_container_width=True)
            st.json(res["router_out"])
else:
    st.caption("Try one of the shortcuts above, or type your own question.")

