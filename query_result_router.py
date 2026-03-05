# query_result_router.py
import os, glob
import pandas as pd
from typing import Optional

# ====== 0050 十大成分股（可自行更新）======
TOP10_0050 = [
    ("2330", "台積電"), ("2317", "鴻海"), ("2454", "聯發科"), ("2308", "台達電"),
    ("2881", "富邦金"), ("2882", "國泰金"), ("2303", "聯電"), ("2382", "廣達"),
    ("2412", "中華電"), ("2891", "中信金"),
]

# ====== 代號↔名稱 對應 ======
TICKER_NAME = {
    "2330":"台積電","2317":"鴻海","2454":"聯發科","2308":"台達電",
    "2881":"富邦金","2882":"國泰金","2303":"聯電","2382":"廣達",
    "2412":"中華電","2891":"中信金",
}
NAME_TICKER = {v: k for k, v in TICKER_NAME.items()}

# ====== 分數 → 文字（僅當沒有 result 欄時才用）======
def score_to_text(x: Optional[float], eps: float = 0.25) -> str:
    if x is None or pd.isna(x): return "無資料"
    if x > eps:  return "看好"
    if x < -eps: return "看壞"
    return "中立"

# ====== 英文標籤（只認 good/neutral/bad）→ 中文 ======
def label_to_text(val: Optional[str]) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "無資料"
    s = str(val).strip().lower()
    if s == "good":    return "看好"
    if s == "neutral": return "中立"
    if s == "bad":     return "看壞"
    # 其他值一律原樣回傳（避免錯判）
    return str(val)

def _pick_latest_result_csv() -> Optional[str]:
    # 允許 ALL_STOCK_RESULT.csv 或 ALL_STOCK_RESULT(1).csv…
    cands = sorted(glob.glob("ALL_STOCK_RESULT*.csv"), key=os.path.getmtime, reverse=True)
    return cands[0] if cands else None

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    對齊欄位名稱為：
    ['ticker','sent_score','tech_score','foreign_score','total_score','result']
    """
    cols_lower = {c.lower(): c for c in df.columns}
    rename = {}

    def pick(keys, target):
        for k in keys:
            if k in cols_lower:
                rename[cols_lower[k]] = target
                return

    pick(["ticker","stock","代號"], "ticker")
    pick(["sent_score","sentiment_score"], "sent_score")
    pick(["tech_score"], "tech_score")
    pick(["foreign_score"], "foreign_score")
    pick(["total_score","score"], "total_score")
    pick(["result","label","final"], "result")

    if rename: df = df.rename(columns=rename)

    for need in ["ticker","sent_score","tech_score","foreign_score","total_score","result"]:
        if need not in df.columns: df[need] = None

    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["ticker_plain"] = (
        df["ticker"]
        .str.replace(".TW","", regex=False)
        .str.replace(".TWO","", regex=False)
    )
    return df

def load_result_df() -> Optional[pd.DataFrame]:
    path = _pick_latest_result_csv()
    if not path:
        return None
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(path)
    # 先把欄名做清理（去掉空白與換行）
    df.columns = [str(c).strip() for c in df.columns]
    return _normalize_columns(df)
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    對齊欄位名稱為：
    ['ticker','sent_score','tech_score','foreign_score','total_score','result']
    若沒有 total_score，且有三個分數與對應權重，則自動計算。
    """
    # 建「標準化名稱 → 原始名稱」查表（lower + strip）
    norm_map = {str(c).strip().lower(): c for c in df.columns}
    rename = {}

    def pick(keys, target):
        for k in keys:
            if k in norm_map:
                rename[norm_map[k]] = target
                return

    # 基本欄位對齊
    pick(["ticker", "stock", "代號"], "ticker")
    pick(["sent_score", "sentiment_score"], "sent_score")
    pick(["tech_score"], "tech_score")
    pick(["foreign_score"], "foreign_score")
    pick(["total_score", "totalscore", "score"], "total_score")
    pick(["result", "label", "final"], "result")   # 會把 '\nresult' 清成 'result'

    if rename:
        df = df.rename(columns=rename)

    # 補齊必要欄位
    for need in ["ticker", "sent_score", "tech_score", "foreign_score", "total_score", "result"]:
        if need not in df.columns:
            df[need] = None

    # 清理 ticker
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["ticker_plain"] = (
        df["ticker"]
        .str.replace(".TW", "", regex=False)
        .str.replace(".TWO", "", regex=False)
    )

    # 若沒有 total_score，且有三個分數與三個權重，則自動計算
    # 權重欄名常見：sent_weight / tech_weight / foreign_weight
    w_norm = {c.strip().lower(): c for c in df.columns}
    has_weights = all(k in w_norm for k in ["sent_weight", "tech_weight", "foreign_weight"])
    if df["total_score"].isna().all() and has_weights:
        sw = df[w_norm["sent_weight"]]
        tw = df[w_norm["tech_weight"]]
        fw = df[w_norm["foreign_weight"]]
        df["total_score"] = (
            df["sent_score"].astype(float) * sw.astype(float) +
            df["tech_score"].astype(float) * tw.astype(float) +
            df["foreign_score"].astype(float) * fw.astype(float)
        )

    return df

def find_row_by_query(df: pd.DataFrame, query: str) -> Optional[pd.Series]:
    q = (query or "").strip().upper()
    if q.isdigit() and len(q)==4:
        hit = df[(df["ticker_plain"]==q) | (df["ticker"]==f"{q}.TW")]
        return hit.iloc[0] if not hit.empty else None
    code = NAME_TICKER.get(query.strip())
    if not code: return None
    hit = df[(df["ticker_plain"]==code) | (df["ticker"]==f"{code}.TW")]
    return hit.iloc[0] if not hit.empty else None

def pretty_reply_from_row(row: pd.Series) -> str:
    code = str(row.get("ticker_plain") or row.get("ticker") or "")
    name = TICKER_NAME.get(code, "（名稱待補）")

    # 優先使用最後一欄 result 的英文標籤（只認 good/neutral/bad）
    result_label = row.get("result")
    if result_label is not None and not (isinstance(result_label, float) and pd.isna(result_label)):
        final_txt = label_to_text(result_label)
    else:
        # 沒有 result 就用 total_score 做備援
        final_txt = score_to_text(row.get("total_score"))

    s_txt = score_to_text(row.get("sent_score"))
    t_txt = score_to_text(row.get("tech_score"))
    f_txt = score_to_text(row.get("foreign_score"))

    return (
        f"👋 您好，以下是「{name}（{code}）」\n"
        f"• 情緒分析：{s_txt}\n"
        f"• 指標分析：{t_txt}\n"
        f"• 外資分析：{f_txt}\n"
        f"• 整體結果：{final_txt}"
    )

def reply_top10_0050() -> str:
    items = [f"{code} {name}" for code, name in TOP10_0050]
    return "0050 十大成分股：\n" + "、".join(items)
# --- 追加到原檔底部或合適位置 ---
import time

# 簡單快取：CSV 讀取與自動重載
_CACHE = {"path": None, "mtime": None, "df": None}

def load_result_df_cached() -> Optional[pd.DataFrame]:
    path = _pick_latest_result_csv()
    if not path:
        return None
    mtime = os.path.getmtime(path)
    if _CACHE["path"] != path or _CACHE["mtime"] != mtime or _CACHE["df"] is None:
        df = load_result_df()  # 你現有的載入＋欄名清理流程
        _CACHE.update({"path": path, "mtime": mtime, "df": df})
    return _CACHE["df"]

# 模糊查詢（名稱包含 / 代號前綴）
def search_rows(df: pd.DataFrame, keyword: str, limit: int = 10) -> pd.DataFrame:
    kw = keyword.strip()
    # 名稱表可擴充：這裡用你現有的 TICKER_NAME 映射反查
    name_hits = [code for code, name in TICKER_NAME.items() if kw in name]
    mask = (
        df["ticker_plain"].str.startswith(kw.upper(), na=False) |
        df["ticker_plain"].isin(name_hits)
    )
    return df[mask].head(limit)

# 以 result（good/neutral/bad）與 total_score 做排序
def rank(df: pd.DataFrame, topn: int = 10, direction: str = "good") -> pd.DataFrame:
    # 把 result 轉成分數排序鍵：good=+1, neutral=0, bad=-1
    score_map = {"good": 1, "neutral": 0, "bad": -1}
    res = (df.get("result") or pd.Series([None]*len(df))).astype(str).str.lower()
    primary = res.map(score_map).fillna(0)
    # 次排序使用 total_score（缺就填0）
    secondary = pd.to_numeric(df.get("total_score"), errors="coerce").fillna(0)

    if direction == "good":
        ordered = df.assign(_p=primary, _s=secondary).sort_values(by=["_p","_s"], ascending=[False, False])
    elif direction == "bad":
        ordered = df.assign(_p=primary, _s=secondary).sort_values(by=["_p","_s"], ascending=[True, True])
    else:  # neutral
        # 接近0且標籤為neutral
        neutral_mask = res.eq("neutral")
        ordered = df.assign(_s=(secondary.abs())).sort_values(by=["_s"], ascending=True)
        ordered = ordered[neutral_mask] if neutral_mask.any() else ordered
    return ordered.head(topn).drop(columns=["_p","_s"], errors="ignore")

def format_rank_list(df: pd.DataFrame, title: str) -> str:
    if df is None or df.empty:
        return f"{title}\n（無資料）"
    lines = []
    for _, r in df.iterrows():
        code = str(r.get("ticker_plain",""))
        name = TICKER_NAME.get(code, "")
        lab = label_to_text(r.get("result"))
        lines.append(f"{code} {name}｜整體：{lab}")
    return f"{title}\n" + "\n".join(lines)
