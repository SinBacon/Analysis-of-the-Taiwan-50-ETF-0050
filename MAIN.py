# -*- coding: utf-8 -*-
import os, re, sys, time, json, random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import talib
from bs4 import BeautifulSoup
from pandas.tseries.offsets import DateOffset

# ========= 調用llama3.2 =========
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

#  FinMind v4 Token
FINMIND_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNS0wOC0xMSAxNjowMDo1MCIsInVzZXJfaWQiOiJCYWNvbiIsImlwIjoiMjIzLjEzOS4xMTEuNDUiLCJleHAiOjE3NTU1MDQwNTB9.QL88mT4UNiNgUcKNQ0e0LmiZ51b2YeVeAQHCccXFTds"

# ======= 全域權重（缺項將自動正規化） =======
SENT_W_TARGET = 0.25
FOREIGN_W_TARGET = 0.40  
TECH_W_TARGET = 0.35

# ======= 技術面子權重（正規化至 1） =======
TECH_SUB_WEIGHTS = {
    'sig_RSI': 5,
    'sig_MACD': 7,
    'sig_KD': 2,
    'sig_SMAx': 6,
    'sig_BB': 4,
    'sig_OBV': 3,
    'sig_BIAS10': 3,
}
TECH_SUB_SUM = sum(TECH_SUB_WEIGHTS.values())
TECH_SUB_WEIGHTS = {k: v / TECH_SUB_SUM for k, v in TECH_SUB_WEIGHTS.items()}

# 近期視窗與衰減
WINDOW_DAYS = 20
LAMBDA_RECENT = 0.3

# ========= 公用 =========
def normalize_stock_id(ticker: str) -> str:
    return ticker.split('.')[0] if '.' in ticker else ticker

def prompt_yes_no(question: str, default: str = "n") -> bool:
    default = default.lower()
    suffix = " [Y/n]: " if default == "y" else " [y/N]: "
    while True:
        ans = input(question + suffix).strip().lower()
        if ans == "" and default in ("y", "n"):
            return default == "y"
        if ans in ("y", "yes", "是"): return True
        if ans in ("n", "no", "否"): return False
        print("請輸入 y 或 n。")

# ========= PTT STOCK crawler=========
def get_latest_index():
    url = "https://www.ptt.cc/bbs/Stock/index.html"
    try:
        r = requests.get(url, cookies={"over18": "1"}, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        up_a = None
        for a in soup.select(".btn-group-paging a"):
            if a.get_text(strip=True) == "上頁":
                up_a = a; break
        if not up_a or not up_a.get("href"): return None
        m = re.search(r"index(\d+)\.html", up_a["href"])
        if not m: return None
        return int(m.group(1)) + 1
    except Exception:
        return None

def parse_ptt_date(md_str, today_dt):
    md_str = md_str.strip()
    try:
        d = datetime.strptime(f"{today_dt.year}/{md_str}", "%Y/%m/%d")
        if d > today_dt: d = d.replace(year=today_dt.year - 1)
        return d
    except Exception:
        return None

def fetch_article_content(article_url):
    try:
        res = requests.get(article_url, cookies={"over18": "1"}, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        main = soup.find("div", id="main-content")
        if not main: return None
        for tag in main.find_all(['div', 'span']): tag.decompose()
        return main.get_text().split('--')[0].strip()
    except:
        return None

def crawl_ptt_recent_10m(keyword: str):
    today_dt = datetime.now()
    start_dt = (pd.Timestamp(today_dt) - DateOffset(months=10)).to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
    end_dt = today_dt
    print(f"🔎 僅抓取日期區間：{start_dt.date()} ～ {end_dt.date()} 的文章")

    base_url = "https://www.ptt.cc/bbs/Stock/index{}.html"
    start_index = get_latest_index() or 9200
    max_pages_to_try = 1000
    pages_crawled, stop_crawl = 0, False
    rows, seen_titles = [], set()

    for i in range(start_index, 0, -1):
        if stop_crawl or pages_crawled >= max_pages_to_try: break
        try:
            response = requests.get(base_url.format(i), cookies={"over18": "1"},
                                    headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            response.raise_for_status()
        except:
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.find_all("div", class_="r-ent")
        page_max_date, kept = None, 0

        for article in articles:
            title_tag = article.find("div", class_="title")
            title_link = title_tag.find("a") if title_tag else None
            raw_title = title_tag.text.strip() if title_tag else "無標題"
            title = raw_title
            if keyword.lower() not in title.lower() or title in seen_titles: continue

            date_tag = article.find("div", class_="date")
            art_dt = parse_ptt_date(date_tag.text.strip() if date_tag else "", today_dt)
            if not art_dt: continue

            if (page_max_date is None) or (art_dt > page_max_date): page_max_date = art_dt
            if not (start_dt <= art_dt <= end_dt): continue

            article_url = "https://www.ptt.cc" + title_link["href"] if title_link else None
            content = fetch_article_content(article_url) if article_url else "無法取得內文"

            rows.append({"時間": art_dt.strftime("%Y-%m-%d"), "標題": title, "內文": content})
            seen_titles.add(title); kept += 1

        pages_crawled += 1
        print(f"✅ 已完成爬取第 {i} 頁，本頁保留 {kept} 筆，累計共 {len(rows)} 筆")
        if (page_max_date and (page_max_date < start_dt)) or len(rows)>=20:
            print("⏹️ 偵測到頁面日期已早於起始區間，停止往前翻頁。"); stop_crawl = True
        time.sleep(random.uniform(0.4, 0.7))

    if not rows:
        print("❌ 在指定日期區間內沒有找到符合關鍵字的文章。"); sys.exit(1)

    df = pd.DataFrame(rows)
    xlsx = f"ptt_stock_{keyword}.xlsx"
    df.to_excel(xlsx, index=False)
    print(f"💾 已儲存 PTT 文章到 {xlsx}")
    print(f"📅 抓到的日期範圍：{df['時間'].min()} ～ {df['時間'].max()}")
    return df

# ========= 情緒分析 =========
def analyze_sentiment_ollama(text: str) -> str:
    prompt = f"""
    請根據以下文章內文，判斷其情緒是「正面」、「中立」或「負面」。
    你只需要回答這三個詞中的一個，不需要任何額外解釋。
    文章內文：
    「{text}」
    請回答：
    """
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    try:
        res = requests.post(OLLAMA_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
        sentiment = json.loads(res.text).get("response", "").strip()
        if "正面" in sentiment: return "正面"
        if "負面" in sentiment: return "負面"
        if "中立" in sentiment: return "中立"
        return "分析失敗"
    except:
        return "連線失敗"

def run_sentiment_pipeline(ptt_df: pd.DataFrame, keyword: str):
    log_file = "sentiment_progress.log"
    open(log_file, "w", encoding="utf-8").close()

    df = ptt_df.copy()
    df['標題'] = df['標題'].fillna('')
    df['內文'] = df['內文'].fillna('')
    df['分析文本'] = df['標題'] + '。' + df['內文']
    df['情緒分析結果'] = "未分析"

    total_rows = len(df)
    for idx, row in df.iterrows():
        text = row['分析文本']
        if text.strip():
            msg = f"🔍 分析第 {idx+1}/{total_rows} 筆資料..."
            print(msg, flush=True)
            with open(log_file, "a", encoding="utf-8") as f: f.write(f"{datetime.now()} {msg}\n")
            senti = analyze_sentiment_ollama(text)
            df.at[idx, '情緒分析結果'] = senti
            msg2 = f"✅ 結果：{senti}"
            print(msg2, flush=True)
            with open(log_file, "a", encoding="utf-8") as f: f.write(f"{datetime.now()} {msg2}\n")
            time.sleep(1)
        else:
            skip = f"ℹ️ 第 {idx+1} 筆內容空白，跳過"
            print(skip, flush=True)
            with open(log_file, "a", encoding="utf-8") as f: f.write(f"{datetime.now()} {skip}\n")
            df.at[idx, '情緒分析結果'] = "內容空白"

    out_xlsx = f"{keyword}_with_sentiment.xlsx"
    df.to_excel(out_xlsx, index=False)
    print(f"✅ 完成情緒分析並儲存結果：{out_xlsx}")

    # 月彙總
    df['時間'] = pd.to_datetime(df['時間'], errors='coerce')
    df = df.dropna(subset=['時間'])
    df['月份'] = df['時間'].dt.to_period('M')
    monthly = df.groupby(['月份', '情緒分析結果']).size().unstack(fill_value=0)
    monthly['每月判斷'] = monthly.apply(
        lambda r: 'good' if r.get('正面', 0) > r.get('負面', 0) else 'bad' if r.get('負面', 0) > r.get('正面', 0) else 'neutral',
        axis=1
    )
    out_csv = f"{keyword}_monthly_sentiment.csv"
    monthly.to_csv(out_csv, encoding="utf-8-sig")
    print(f"📊 已輸出每月情緒分析：{out_csv}")

    # 轉成 -1~+1 分數（指數衰減）
    m2 = monthly.reset_index().copy()
    m2['月份'] = m2['月份'].astype(str)
    m2['sent_score'] = m2['每月判斷'].map({"good": 1, "neutral": 0, "bad": -1}).fillna(0)

    def to_period_m_safe(s):
        try: return pd.Period(s, freq="M")
        except: return pd.NaT

    def month_index(p: pd.Period) -> int:
        return int(p.year)*12 + int(p.month)

    m2['PeriodM'] = m2['月份'].apply(to_period_m_safe)
    m2 = m2.dropna(subset=['PeriodM'])
    latest = m2['PeriodM'].max()
    latest_idx = month_index(latest)
    m2['months_ago'] = latest_idx - m2['PeriodM'].apply(month_index)
    m2['months_ago'] = pd.to_numeric(m2['months_ago'], errors='coerce').fillna(0).clip(lower=0)
    LAMBDA = 0.6
    m2['weight'] = np.exp(-LAMBDA * m2['months_ago'])
    if m2['weight'].sum() == 0: m2.loc[:, 'weight'] = 1.0
    sent_weighted_score = float((m2['sent_score'] * m2['weight']).sum() / m2['weight'].sum())
    return df, sent_weighted_score

# ========= 技術面 =========
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['Close'])
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'])

    # KD(9,3,3)
    high9 = df['High'].rolling(9).max()
    low9  = df['Low'].rolling(9).min()
    rsv = (df['Close'] - low9) / (high9 - low9) * 100
    df['K'] = rsv.ewm(alpha=1/3, adjust=False).mean()
    df['D'] = df['K'].ewm(alpha=1/3, adjust=False).mean()

    # 均線
    df['SMA5']  = df['Close'].rolling(5).mean()
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA20'] = df['Close'].rolling(20).mean()

    # 布林
    df['BB_MID'] = df['SMA20']
    df['BB_STD'] = df['Close'].rolling(20).std(ddof=0)
    df['BB_UP']  = df['BB_MID'] + 2 * df['BB_STD']
    df['BB_DN']  = df['BB_MID'] - 2 * df['BB_STD']

    # 10日乖離
    df['BIAS10'] = (df['Close'] / df['SMA10'] - 1.0) * 100.0
    return df

def compute_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    df['sig_RSI'] = 0
    df.loc[df['RSI_14'] > 60, 'sig_RSI'] = 1
    df.loc[df['RSI_14'] < 40, 'sig_RSI'] = -1

    df['sig_MACD'] = 0
    cross_up   = (df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))
    cross_down = (df['MACD'] < df['MACD_signal']) & (df['MACD'].shift(1) >= df['MACD_signal'].shift(1))
    df.loc[cross_up, 'sig_MACD'] = 1
    df.loc[cross_down, 'sig_MACD'] = -1

    df['sig_KD'] = 0
    kd_up   = (df['K'] < 20) & ((df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1)))
    kd_down = (df['K'] > 80) & ((df['K'] < df['D']) & (df['K'].shift(1) >= df['D'].shift(1)))
    df.loc[kd_up, 'sig_KD'] = 1
    df.loc[kd_down, 'sig_KD'] = -1

    df['sig_SMAx'] = 0
    ma_up   = (df['SMA5'] > df['SMA20']) & (df['SMA5'].shift(1) <= df['SMA20'].shift(1))
    ma_down = (df['SMA5'] < df['SMA20']) & (df['SMA5'].shift(1) >= df['SMA20'].shift(1))
    df.loc[ma_up, 'sig_SMAx'] = 1
    df.loc[ma_down, 'sig_SMAx'] = -1

    df['sig_BB'] = 0
    df.loc[df['Close'] > df['BB_UP'], 'sig_BB'] = -1
    df.loc[df['Close'] < df['BB_DN'], 'sig_BB'] = 1

    df['sig_OBV'] = 0
    obv_roll = df['OBV'].rolling(5)
    obv_slope = obv_roll.apply(lambda x: (x.iloc[-1] - x.iloc[0]) if len(x.dropna())==5 else 0, raw=False)
    df.loc[obv_slope > 0, 'sig_OBV'] = 1
    df.loc[obv_slope < 0, 'sig_OBV'] = -1

    df['sig_BIAS10'] = 0
    df.loc[df['BIAS10'] > 5.0,  'sig_BIAS10'] = -1
    df.loc[df['BIAS10'] < -5.0, 'sig_BIAS10'] = 1
    return df

def compute_recent_tech_score(df: pd.DataFrame, ticker: str) -> float:
    tail = df.tail(WINDOW_DAYS).copy()
    n = len(tail)
    if n == 0: return 0.0
    ages = np.arange(n)[::-1]
    day_w = np.exp(-LAMBDA_RECENT * ages).astype(float)
    day_w = day_w / day_w.sum()

    tail['tech_combo'] = 0.0
    for col, w in TECH_SUB_WEIGHTS.items():
        if col not in tail.columns: tail[col] = 0
        tail['tech_combo'] += w * tail[col].fillna(0)

    tech_recent_weighted = float(np.dot(tail['tech_combo'].values, day_w))

    recent_out = pd.DataFrame([{
        'Ticker': ticker,
        'WindowDays': n,
        'TechRecentWeighted(-1~+1)': round(tech_recent_weighted, 6)
    }])
    recent_out_name = f"{ticker}_Recent20D_TechScore.csv"
    recent_out.to_csv(recent_out_name, index=False, encoding='utf-8-sig')
    print(f"📈 近期技術面（20 日×子權重）已輸出：{recent_out_name}")
    return tech_recent_weighted

# ========= 外資（FinMind v4 REST） =========
def fetch_foreign_v4(stock_id: str, start: str, end: str,
                     token: str = FINMIND_TOKEN,
                     max_retries: int = 3, backoff_sec: float = 1.5) -> pd.DataFrame:
    url = "https://api.finmindtrade.com/api/v4/data"
    params = {
        "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
        "data_id": stock_id,
        "start_date": start,
        "end_date": end,
        "token": token,
    }
    for attempt in range(1, max_retries + 1):
        r = requests.get(url, params=params, timeout=15)
        try:
            j = r.json()
        except Exception:
            j = {"status": r.status_code, "msg": "non-json response"}
        if j.get("status") in (429, 429.0):
            time.sleep(backoff_sec * attempt); continue
        if j.get("status") == 200 and j.get("data"):
            raw = pd.DataFrame(j["data"])
            df = raw[raw["name"] == "Foreign_Investor"].copy()
            if df.empty:
                return pd.DataFrame(columns=["Date", "NetBuy"])
            df["NetBuy"] = df["buy"] - df["sell"]
            df["Date"] = pd.to_datetime(df["date"]).dt.date
            return df[["Date", "NetBuy"]].sort_values("Date").reset_index(drop=True)
        if attempt == max_retries:
            raise ValueError(f"FinMind v4 取得失敗：{j}")
        time.sleep(backoff_sec * attempt)
    return pd.DataFrame(columns=["Date", "NetBuy"])

def fetch_and_cache_foreign_csv_finmind(ticker: str, months: int = 6,
                                        token: str = FINMIND_TOKEN) -> pd.DataFrame:
    stock_id = normalize_stock_id(ticker)
    end_date = datetime.today().date()
    start_date = (end_date - timedelta(days=int(months*30)))
    df = fetch_foreign_v4(stock_id,
                          start_date.strftime("%Y-%m-%d"),
                          end_date.strftime("%Y-%m-%d"),
                          token=token)
    if df.empty:
        print("⚠️ FinMind v4 未取得外資資料。"); return df
    df.to_csv(f"foreign_{stock_id}.csv", index=False, encoding="utf-8-sig")
    df.to_csv(f"foreign_{ticker}.csv",   index=False, encoding="utf-8-sig")
    print(f"💾 已儲存外資資料（v4）：foreign_{stock_id}.csv, foreign_{ticker}.csv")
    return df

def compute_foreign_score_finmind(
    ticker: str,
    days: int = 20,
    force_refresh: bool = False,
    token: str = FINMIND_TOKEN
) -> float | None:
    """
    外資分數（幅度加權版）：
    取近 days 天 NetBuy，使用 |NetBuy| 當權重，計算：
        score = sum(NetBuy) / sum(|NetBuy|)
    得到 -1 ~ +1 的分數。若全部為 0，回傳 0。
    """
    # 先取得/讀取資料（延用你原本的流程）
    if force_refresh:
        fdf = fetch_and_cache_foreign_csv_finmind(ticker, months=6, token=token)
        if fdf is None or fdf.empty:
            return None
    else:
        candidates = [f"foreign_{ticker}.csv", f"foreign_{normalize_stock_id(ticker)}.csv"]
        fdf = None
        for path in candidates:
            if os.path.exists(path):
                try:
                    tmp = pd.read_csv(path)
                    if not tmp.empty and {'Date','NetBuy'}.issubset(tmp.columns):
                        fdf = tmp
                        break
                except:
                    pass
        if fdf is None:
            fdf = fetch_and_cache_foreign_csv_finmind(ticker, months=6, token=token)
            if fdf is None or fdf.empty:
                return None

    # 取近 N 天並計算加權分數
    fdf['Date'] = pd.to_datetime(fdf['Date'], errors='coerce')
    fdf = fdf.dropna(subset=['Date']).sort_values('Date').tail(days)
    if fdf.empty or 'NetBuy' not in fdf.columns:
        return None

    net = pd.to_numeric(fdf['NetBuy'], errors='coerce').fillna(0.0).astype(float)
    abs_net = np.abs(net)

    # 全部為 0 的情況 → 中立
    denom = abs_net.sum()
    if denom == 0:
        return 0.0

    score = float(net.sum() / denom)   # 自然落在 -1 ~ +1
    # 輕微數值保護（浮點誤差）
    if score > 1:  score = 1.0
    if score < -1: score = -1.0
    return score
def aggregate_final_score(
    ticker: str,
    sent_weighted_score: float,
    tech_recent_weighted: float,
    foreign_force_refresh: bool,
    finmind_token: str,
    eps: float = 0.25
) -> pd.DataFrame:
    """
    彙總總分並寫入 ALL_STOCK_RESULT.csv
    - 權重目標：SENT_W_TARGET / FOREIGN_W_TARGET / TECH_W_TARGET（全域常數）
    - 缺項會自動正規化權重
    - 外資分數使用 compute_foreign_score_finmind（已為幅度加權版）
    - 輸出同時顯示各面向「分數」與「實際使用權重」
    """
    # 個別分數
    sent_score = float(sent_weighted_score) if sent_weighted_score is not None else None
    tech_score = float(tech_recent_weighted) if tech_recent_weighted is not None else None
    foreign_score = compute_foreign_score_finmind(
        ticker=ticker,
        force_refresh=foreign_force_refresh,
        token=finmind_token
    )

    # 權重規劃（若某面向沒分數，權重給 0）
    weight_plan = {
        "sent": SENT_W_TARGET if sent_score is not None else 0.0,
        "foreign": FOREIGN_W_TARGET if foreign_score is not None else 0.0,
        "tech": TECH_W_TARGET if tech_score is not None else 0.0,
    }
    total_w = sum(weight_plan.values())
    if total_w == 0:
        raise ValueError("三個來源皆無分數，無法計算總分。")

    # 正規化後的實際使用權重
    used_weights = {k: (w / total_w) for k, w in weight_plan.items()}

    # 總分
    total_score = (
        (sent_score or 0.0)    * used_weights["sent"] +
        (foreign_score or 0.0) * used_weights["foreign"] +
        (tech_score or 0.0)    * used_weights["tech"]
    )

    # 分類
    if total_score > eps:
        result = "good"
    elif total_score < -eps:
        result = "bad"
    else:
        result = "neutral"

    # 輸出一列
    out_row = {
        "STOCK": ticker,
        "sent_score": round(sent_score, 6) if sent_score is not None else None,
        "sent_weight": round(used_weights["sent"], 6),
        "tech_score": round(tech_score, 6) if tech_score is not None else None,
        "tech_weight": round(used_weights["tech"], 6),
        "foreign_score": round(foreign_score, 6) if foreign_score is not None else None,
        "foreign_weight": round(used_weights["foreign"], 6),
        "total_score": round(float(total_score), 6),
        "result": result,
    }
    out_df = pd.DataFrame([out_row])

    # 追加寫入 CSV
    out_name = "ALL_STOCK_RESULT.csv"
    if os.path.exists(out_name):
        out_df.to_csv(out_name, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        out_df.to_csv(out_name, index=False, encoding="utf-8-sig")

    print(out_df)
    print(f"📊 已寫入：{out_name}")
    return out_df

# ========= 主流程 =========
def main():
    keyword = input("請輸入要搜尋的股票關鍵字（如：台積電）：").strip()
    ticker = input("請輸入 Yahoo 股票代號（如：2330.TW）：").strip()
    stock_id = normalize_stock_id(ticker)

    # --- PTT：若已有檔案先詢問 ---
    ptt_path = f"ptt_stock_{keyword}.xlsx"
    if os.path.exists(ptt_path):
        do_crawl_ptt = prompt_yes_no(f"偵測到已存在 {ptt_path}，要重新爬 PTT 嗎？", default="n")
    else:
        do_crawl_ptt = True

    if do_crawl_ptt:
        ptt_df = crawl_ptt_recent_10m(keyword)
    else:
        print(f"➡️ 跳過 PTT 爬蟲，改讀取現有檔案：{ptt_path}")
        ptt_df = pd.read_excel(ptt_path)

    # --- 情緒 ---
    _, sent_weighted_score = run_sentiment_pipeline(ptt_df, keyword)

    # --- 技術面 ---
    hist = yf.Ticker(ticker).history(period="1y").filter(["Open", "High", "Low", "Close", "Volume"]).reset_index()
    hist['Date'] = pd.to_datetime(hist['Date'], errors='coerce')
    hist = compute_indicators(hist)
    hist = compute_signal_columns(hist)
    tech_recent_weighted = compute_recent_tech_score(hist, ticker)

    # --- 外資（FinMind v4）：若已有 CSV 先詢問是否重抓 ---
    foreign_exists = os.path.exists(f"foreign_{ticker}.csv") or os.path.exists(f"foreign_{stock_id}.csv")
    if foreign_exists:
        foreign_force_refresh = prompt_yes_no("偵測到已存在外資 CSV，要重新用 FinMind 抓取嗎？", default="n")
    else:
        foreign_force_refresh = True  # 沒檔案就需要抓

    # --- 總分 ---
    aggregate_final_score(
        ticker=ticker,
        sent_weighted_score=sent_weighted_score,
        tech_recent_weighted=tech_recent_weighted,
        foreign_force_refresh=foreign_force_refresh,
        finmind_token=FINMIND_TOKEN,
        eps=0.25
    )

if __name__ == "__main__":
    main()
