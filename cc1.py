# export_all_in_one.py
# -*- coding: utf-8 -*-
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import talib
import requests

# ========= 參數 =========
WINDOW_DAYS = 20
TECH_W = 0.6
FOREIGN_W = 0.4
FINMIND_TOKEN = os.getenv("FINMIND_TOKEN", "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNS0wOC0xOSAxMTo0MzoyMSIsInVzZXJfaWQiOiJCYWNvbiIsImlwIjoiNTkuMTIwLjI0OS4xMDAifQ.6gio41jHath2NHeRWfNL4ADi7RsxVQxFuBSmA_4ve1c")

# ========= 工具 =========
def normalize_stock_id(ticker: str) -> str:
    return ticker.split('.')[0] if '.' in ticker else ticker

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def strip_tz(series_or_col):
    s = pd.to_datetime(series_or_col, errors='coerce')
    try:
        return s.dt.tz_localize(None)
    except Exception:
        return s

# ========= 技術指標 =========
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df['SMA5']  = df['Close'].rolling(5).mean()
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA20'] = df['Close'].rolling(20).mean()

    df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['Close'])
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])

    high9 = df['High'].rolling(9).max()
    low9  = df['Low'].rolling(9).min()
    rsv = (df['Close'] - low9) / (high9 - low9) * 100
    df['K'] = rsv.ewm(alpha=1/3, adjust=False).mean()
    df['D'] = df['K'].ewm(alpha=1/3, adjust=False).mean()

    df['BB_MID'] = df['SMA20']
    df['BB_STD'] = df['Close'].rolling(20).std(ddof=0)
    df['BB_UP']  = df['BB_MID'] + 2 * df['BB_STD']
    df['BB_DN']  = df['BB_MID'] - 2 * df['BB_STD']

    df['BIAS10'] = (df['Close'] / df['SMA10'] - 1.0) * 100.0
    return df

def compute_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    df['sig_RSI'] = np.where(df['RSI_14'] > 60, 1, np.where(df['RSI_14'] < 40, -1, 0))
    df['sig_MACD'] = np.where(df['MACD'] > df['MACD_signal'], 1, -1)
    df['sig_KD']   = np.where(df['K'] > df['D'], 1, -1)
    df['sig_SMAx'] = np.where(df['SMA5'] > df['SMA20'], 1, -1)
    df['sig_BB']   = np.where(df['Close'] > df['BB_UP'], -1, np.where(df['Close'] < df['BB_DN'], 1, 0))
    df['sig_OBV']  = np.where(df['OBV'].diff() > 0, 1, -1)
    df['sig_BIAS10'] = np.where(df['BIAS10'] > 5, -1, np.where(df['BIAS10'] < -5, 1, 0))
    return df

def compute_recent_tech_score(df: pd.DataFrame) -> float:
    tail = df.tail(WINDOW_DAYS).copy()
    if tail.empty: return 0.0
    cols = ['sig_RSI','sig_MACD','sig_KD','sig_SMAx','sig_BB','sig_OBV','sig_BIAS10']
    cols = [c for c in cols if c in tail.columns]
    if not cols: return 0.0
    return float(tail[cols].mean().mean())

# ========= 外資 =========
def fetch_foreign_v4(stock_id: str, start: str, end: str, token: str) -> pd.DataFrame:
    url = "https://api.finmindtrade.com/api/v4/data"
    params = {
        "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
        "data_id": stock_id,
        "start_date": start,
        "end_date": end,
        "token": token,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        j = r.json()
    except Exception:
        return pd.DataFrame()

    if j.get("status") != 200 or not j.get("data"):
        return pd.DataFrame()

    raw = pd.DataFrame(j["data"])
    df = raw[raw["name"] == "Foreign_Investor"].copy()
    if df.empty: return pd.DataFrame()

    df["NetBuy"] = pd.to_numeric(df["buy"], errors="coerce").fillna(0) - pd.to_numeric(df["sell"], errors="coerce").fillna(0)
    df["Date"] = strip_tz(raw["date"])
    df = df[["Date", "NetBuy"]].sort_values("Date").reset_index(drop=True)
    return df

def compute_foreign_score(fdf: pd.DataFrame) -> float:
    if fdf is None or fdf.empty: return 0.0
    tail = fdf.tail(WINDOW_DAYS)
    if tail.empty: return 0.0
    net = pd.to_numeric(tail["NetBuy"], errors="coerce").fillna(0.0)
    denom = net.abs().sum()
    if denom == 0: return 0.0
    return float(net.sum() / denom)

# ========= 主流程 =========
def main():
    ticker = input("請輸入 Yahoo 股票代號（如：2330.TW）：").strip()
    stock_id = normalize_stock_id(ticker)

    outdir = os.path.join("data", stock_id)
    ensure_dir(outdir)

    # 1) 股價
    hist = yf.Ticker(ticker).history(period="3mo").reset_index()
    hist.rename(columns={"Date": "Date"}, inplace=True)
    hist["Date"] = strip_tz(hist["Date"])
    hist = compute_indicators(hist)
    hist = compute_signal_columns(hist)

    # 2) 外資
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=60)
    fdf = fetch_foreign_v4(stock_id, str(start_date), str(end_date), FINMIND_TOKEN)

    # 合併
    merged = hist.merge(fdf, how="left", on="Date")

    # 3) 計分
    tech_score = compute_recent_tech_score(hist)
    foreign_score = compute_foreign_score(fdf)
    total_score = tech_score * TECH_W + foreign_score * FOREIGN_W

    merged["TechScore_20D"] = round(tech_score, 6)
    merged["ForeignScore_20D"] = round(foreign_score, 6)
    merged["TotalScore_20D"] = round(total_score, 6)
    merged["CalcDate"] = datetime.now().strftime("%Y-%m-%d")

    # 4) 單一輸出
    out_csv = os.path.join(outdir, f"{stock_id}_all_in_one.csv")
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("✅ 已輸出單一檔案：", out_csv)

if __name__ == "__main__":
    main()
