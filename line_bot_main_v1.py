# line_bot_main_v1.py（支援「訂閱時間 HH:MM」每日推播）
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

import os, json, re
from datetime import datetime
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler

from query_result_router import (
    load_result_df_cached, find_row_by_query, pretty_reply_from_row,
    search_rows, rank, format_rank_list, reply_top10_0050, label_to_text, TICKER_NAME
)

load_dotenv()
app = Flask(__name__)

line_bot_api = LineBotApi(os.getenv("LINE_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))

# === APScheduler ===
scheduler = BackgroundScheduler(timezone="Asia/Taipei")
scheduler.start()

SUB_PATH = "subscriptions.json"

def _load_sub():
    if os.path.exists(SUB_PATH):
        with open(SUB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_sub(data):
    with open(SUB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _job_id(uid: str) -> str:
    return f"daily_push_{uid}"

def _schedule_user_push(uid: str, hhmm: str):
    """建立或更新使用者的每日推播排程"""
    m = re.match(r"^([01]?\d|2[0-3]):([0-5]\d)$", hhmm)
    if not m:
        raise ValueError("時間格式錯誤，請用 HH:MM（例：14:30）")
    hour, minute = int(m.group(1)), int(m.group(2))

    # 若已存在同 id 的 job，直接覆蓋
    job_id = _job_id(uid)
    scheduler.add_job(
        func=_push_user_digest_job,
        trigger="cron",
        hour=hour,
        minute=minute,
        id=job_id,
        replace_existing=True,
        args=[uid],
        misfire_grace_time=60,  # 過期1分鐘內補跑
        max_instances=1,
        coalesce=True,
    )

def _cancel_user_push(uid: str):
    job_id = _job_id(uid)
    try:
        scheduler.remove_job(job_id)
    except Exception:
        pass  # 沒有 job 也沒關係

def _compact_line_for_code(df, code: str) -> str:
    """回傳一行摘要：2330 台積電｜整體：看好"""
    # 允許使用者存 .TW 但我們以四碼為主
    q = code.strip().upper().replace(".TW", "").replace(".TWO", "")
    row = find_row_by_query(df, q)
    if row is None:
        return f"{q}｜查無資料"
    # 取名稱（若表中沒有就用空字串）
    name = TICKER_NAME.get(str(row.get("ticker_plain") or q), "")
    # 取 result → 中文（或用 total_score 後備，已在 pretty/label 處理）
    lab = label_to_text(row.get("result"))
    if lab == str(row.get("result")):  # 表示不是 good/neutral/bad，可能 None；交給 pretty 當後備
        # 用 pretty 的整體結果（會用 total_score 萃取文字）
        pretty = pretty_reply_from_row(row)
        # 從 pretty 中抓「整體結果」那一行最後兩字（看好/中立/看壞）
        m = re.search(r"整體結果：([^\n]+)", pretty)
        lab = m.group(1) if m else "無資料"
    name_part = f" {name}" if name else ""
    return f"{q}{name_part}｜整體：{lab}"

def _build_user_digest(uid: str) -> str:
    subs = _load_sub()
    user = subs.get(uid, {})
    codes = user.get("codes", []) if isinstance(user, dict) else subs.get(uid, [])
    if not codes:
        return "你目前沒有訂閱任何股票。"

    df = load_result_df_cached()
    if df is None:
        return "找不到 ALL_STOCK_RESULT.csv，請先放到執行目錄。"

    lines = ["📬 訂閱股票每日摘要"]
    for code in codes:
        lines.append(_compact_line_for_code(df, str(code)))
    return "\n".join(lines)

def _push_user_digest_job(uid: str):
    """排程呼叫：把摘要推給指定使用者"""
    try:
        msg = _build_user_digest(uid)
        line_bot_api.push_message(uid, TextSendMessage(text=msg))
    except Exception as e:
        print(f"[PUSH][{uid}] 失敗：{e}")

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature', '')
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    uid = event.source.user_id
    text = (event.message.text or "").strip()
    df = load_result_df_cached()

    # === 訂閱時間 HH:MM ===
    if text.startswith("訂閱時間"):
        hhmm = text.replace("訂閱時間", "").strip()
        try:
            _schedule_user_push(uid, hhmm)
            subs = _load_sub()
            # 統一結構：{ uid: {"codes":[...], "time":"HH:MM"} }
            user = subs.get(uid) or {}
            if isinstance(user, list):  # 舊格式升級
                user = {"codes": user, "time": hhmm}
            else:
                user["time"] = hhmm
            subs[uid] = user
            _save_sub(subs)
            return line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"已設定每日 {hhmm} 推播訂閱股票摘要。"))
        except ValueError as ve:
            return line_bot_api.reply_message(event.reply_token, TextSendMessage(text=str(ve)))

    # === 取消推播 ===
    if text in {"取消推播", "關閉推播"}:
        _cancel_user_push(uid)
        subs = _load_sub()
        user = subs.get(uid) or {}
        if isinstance(user, dict) and "time" in user:
            user.pop("time")
            subs[uid] = user
            _save_sub(subs)
        return line_bot_api.reply_message(event.reply_token, TextSendMessage(text="已取消你的每日推播。"))

    # === 查推播（顯示目前時間與訂閱清單） ===
    if text in {"查推播", "推播狀態"}:
        subs = _load_sub()
        user = subs.get(uid) or {}
        time_str = user.get("time", "（未設定）") if isinstance(user, dict) else "（未設定）"
        codes = user.get("codes", []) if isinstance(user, dict) else subs.get(uid, [])
        lst = "、".join(codes) if codes else "（無）"
        return line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"推播時間：{time_str}\n訂閱清單：{lst}"))

    # === 0050 成分股 ===
    if text in {"查看0050", "0050成分股", "看0050的成分股"}:
        return line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_top10_0050()))

    # === 排行：看好 / 看壞 / 中立 ===
    if text.startswith("排行"):
        if df is None:
            return line_bot_api.reply_message(event.reply_token, TextSendMessage(text="找不到 ALL_STOCK_RESULT.csv"))
        if "看壞" in text:
            msg = format_rank_list(rank(df, 10, "bad"), "📉 看壞前10")
        elif "中立" in text:
            msg = format_rank_list(rank(df, 10, "neutral"), "😐 中立前10")
        else:
            msg = format_rank_list(rank(df, 10, "good"), "📈 看好前10")
        return line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg))

    # === 訂閱 / 取消 / 我的訂閱 ===
    if text.startswith("訂閱 "):
        code = text.split(" ", 1)[1].strip()
        subs = _load_sub()
        user = subs.get(uid) or {}
        if isinstance(user, list):  # 舊結構升級
            user = {"codes": user}
        user.setdefault("codes", [])
        if code not in user["codes"]:
            user["codes"].append(code)
            subs[uid] = user
            _save_sub(subs)
        return line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"已訂閱：{code}"))
    if text.startswith("取消 "):
        code = text.split(" ", 1)[1].strip()
        subs = _load_sub()
        user = subs.get(uid) or {}
        if isinstance(user, dict) and code in user.get("codes", []):
            user["codes"].remove(code)
            subs[uid] = user
            _save_sub(subs)
            return line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"已取消：{code}"))
        return line_bot_api.reply_message(event.reply_token, TextSendMessage(text="你沒有訂閱這個代號"))
    if text in {"我的訂閱", "訂閱清單"}:
        subs = _load_sub()
        user = subs.get(uid) or {}
        codes = user.get("codes", []) if isinstance(user, dict) else subs.get(uid, [])
        return line_bot_api.reply_message(event.reply_token, TextSendMessage(text="你的訂閱：\n" + ("、".join(codes) or "（空）")))

    # === 查個股：代號或名稱；若多筆命中列清單 ===
    if text.startswith("查 "):
        q = text[2:].strip()
        if df is None:
            return line_bot_api.reply_message(event.reply_token, TextSendMessage(text="找不到 ALL_STOCK_RESULT.csv"))
        row = find_row_by_query(df, q)
        if row is not None:
            return line_bot_api.reply_message(event.reply_token, TextSendMessage(text=pretty_reply_from_row(row)))
        hits = search_rows(df, q, limit=10)
        if not hits.empty:
            lines = []
            for _, r in hits.iterrows():
                code = str(r.get("ticker_plain", ""))
                name = TICKER_NAME.get(code, "")
                lines.append(f"{code} {name}".strip())
            msg = "找到多筆，請再輸入完整代號：\n" + "\n".join(lines)
            return line_bot_api.reply_message(event.reply_token, TextSendMessage(text=msg))
        return line_bot_api.reply_message(event.reply_token, TextSendMessage(text="查無此股票"))

    # === 預設說明 ===
    help_msg = (
        "可用指令：\n"
        "• 查 2330 / 查 台積電\n"
        "• 排行 看好（或 看壞 / 中立）\n"
        "• 查看0050\n"
        "• 訂閱 2330｜取消 2330｜我的訂閱\n"
        "• 訂閱時間 14:30（每天 14:30 自動推播你的訂閱股票摘要）\n"
        "• 查推播｜取消推播"
    )
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=help_msg))

if __name__ == "__main__":
    # 可視需要改 port
    app.run(port=5000)
