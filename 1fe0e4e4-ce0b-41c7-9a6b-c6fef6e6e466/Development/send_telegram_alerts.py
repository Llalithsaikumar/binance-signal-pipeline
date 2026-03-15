import os
import json
import urllib.request
import urllib.parse

# ── Read credentials from environment variables ───────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── Direction emoji mapping ───────────────────────────────────────────────────
SIGNAL_EMOJI = {
    "UP":   "🟢 📈",
    "DOWN": "🔴 📉",
}

# ── Low-level Telegram send (stdlib only, no third-party deps) ────────────────
def _telegram_send(token: str, chat_id: str, text: str, parse_mode: str = "MarkdownV2") -> dict:
    """
    POST to Telegram sendMessage endpoint using only urllib (stdlib).
    Returns the JSON response dict.
    """
    url     = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        "chat_id":    chat_id,
        "text":       text,
        "parse_mode": parse_mode,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data    = payload,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


# ── Escape special chars for MarkdownV2 ──────────────────────────────────────
_MDV2_SPECIAL = r"\_*[]()~`>#+-=|{}.!"

def _escape_mdv2(text: str) -> str:
    """Escape all MarkdownV2 reserved characters."""
    for ch in _MDV2_SPECIAL:
        text = text.replace(ch, f"\\{ch}")
    return text


# ── Format a single alert message (MarkdownV2) ────────────────────────────────
def format_alert(sig: dict) -> str:
    direction = sig["signal"]           # "UP" or "DOWN"
    emoji     = SIGNAL_EMOJI[direction]
    coin      = _escape_mdv2(sig["coin"])
    conf_pct  = sig["confidence"] * 100
    price     = sig["price"]

    msg = (
        f"{emoji} *{coin} — {direction} SIGNAL*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 *Price:*        `${price:,.4f}`\n"
        f"🎯 *Confidence:*   `{conf_pct:.1f}%`\n"
        f"⏱ *Timeframe:*    1\\-minute candles \\(30\\-bar GRU window\\)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ _This is an automated ML signal\\. "
        f"Always do your own research\\. "
        f"Manual trade — NOT financial advice\\._"
    )
    return msg


# ── Send all qualifying alerts ─────────────────────────────────────────────────
def _send_alerts() -> int:
    sent = 0

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("  ⚠️  TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — skipping send.")
        return sent

    for sig in signals_list:
        direction = sig.get("signal", "")

        # Skip SIDEWAYS signals silently
        if direction == "SIDEWAYS":
            print(f"  ⏭  [{sig['coin']}] SIDEWAYS — skipped.")
            continue

        if direction not in SIGNAL_EMOJI:
            print(f"  ⚠️  [{sig['coin']}] Unknown signal '{direction}' — skipped.")
            continue

        msg      = format_alert(sig)
        response = _telegram_send(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg)

        if response.get("ok"):
            print(f"  ✅ Alert sent → {sig['coin']} {direction}  ({sig['confidence']*100:.1f}%)")
            sent += 1
        else:
            print(f"  ❌ Telegram API error for {sig['coin']}: {response}")

    return sent


# ── Run ────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  TELEGRAM ALERT DISPATCHER")
print("=" * 60)
print(f"  signals_list contains {len(signals_list)} signal(s):")
for s in signals_list:
    print(f"    • {s['coin']:<5} {s['signal']:<9}  conf={s['confidence']*100:.1f}%")

print()
alerts_sent = _send_alerts()

print()
print(f"  📬 Done. {alerts_sent} Telegram alert(s) sent.")
print("=" * 60)
