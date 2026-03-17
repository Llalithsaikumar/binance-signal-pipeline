import html
import json
import os
import urllib.request


TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

SIGNAL_EMOJI = {
    "UP": "🟢 📈",
    "DOWN": "🔴 📉",
    "SIDEWAYS": "🟡 ↔️",
}


def _telegram_send(token: str, chat_id: str, text: str, parse_mode: str = "HTML") -> dict:
    """POST to Telegram sendMessage using only the Python standard library."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def _fmt_price(value) -> str:
    return f"{float(value):,.4f}"


def _fmt_pct(value) -> str:
    return f"{float(value):+.2f}%"


def _fmt_float(value, places: int = 6) -> str:
    return f"{float(value):.{places}f}"


def format_alert(sig: dict) -> str:
    signal = str(sig.get("signal", "UNKNOWN"))
    emoji = SIGNAL_EMOJI.get(signal, "ℹ️")
    coin = html.escape(str(sig.get("coin", "UNKNOWN")))
    symbol = html.escape(str(sig.get("symbol", coin)))
    timestamp = html.escape(str(sig.get("timestamp", "")))

    current_price = sig.get("price", 0.0)
    target_price = sig.get("target_price", current_price)
    target_price_low = sig.get("target_price_low", target_price)
    target_price_high = sig.get("target_price_high", target_price)
    expected_move_pct = sig.get("expected_move_pct", 0.0)
    move_range_low_pct = sig.get("move_range_low_pct", expected_move_pct)
    move_range_high_pct = sig.get("move_range_high_pct", expected_move_pct)
    forecast_horizon_min = int(sig.get("forecast_horizon_min", 0) or 0)
    forecast_support = int(sig.get("forecast_support", 0) or 0)
    confidence_pct = float(sig.get("confidence", 0.0)) * 100.0
    conf_gate_pct = float(sig.get("conf_threshold", 0.0)) * 100.0
    atr_14 = sig.get("atr_14", 0.0)
    atr_p90 = sig.get("atr_p90", 0.0)
    volatility_status = "HIGH" if sig.get("volatility_warning") else "NORMAL"

    lines = [
        f"{emoji} <b>{symbol}</b> <b>{signal} Forecast</b>",
        f"<b>Coin:</b> <code>{coin}</code>",
        f"<b>Movement:</b> <code>{signal}</code>",
        f"<b>Current Price:</b> <code>{_fmt_price(current_price)}</code>",
        f"<b>Target Price ({forecast_horizon_min}m):</b> <code>{_fmt_price(target_price)}</code>",
        f"<b>Expected Move:</b> <code>{_fmt_pct(expected_move_pct)}</code>",
        f"<b>Price Range:</b> <code>{_fmt_price(target_price_low)} - {_fmt_price(target_price_high)}</code>",
        f"<b>Move Range:</b> <code>{_fmt_pct(move_range_low_pct)} to {_fmt_pct(move_range_high_pct)}</code>",
        f"<b>Confidence:</b> <code>{confidence_pct:.1f}%</code> | <b>Gate:</b> <code>{conf_gate_pct:.1f}%</code>",
        f"<b>ATR(14):</b> <code>{_fmt_float(atr_14)}</code> | <b>ATR p90:</b> <code>{_fmt_float(atr_p90)}</code>",
        f"<b>Volatility:</b> <code>{volatility_status}</code>",
        f"<b>Forecast Basis:</b> <code>{forecast_support:,} historical matches</code>",
        f"<b>Time:</b> <code>{timestamp}</code>",
        "",
        f"<i>Target price is an estimate derived from historical {forecast_horizon_min}-minute forward returns for this signal class.</i>",
    ]
    return "\n".join(lines)


def _send_alerts() -> int:
    sent = 0

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("  Warning: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set; skipping send.")
        return sent

    for sig in signals_list:
        direction = str(sig.get("signal", ""))
        if direction not in SIGNAL_EMOJI:
            print(f"  Warning: [{sig.get('coin', '?')}] unknown signal '{direction}'; skipped.")
            continue

        msg = format_alert(sig)
        response = _telegram_send(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg)

        if response.get("ok"):
            print(
                f"  Alert sent -> {sig.get('coin', '?')} {direction}  "
                f"target={float(sig.get('target_price', sig.get('price', 0.0))):,.4f}  "
                f"conf={float(sig.get('confidence', 0.0)) * 100:.1f}%"
            )
            sent += 1
        else:
            print(f"  Telegram API error for {sig.get('coin', '?')}: {response}")

    return sent


print("=" * 72)
print("  TELEGRAM ALERT DISPATCHER")
print("=" * 72)
print(f"  signals_list contains {len(signals_list)} signal(s):")
for s in signals_list:
    print(
        f"    - {s.get('coin', '?'):<5} {s.get('signal', '?'):<9}  "
        f"conf={float(s.get('confidence', 0.0))*100:.1f}%  "
        f"target={float(s.get('target_price', s.get('price', 0.0))):,.4f}"
    )

print()
alerts_sent = _send_alerts()

print()
print(f"  Done. {alerts_sent} Telegram alert(s) sent.")
print("=" * 72)
