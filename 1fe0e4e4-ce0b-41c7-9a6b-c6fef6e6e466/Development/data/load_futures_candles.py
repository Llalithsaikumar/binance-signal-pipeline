
import pandas as pd
import urllib.request
import urllib.parse
import json
from datetime import datetime, timedelta, timezone

# ── Pipeline Config ────────────────────────────────────────────────────────────
INTERVAL = "1m"
LIMIT    = 1000          # rows per request (Binance max is 1500)
BASE_URL = "https://fapi.binance.com/fapi/v1/klines"

now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

# Standard 6-month start (used as default)
start_ms = int((datetime.now(timezone.utc) - timedelta(days=180)).timestamp() * 1000)

# ── COIN_CONFIG ────────────────────────────────────────────────────────────────
# Each entry: symbol (Binance perp ticker), start date for data ingestion,
# threshold (%) for UP/DOWN/SIDEWAYS labelling, and per-coin training hyperparams:
#   epochs              – max training iterations (max_iter for MLP)
#   dropout             – implicit regularisation hint (stored for downstream use)
#   l2                  – L2 weight decay / alpha for MLPClassifier
#   fee                 – per-trade fee in % (used by RL reward function)
#   confidence_threshold – minimum model confidence to emit a live signal
#
# Standard coins: 30 epochs, dropout=0.2, l2=0.0,   fee=0.02%, conf_thresh=0.65
# Meme/volatile  (NEIRO, ZEREBRO): 20 epochs, dropout=0.35, l2=0.001, fee=0.06%, conf_thresh=0.75
COIN_CONFIG = {
    "BTC":    {"symbol": "BTCUSDT",     "start": start_ms, "threshold": 0.3,  "epochs": 30, "dropout": 0.20, "l2": 0.0,   "fee": 0.02, "confidence_threshold": 0.65},
    "ETH":    {"symbol": "ETHUSDT",     "start": start_ms, "threshold": 0.3,  "epochs": 30, "dropout": 0.20, "l2": 0.0,   "fee": 0.02, "confidence_threshold": 0.65},
    "SOL":    {"symbol": "SOLUSDT",     "start": start_ms, "threshold": 0.3,  "epochs": 30, "dropout": 0.20, "l2": 0.0,   "fee": 0.02, "confidence_threshold": 0.65},
    "BNB":    {"symbol": "BNBUSDT",     "start": start_ms, "threshold": 0.3,  "epochs": 30, "dropout": 0.20, "l2": 0.0,   "fee": 0.02, "confidence_threshold": 0.65},
    "XRP":    {"symbol": "XRPUSDT",     "start": start_ms, "threshold": 0.3,  "epochs": 30, "dropout": 0.20, "l2": 0.0,   "fee": 0.02, "confidence_threshold": 0.65},
    "DOGE":   {"symbol": "DOGEUSDT",    "start": start_ms, "threshold": 0.3,  "epochs": 30, "dropout": 0.20, "l2": 0.0,   "fee": 0.02, "confidence_threshold": 0.65},
    "NEIRO":  {"symbol": "NEIROUSDT",   "start": int(datetime(2024, 9, 17, tzinfo=timezone.utc).timestamp() * 1000), "threshold": 1.5, "epochs": 20, "dropout": 0.35, "l2": 0.001, "fee": 0.06, "confidence_threshold": 0.75},
    "ZEREBRO":{"symbol": "ZEREBROUSDT", "start": int(datetime(2025, 1, 3,  tzinfo=timezone.utc).timestamp() * 1000), "threshold": 1.5, "epochs": 20, "dropout": 0.35, "l2": 0.001, "fee": 0.06, "confidence_threshold": 0.75},
}

# Keep backward-compatible SYMBOLS dict for downstream blocks
SYMBOLS = {k.lower(): v["symbol"] for k, v in COIN_CONFIG.items()}

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
]


def fetch_klines(symbol: str, coin_start_ms: int) -> pd.DataFrame:
    """Page through Binance FAPI 1m klines from coin_start_ms → now (public endpoint)."""
    all_rows = []
    cursor   = coin_start_ms

    while cursor < now_ms:
        params = urllib.parse.urlencode({
            "symbol":    symbol,
            "interval":  INTERVAL,
            "startTime": cursor,
            "endTime":   now_ms,
            "limit":     LIMIT,
        })
        url = f"{BASE_URL}?{params}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            batch = json.loads(resp.read())

        if not batch:
            break
        all_rows.extend(batch)
        cursor = batch[-1][0] + 60_000   # advance past last candle (+1 min)
        if len(batch) < LIMIT:
            break                         # no more data

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    numeric_cols = ["open", "high", "low", "close", "volume",
                    "quote_volume", "taker_buy_base_vol", "taker_buy_quote_vol"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df.drop(columns=["ignore"], inplace=True)
    return df


# ── Fetch all 8 perpetual futures ─────────────────────────────────────────────
print("Fetching 1m USDT perp futures candles for 8 coins …\n")
print(f"  {'COIN':<10} {'FROM':<14} {'TO':<6}  {'THRESHOLD':<12} {'EPOCHS':<8} {'DROPOUT':<10} {'L2':<8} {'FEE%':<8} {'CONF_T'}")
print(f"  {'────':<10} {'────':<14} {'──':<6}  {'─────────':<12} {'──────':<8} {'───────':<10} {'──':<8} {'────':<8} {'──────'}")
for coin, cfg in COIN_CONFIG.items():
    _from = datetime.fromtimestamp(cfg["start"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    print(f"  {coin:<10} {_from:<14} {'now':<6}  ±{cfg['threshold']:<11} {cfg['epochs']:<8} {cfg['dropout']:<10} {cfg['l2']:<8} {cfg['fee']:<8} {cfg['confidence_threshold']}")
print()

btc_df    = fetch_klines(COIN_CONFIG["BTC"]["symbol"],     COIN_CONFIG["BTC"]["start"])
eth_df    = fetch_klines(COIN_CONFIG["ETH"]["symbol"],     COIN_CONFIG["ETH"]["start"])
sol_df    = fetch_klines(COIN_CONFIG["SOL"]["symbol"],     COIN_CONFIG["SOL"]["start"])
bnb_df    = fetch_klines(COIN_CONFIG["BNB"]["symbol"],     COIN_CONFIG["BNB"]["start"])
xrp_df    = fetch_klines(COIN_CONFIG["XRP"]["symbol"],     COIN_CONFIG["XRP"]["start"])
doge_df   = fetch_klines(COIN_CONFIG["DOGE"]["symbol"],    COIN_CONFIG["DOGE"]["start"])
neiro_df  = fetch_klines(COIN_CONFIG["NEIRO"]["symbol"],   COIN_CONFIG["NEIRO"]["start"])
zerebro_df= fetch_klines(COIN_CONFIG["ZEREBRO"]["symbol"], COIN_CONFIG["ZEREBRO"]["start"])

# ── Candle count summary + warnings ───────────────────────────────────────────
_coin_dfs = [
    ("BTC",     btc_df),
    ("ETH",     eth_df),
    ("SOL",     sol_df),
    ("BNB",     bnb_df),
    ("XRP",     xrp_df),
    ("DOGE",    doge_df),
    ("NEIRO",   neiro_df),
    ("ZEREBRO", zerebro_df),
]

WARN_THRESHOLD = 50_000

print(f"\n{'─'*65}")
print(f"  {'COIN':<10} {'CANDLES':>10}  {'RANGE'}")
print(f"  {'────':<10} {'───────':>10}  {'─────'}")
for _coin, _df in _coin_dfs:
    _n    = len(_df)
    _from = _df["open_time"].iloc[0].strftime("%Y-%m-%d") if _n > 0 else "N/A"
    _to   = _df["open_time"].iloc[-1].strftime("%Y-%m-%d") if _n > 0 else "N/A"
    print(f"  {_coin:<10} {_n:>10,}  {_from} → {_to}")
print(f"{'─'*65}\n")

for _coin, _df in _coin_dfs:
    _n = len(_df)
    if _n < WARN_THRESHOLD:
        print(f"⚠️  {_coin} has limited data ({_n:,} candles) — model may be less reliable")

print("\n✅  All 8 DataFrames loaded successfully.")
print(f"\n  COIN_CONFIG full parameters:")
print(f"  {'COIN':<10} {'THRESHOLD':<12} {'EPOCHS':<8} {'DROPOUT':<10} {'L2':<8} {'FEE%':<8} {'CONF_T'}")
print(f"  {'────':<10} {'─────────':<12} {'──────':<8} {'───────':<10} {'──':<8} {'────':<8} {'──────'}")
for _coin, _cfg in COIN_CONFIG.items():
    print(f"  {_coin:<10} ±{_cfg['threshold']:<11} {_cfg['epochs']:<8} {_cfg['dropout']:<10} {_cfg['l2']:<8} {_cfg['fee']:<8} {_cfg['confidence_threshold']}")
