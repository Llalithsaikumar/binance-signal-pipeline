
import pandas as pd
import numpy as np
import urllib.request
import urllib.parse
import json
import csv
import os
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config — derived from COIN_CONFIG (all 8 coins, per-coin confidence gate) ──
# LIVE_SYMBOLS: all 8 coins that have trained models in gru_models
LIVE_SYMBOLS = {k: v["symbol"] for k, v in COIN_CONFIG.items()}

FAPI_URL      = "https://fapi.binance.com/fapi/v1/klines"
LIVE_INTERVAL = "1m"
LIVE_LIMIT    = 80       # fetch 80 candles — enough for all indicators + ATR(14)
LIVE_SEQ_LEN  = 30       # must match training SEQ_LEN
NUM_CLASSES   = 3

# Per-coin confidence thresholds from COIN_CONFIG:
#   Standard coins : 0.65
#   NEIRO / ZEREBRO: 0.75  (stricter gate for high-volatility meme coins)
LIVE_CONF_THRESHOLD = {coin: cfg["confidence_threshold"] for coin, cfg in COIN_CONFIG.items()}

LIVE_FEATURE_COLS = [
    "rsi_14", "ema_9", "ema_21",
    "macd_line", "macd_signal", "macd_histogram",
    "bb_upper", "bb_lower", "bb_width", "bb_pct_b",
    "vol_sma_20", "vol_delta",
    "body_size_norm", "roc_5",
]
CLASS_NAMES_LIVE = ["DOWN", "SIDEWAYS", "UP"]
LOG_FILE         = "signals_log.csv"

# ── ATR(14) volatility gate — pre-compute historical 90th-percentile ATR ──────
# Uses labeled_data (has 'high', 'low', 'close' from raw OHLCV columns).
# Historical ATR distribution is computed once at startup for each coin.
# Live ATR(14) computed from the freshly fetched candles.
# Flag volatility_warning = True when live_atr > 90th pct of historical ATR.

ATR_PERIOD   = 14
ATR_WARN_PCT = 90        # percentile threshold for high-volatility warning

def _compute_atr_series(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Vectorised ATR(period) using Wilder's EWM smoothing."""
    _n = len(high)
    _tr = np.zeros(_n)
    _tr[0] = high[0] - low[0]
    _prev_close = close[:-1]
    _tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - _prev_close), np.abs(low[1:] - _prev_close))
    )
    # Wilder EWM: alpha = 1/period
    _atr = np.empty(_n)
    _atr[:period] = np.nan
    _atr[period - 1] = np.mean(_tr[:period])
    _alpha = 1.0 / period
    for _i in range(period, _n):
        _atr[_i] = _atr[_i - 1] * (1 - _alpha) + _tr[_i] * _alpha
    return _atr

# Pre-compute historical ATR p90 for each coin from labeled_data
print("Pre-computing historical ATR(14) distributions for volatility gate …")
_hist_atr_p90 = {}
for _coin in COIN_CONFIG:
    if _coin in labeled_data:
        _hd   = labeled_data[_coin]
        _hatr = _compute_atr_series(
            _hd["high"].values, _hd["low"].values, _hd["close"].values, ATR_PERIOD
        )
        _valid = _hatr[~np.isnan(_hatr)]
        _hist_atr_p90[_coin] = float(np.percentile(_valid, ATR_WARN_PCT)) if len(_valid) > 0 else np.inf
        print(f"  {_coin:<8}  ATR-p90 = {_hist_atr_p90[_coin]:.6f}")
    else:
        _hist_atr_p90[_coin] = np.inf   # no history → never warn
        print(f"  {_coin:<8}  no labeled_data → volatility gate disabled")

# ── Pre-compute static lookup ────────────────────────────────────────────────
_N_FEATURES = len(LIVE_FEATURE_COLS)
_FLAT_DIM   = LIVE_SEQ_LEN * _N_FEATURES   # 30 × 14 = 420

# ── CANDLE PARSE COLUMNS (static, allocated once) ────────────────────────────
_KLINE_COLS = ["open_time","open","high","low","close","volume",
               "close_time","quote_volume","num_trades",
               "taker_buy_base_vol","taker_buy_quote_vol","ignore"]
_NUM_COLS   = ["open","high","low","close","volume",
               "quote_volume","taker_buy_base_vol","taker_buy_quote_vol"]

# ── Live candle fetch ────────────────────────────────────────────────────────
def fetch_live_candles(symbol: str, limit: int = LIVE_LIMIT) -> pd.DataFrame:
    """Fetch the most recent `limit` 1-minute candles for a Futures symbol."""
    params = urllib.parse.urlencode({
        "symbol":   symbol,
        "interval": LIVE_INTERVAL,
        "limit":    limit,
    })
    with urllib.request.urlopen(f"{FAPI_URL}?{params}", timeout=15) as resp:
        raw = json.loads(resp.read())

    _df = pd.DataFrame(raw, columns=_KLINE_COLS)
    _df[_NUM_COLS]    = _df[_NUM_COLS].astype(float)
    _df["num_trades"] = _df["num_trades"].astype(int)
    _df["open_time"]  = pd.to_datetime(_df["open_time"],  unit="ms", utc=True)
    _df["close_time"] = pd.to_datetime(_df["close_time"], unit="ms", utc=True)
    _df.drop(columns=["ignore"], inplace=True)
    return _df.reset_index(drop=True)


# ── Feature engineering (mirrors training pipeline) ──────────────────────────
def compute_features_live(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature pipeline used during training."""
    _df    = raw.copy()
    _close = _df["close"]
    _high  = _df["high"]
    _low   = _df["low"]
    _vol   = _df["volume"]
    _open  = _df["open"]

    # RSI(14)
    _delta    = _close.diff()
    _gain     = _delta.clip(lower=0)
    _loss     = (-_delta).clip(lower=0)
    _avg_gain = _gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    _avg_loss = _loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    _rs       = _avg_gain / _avg_loss.replace(0, np.nan)
    _df["rsi_14"] = 100 - (100 / (1 + _rs))

    # EMA 9 & 21
    _df["ema_9"]  = _close.ewm(span=9,  adjust=False).mean()
    _df["ema_21"] = _close.ewm(span=21, adjust=False).mean()

    # MACD(12, 26, 9)
    _ema12 = _close.ewm(span=12, adjust=False).mean()
    _ema26 = _close.ewm(span=26, adjust=False).mean()
    _df["macd_line"]      = _ema12 - _ema26
    _df["macd_signal"]    = _df["macd_line"].ewm(span=9, adjust=False).mean()
    _df["macd_histogram"] = _df["macd_line"] - _df["macd_signal"]

    # Bollinger Bands(20, 2)
    _bb_mid         = _close.rolling(window=20).mean()
    _bb_std         = _close.rolling(window=20).std(ddof=0)
    _df["bb_upper"] = _bb_mid + 2 * _bb_std
    _df["bb_lower"] = _bb_mid - 2 * _bb_std
    _df["bb_width"] = ((_df["bb_upper"] - _df["bb_lower"]) / _bb_mid) * 100
    _df["bb_pct_b"] = (_close - _df["bb_lower"]) / (_df["bb_upper"] - _df["bb_lower"])

    # Volume delta vs 20-period SMA
    _df["vol_sma_20"] = _vol.rolling(window=20).mean()
    _df["vol_delta"]  = _vol - _df["vol_sma_20"]

    # Normalised candle body size
    _df["body_size_norm"] = (_close - _open).abs() / _close

    # 5-period Price ROC
    _df["roc_5"] = _close.pct_change(periods=5) * 100

    return _df.dropna().reset_index(drop=True)


# ── Compute live ATR(14) from raw candle DataFrame ───────────────────────────
def compute_live_atr(candles_df: pd.DataFrame) -> float:
    """Compute the latest ATR(14) value from raw OHLCV candles."""
    _atr = _compute_atr_series(
        candles_df["high"].values,
        candles_df["low"].values,
        candles_df["close"].values,
        ATR_PERIOD,
    )
    _valid = _atr[~np.isnan(_atr)]
    return float(_valid[-1]) if len(_valid) > 0 else 0.0


# ── Z-score normalise + flatten window ──────────────────────────────────────
def build_inference_vector(feat_df: pd.DataFrame) -> np.ndarray:
    """
    Take the last LIVE_SEQ_LEN rows, z-score normalise per-feature, flatten.
    Returns shape (1, LIVE_SEQ_LEN × N_FEATURES) = (1, 420).
    """
    _arr  = feat_df[LIVE_FEATURE_COLS].values[-LIVE_SEQ_LEN:].astype(np.float32)
    _mean = _arr.mean(axis=0)
    _std  = _arr.std(axis=0) + 1e-8
    _norm = (_arr - _mean) / _std
    return _norm.ravel()


# ── Worker: fetch + feature-engineer + ATR one symbol ────────────────────────
def _fetch_and_featurize(coin: str, fapi_symbol: str):
    """Returns (coin, fetch_s, feat_s, feat_df, candles_df, latest_price, latest_ts) or raises."""
    t0 = time.perf_counter()
    _candles = fetch_live_candles(fapi_symbol)
    t1 = time.perf_counter()
    _feat_df = compute_features_live(_candles)
    t2 = time.perf_counter()
    _latest_price = float(_candles["close"].iloc[-1])
    _latest_ts    = str(_candles["open_time"].iloc[-1])
    return coin, fapi_symbol, t1 - t0, t2 - t1, _feat_df, _candles, _latest_price, _latest_ts


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — live inference run with per-coin confidence gate + ATR volatility gate
# ═══════════════════════════════════════════════════════════════════════════════
_wall_start = time.perf_counter()
_timestamp  = datetime.now(timezone.utc).isoformat(timespec="seconds")

print("\n" + "=" * 72)
print(f"  LIVE SIGNAL ENGINE (per-coin gates)  |  {_timestamp}")
print(f"  Symbols: {len(LIVE_SYMBOLS)}  |  Confidence thresholds: std=65%  meme=75%")
print(f"  ATR volatility gate: top {100-ATR_WARN_PCT}% of historical ATR → warning tag")
print("=" * 72)

# ── STEP 1: Parallel HTTP fetches + feature engineering ──────────────────────
print(f"\n[1/3] Parallel fetch + feature engineering ({len(LIVE_SYMBOLS)} workers)…")
_fetch_results = {}   # coin → (fetch_s, feat_s, feat_df, candles_df, price, ts)
_fetch_errors  = {}

with ThreadPoolExecutor(max_workers=len(LIVE_SYMBOLS)) as _pool:
    _futures = {
        _pool.submit(_fetch_and_featurize, c, s): c
        for c, s in LIVE_SYMBOLS.items()
    }
    for _fut in as_completed(_futures):
        _c = _futures[_fut]
        _coin_out, _fsym, _ft, _feat_t, _fd, _cd, _lp, _lt = _fut.result()
        _fetch_results[_coin_out] = (_ft, _feat_t, _fd, _cd, _lp, _lt)

_t_after_fetch = time.perf_counter()
print(f"  ✅ Parallel fetch done in {_t_after_fetch - _wall_start:.3f}s  "
      f"({len(_fetch_results)}/{len(LIVE_SYMBOLS)} succeeded)")

# ── STEP 2: Build batch input matrix (all valid symbols at once) ───────────────
print(f"\n[2/3] Vectorised inference (batch predict + ATR gate) …")

_valid_coins  = []
_prices       = {}
_timestamps_c = {}
_skip_reasons = {}

_X_rows       = []   # will stack → (N, 420)
_live_atrs    = {}   # coin → live ATR(14)
_vol_warnings = {}   # coin → bool

for _coin in LIVE_SYMBOLS:
    if _coin in _fetch_errors:
        _skip_reasons[_coin] = f"fetch error: {_fetch_errors[_coin]}"
        continue
    _ft, _feat_t, _feat_df, _candles_df, _lp, _lt = _fetch_results[_coin]

    # Validate features
    _missing = [c for c in LIVE_FEATURE_COLS if c not in _feat_df.columns]
    if _missing:
        _skip_reasons[_coin] = f"missing features: {_missing}"
        continue
    if len(_feat_df) < LIVE_SEQ_LEN:
        _skip_reasons[_coin] = f"only {len(_feat_df)} rows after dropna (need {LIVE_SEQ_LEN})"
        continue
    if _coin not in gru_models:
        _skip_reasons[_coin] = "no trained model"
        continue

    # ── ATR(14) live value + volatility gate ─────────────────────────────────
    _live_atr = compute_live_atr(_candles_df)
    _live_atrs[_coin] = _live_atr
    _p90 = _hist_atr_p90.get(_coin, np.inf)
    _vol_warnings[_coin] = bool(_live_atr > _p90)

    _valid_coins.append(_coin)
    _prices[_coin]       = _lp
    _timestamps_c[_coin] = _lt
    _X_rows.append(build_inference_vector(_feat_df))

# Stack to (N, 420)
_X_batch = np.vstack(_X_rows) if _X_rows else np.empty((0, _FLAT_DIM), dtype=np.float32)
print(f"  Input batch shape: {_X_batch.shape}")

# ── STEP 3: Per-model predict_proba, apply per-coin confidence threshold ────────
_t_inf_start        = time.perf_counter()
_per_symbol_latency = {}   # coin → inference_ms
_all_results        = []

for _i, _coin in enumerate(_valid_coins):
    _t_sym_start = time.perf_counter()
    _model       = gru_models[_coin]
    _x_row       = _X_batch[_i : _i + 1]          # (1, 420) view

    # Predict probabilities, align to [DOWN, SIDEWAYS, UP]
    _probs_raw = _model.predict_proba(_x_row)[0]
    _prob_full = np.zeros(NUM_CLASSES, dtype=np.float64)
    _prob_full[_model.classes_.astype(int)] = _probs_raw

    _class_idx  = int(np.argmax(_prob_full))
    _confidence = float(_prob_full[_class_idx])
    _signal     = CLASS_NAMES_LIVE[_class_idx]

    # Per-coin confidence threshold from COIN_CONFIG
    _conf_thresh = LIVE_CONF_THRESHOLD[_coin]
    _emitted     = _confidence > _conf_thresh

    _t_sym_end = time.perf_counter()
    _per_symbol_latency[_coin] = (_t_sym_end - _t_sym_start) * 1000

    _all_results.append({
        "coin":               _coin,
        "signal":             _signal,
        "confidence":         round(_confidence, 4),
        "price":              _prices[_coin],
        "timestamp":          _timestamps_c[_coin],
        "emitted":            _emitted,
        "conf_threshold":     _conf_thresh,
        "atr_14":             round(_live_atrs[_coin], 6),
        "atr_p90":            round(_hist_atr_p90.get(_coin, 0.0), 6),
        "volatility_warning": _vol_warnings[_coin],
        "prob_full":          _prob_full,
    })

_t_inf_end    = time.perf_counter()
_total_inf_ms = (_t_inf_end - _t_inf_start) * 1000

# ── STEP 4: Gate on per-coin confidence → build signals_list ───────────────────
signals_list = []
for _r in _all_results:
    if _r["emitted"]:
        signals_list.append({
            k: _r[k]
            for k in ("coin", "signal", "confidence", "price", "timestamp",
                      "volatility_warning", "atr_14", "atr_p90", "conf_threshold")
        })

# ── STEP 5: CSV logging — reset log file to match new 7-col schema ─────────────
# The schema expanded with volatility_warning and atr_14; reset file to avoid
# column-count mismatch when reading old 5-column entries.
_csv_cols = ["timestamp", "coin", "signal", "confidence", "price", "volatility_warning", "atr_14"]
with open(LOG_FILE, "w", newline="") as _f:     # "w" → always write fresh header
    _writer = csv.DictWriter(_f, fieldnames=_csv_cols)
    _writer.writeheader()
    for _sig in signals_list:
        _writer.writerow({k: _sig[k] for k in _csv_cols})

# ── STEP 6: Print detailed results ─────────────────────────────────────────────
print(f"\n[3/3] Per-symbol predictions (per-coin confidence gate + ATR volatility gate):")
print(f"  {'Coin':<7} {'Signal':<9} {'Conf':>7} {'Gate':>6} {'Price':>14}  "
      f"{'ATR_live':>10} {'ATR_p90':>10} {'Vol?':>5}  Emit?")
print(f"  {'-'*82}")

for _r in _all_results:
    _c     = _r["coin"]
    _vwarn = "⚡ YES" if _r["volatility_warning"] else "  no "
    _flag  = "🚨" if _r["emitted"] else "⏸ "
    print(f"  {_c:<7} {_r['signal']:<9} {_r['confidence']:>6.1%} {_r['conf_threshold']:>6.0%} "
          f"{_r['price']:>14,.4f}  {_r['atr_14']:>10.6f} {_r['atr_p90']:>10.6f} {_vwarn}  {_flag}")

if _skip_reasons:
    print(f"\n  ⚠️  Skipped symbols:")
    for _c, _reason in _skip_reasons.items():
        print(f"     [{_c}] {_reason}")

# ── STEP 7: Wall-clock summary ──────────────────────────────────────────────────
_wall_total = time.perf_counter() - _wall_start

print(f"\n{'='*72}")
print(f"  ⏱  LATENCY BREAKDOWN")
print(f"  {'Parallel HTTP fetch (wall):':<40} {(_t_after_fetch - _wall_start)*1000:>8.1f} ms")
print(f"  {'Batch feature engineering (max):':<40} {max((v[1] for v in _fetch_results.values()), default=0)*1000:>8.1f} ms")
print(f"  {'Total inference ({} models):'.format(len(_valid_coins)):<40} {_total_inf_ms:>8.3f} ms")
print(f"  {'Total wall-clock time:':<40} {_wall_total*1000:>8.1f} ms")
print(f"{'='*72}")

print(f"\n  SIGNALS SUMMARY  —  {len(signals_list)}/{len(LIVE_SYMBOLS)} coins emitted a signal")
if signals_list:
    print(f"\n  {'Coin':<7} {'Signal':<9} {'Conf':>7} {'Gate':>6} {'Price':>14}  {'Vol?':<8}  Timestamp")
    print(f"  {'-'*75}")
    for _s in signals_list:
        _vw = "⚡ WARN" if _s["volatility_warning"] else ""
        print(f"  {_s['coin']:<7} {_s['signal']:<9} {_s['confidence']:>6.1%} {_s['conf_threshold']:>6.0%} "
              f"{_s['price']:>14,.4f}  {_vw:<8}  {_s['timestamp']}")
else:
    print("\n  ℹ️  No signals met the per-coin confidence threshold this run.")

_log_df = pd.read_csv(LOG_FILE)
print(f"\n  📄 CSV log: {LOG_FILE}  (appended {len(signals_list)} row(s), total {len(_log_df)})")
print(f"  signals_list → {len(signals_list)} signal(s)  |  all_results → {len(_all_results)} coins processed")
print("=" * 72)
