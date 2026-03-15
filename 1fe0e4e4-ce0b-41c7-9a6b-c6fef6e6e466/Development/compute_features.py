
import pandas as pd
import numpy as np

# ── Per-coin threshold comes from COIN_CONFIG (defined in load_futures_candles) ──
# COIN_CONFIG[coin]["threshold"] → % threshold for UP/DOWN/SIDEWAYS labelling
# NEIRO and ZEREBRO use 1.5%; all others use 0.3%

# ── Select the raw DataFrame for this Fleet slice ────────────────────────────
if coin_symbol == "BTC":
    _raw = btc_df.copy()
elif coin_symbol == "ETH":
    _raw = eth_df.copy()
elif coin_symbol == "SOL":
    _raw = sol_df.copy()
elif coin_symbol == "BNB":
    _raw = bnb_df.copy()
elif coin_symbol == "XRP":
    _raw = xrp_df.copy()
elif coin_symbol == "DOGE":
    _raw = doge_df.copy()
elif coin_symbol == "NEIRO":
    _raw = neiro_df.copy()
else:  # ZEREBRO
    _raw = zerebro_df.copy()

_sym       = str(coin_symbol)
_threshold = COIN_CONFIG[_sym]["threshold"]   # per-coin label threshold (%)
_close     = _raw["close"]
_high      = _raw["high"]
_low       = _raw["low"]
_vol       = _raw["volume"]
_open      = _raw["open"]

# ── 1. RSI(14) — Wilder's smoothed RSI ──────────────────────────────────────
def _rsi(series, window=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

_raw["rsi_14"] = _rsi(_close, 14)

# ── 2. EMA 9 & EMA 21 ────────────────────────────────────────────────────────
_raw["ema_9"]  = _close.ewm(span=9,  adjust=False).mean()
_raw["ema_21"] = _close.ewm(span=21, adjust=False).mean()

# ── 3. MACD(12,26,9) ─────────────────────────────────────────────────────────
_ema12 = _close.ewm(span=12, adjust=False).mean()
_ema26 = _close.ewm(span=26, adjust=False).mean()
_raw["macd_line"]      = _ema12 - _ema26
_raw["macd_signal"]    = _raw["macd_line"].ewm(span=9, adjust=False).mean()
_raw["macd_histogram"] = _raw["macd_line"] - _raw["macd_signal"]

# ── 4. Bollinger Bands(20,2) ──────────────────────────────────────────────────
_bb_mid   = _close.rolling(window=20).mean()
_bb_std   = _close.rolling(window=20).std(ddof=0)
_raw["bb_upper"]  = _bb_mid + 2 * _bb_std
_raw["bb_lower"]  = _bb_mid - 2 * _bb_std
_raw["bb_width"]  = ((_raw["bb_upper"] - _raw["bb_lower"]) / _bb_mid) * 100  # width %
_raw["bb_pct_b"]  = (_close - _raw["bb_lower"]) / (_raw["bb_upper"] - _raw["bb_lower"])  # %B

# ── 5. Volume delta vs 20-period rolling average ─────────────────────────────
_raw["vol_sma_20"] = _vol.rolling(window=20).mean()
_raw["vol_delta"]  = _vol - _raw["vol_sma_20"]

# ── 6. Normalized candle body size: |close - open| / close ───────────────────
_raw["body_size_norm"] = (_close - _open).abs() / _close

# ── 7. 5-period Price ROC: (close / close[n-5] - 1) * 100 ───────────────────
_raw["roc_5"] = _close.pct_change(periods=5) * 100

# ── 8. Store per-coin label threshold as a column (read by label_targets) ────
_raw["label_threshold"] = _threshold

# ── Drop NaN rows post-calculation ───────────────────────────────────────────
featured_df = _raw.dropna().reset_index(drop=True)
featured_symbol = _sym

print(f"[{_sym}] rows={len(featured_df):,}  cols={featured_df.shape[1]}  "
      f"nans={featured_df.isna().sum().sum()}  label_threshold=±{_threshold}%")
print(f"  cols: {list(featured_df.columns)}")
