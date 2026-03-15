
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ── Per-coin thresholds come from COIN_CONFIG (defined in load_futures_candles) ──
# COIN_CONFIG[coin]["threshold"] → % threshold for UP/DOWN/SIDEWAYS labelling.
# The threshold is carried through the fleet via the "label_threshold" column
# on each featured DataFrame (set in compute_features). We read it directly
# from the column to ensure self-contained, reliable operation post-aggregation.
#
# NEIRO   → ±1.5%   (high-volatility meme coin)
# ZEREBRO → ±1.5%   (high-volatility meme coin)
# All others → ±0.3%

LABEL_MAP = {1: "UP", 0: "SIDEWAYS", -1: "DOWN"}

# All 8 coins — the full set in the pipeline
COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "NEIRO", "ZEREBRO"]

# ── Meme coin set — drives special RL rewards and higher confidence gate ───────
MEME_COINS = {"NEIRO", "ZEREBRO"}

# ── Redefine COIN_CONFIG with full per-coin parameters ───────────────────────
# Standard coins : 30 epochs, dropout=0.20, l2=0.0, fee=0.02%, conf_thresh=0.65
# NEIRO / ZEREBRO: 20 epochs, dropout=0.35, l2=0.001, fee=0.06%, conf_thresh=0.75
COIN_CONFIG = {
    "BTC":    {"symbol": "BTCUSDT",     "threshold": 0.3,  "epochs": 30, "dropout": 0.20, "l2": 0.0,   "fee": 0.02, "confidence_threshold": 0.65},
    "ETH":    {"symbol": "ETHUSDT",     "threshold": 0.3,  "epochs": 30, "dropout": 0.20, "l2": 0.0,   "fee": 0.02, "confidence_threshold": 0.65},
    "SOL":    {"symbol": "SOLUSDT",     "threshold": 0.3,  "epochs": 30, "dropout": 0.20, "l2": 0.0,   "fee": 0.02, "confidence_threshold": 0.65},
    "BNB":    {"symbol": "BNBUSDT",     "threshold": 0.3,  "epochs": 30, "dropout": 0.20, "l2": 0.0,   "fee": 0.02, "confidence_threshold": 0.65},
    "XRP":    {"symbol": "XRPUSDT",     "threshold": 0.3,  "epochs": 30, "dropout": 0.20, "l2": 0.0,   "fee": 0.02, "confidence_threshold": 0.65},
    "DOGE":   {"symbol": "DOGEUSDT",    "threshold": 0.3,  "epochs": 30, "dropout": 0.20, "l2": 0.0,   "fee": 0.02, "confidence_threshold": 0.65},
    "NEIRO":  {"symbol": "NEIROUSDT",   "threshold": 1.5,  "epochs": 20, "dropout": 0.35, "l2": 0.001, "fee": 0.06, "confidence_threshold": 0.75},
    "ZEREBRO":{"symbol": "ZEREBROUSDT", "threshold": 1.5,  "epochs": 20, "dropout": 0.35, "l2": 0.001, "fee": 0.06, "confidence_threshold": 0.75},
}

labeled_data = {}

print("=" * 70)
print("  TARGET LABEL GENERATION  (horizon = 5 bars, per-coin threshold)")
print("  Thresholds sourced from COIN_CONFIG via label_threshold column")
print("=" * 70)

for _coin in COINS:
    _df = featured_dfs[_coin].copy()

    # -- Read per-coin threshold from the DataFrame column set by compute_features
    _threshold = float(_df["label_threshold"].iloc[0])

    # -- 5-bar forward return (no leakage: close shifted by -5) --------------
    _close_ahead   = _df["close"].shift(-5)
    _future_return = (_close_ahead - _df["close"]) / _df["close"] * 100

    # -- Apply label rules (before dropping NaN rows) ------------------------
    _label = np.where(
        _future_return >  _threshold,    1,    # UP
        np.where(_future_return < -_threshold, -1, 0)  # DOWN / SIDEWAYS
    )
    _df["label"] = _label  # dtype int64; NaN rows get 0 but will be dropped

    # -- Drop last 5 rows: future_return is NaN there → data leakage risk ----
    _df = _df.iloc[:-5].copy()

    labeled_data[_coin] = _df

    # -- Class distribution --------------------------------------------------
    _counts = _df["label"].value_counts().sort_index()
    _total  = len(_df)
    _dist   = {LABEL_MAP[k]: f"{v:>8,} ({v / _total * 100:5.1f}%)" for k, v in _counts.items()}

    print(f"\n  {_coin}  — rows: {_total:,}  |  threshold: ±{_threshold}%  |  cols: {_df.shape[1]}")
    for _lname, _lstat in _dist.items():
        print(f"    {_lname:>9s}  {_lstat}")

print("\n" + "=" * 70)
print(f"  labeled_data keys  : {list(labeled_data.keys())}")
print(f"  Columns per coin   : {list(labeled_data['BTC'].columns[-5:])}")
print(f"  Label dtype        : {labeled_data['BTC']['label'].dtype}")
print()
print("  COIN_CONFIG — full per-coin parameters (forwarded downstream):")
print(f"  {'COIN':<10} {'EPOCHS':<8} {'DROPOUT':<10} {'L2':<7} {'FEE%':<8} {'CONF_THRESH'}")
for _c, _cfg in COIN_CONFIG.items():
    _tag = " ← meme" if _c in MEME_COINS else ""
    print(f"  {_c:<10} {_cfg['epochs']:<8} {_cfg['dropout']:<10} {_cfg['l2']:<7} {_cfg['fee']:<8} {_cfg['confidence_threshold']}{_tag}")
print("=" * 70)
