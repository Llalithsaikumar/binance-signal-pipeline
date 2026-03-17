
import pandas as pd
import numpy as np
import io

# ══════════════════════════════════════════════════════════════════════════════
# PPO FEATURE MATRIX BUILDER
# ══════════════════════════════════════════════════════════════════════════════
# Reads signals_log.csv, inspects schema, then engineers a return-based feature
# matrix suitable for PPO training:
#   • Log returns
#   • Rolling volatility (20-bar and 5-bar)
#   • Momentum signals (5-bar and 20-bar)
#   • Action labels: 0=SELL, 1=HOLD, 2=BUY  (based on 1-step forward return sign)
#
# Also augments the existing `labeled_data` dict (from label_targets upstream)
# with PPO-specific features, so this block serves as a bridge between the
# GRU/MLP pipeline and PPO training.
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Load & Inspect signals_log.csv ────────────────────────────────────────
print("=" * 70)
print("  STEP 1 — Load signals_log.csv")
print("=" * 70)

_signals_raw = pd.read_csv("signals_log.csv")

print(f"\n  Shape      : {_signals_raw.shape}")
print(f"  Columns    : {list(_signals_raw.columns)}")
print(f"  Dtypes     :\n{_signals_raw.dtypes.to_string()}")
print(f"\n  Head (3 rows):\n{_signals_raw.head(3).to_string()}")
print(f"\n  Null counts:\n{_signals_raw.isnull().sum().to_string()}")
print(f"\n  Descriptive stats:\n{_signals_raw.describe().to_string()}")

# ── 2. Identify the close / price column in signals_log ─────────────────────
# Heuristic: pick whichever column name contains 'close' or 'price'
_price_col = None
for _c in _signals_raw.columns:
    if "close" in _c.lower() or "price" in _c.lower():
        _price_col = _c
        break

# Fallback: first numeric column
if _price_col is None:
    for _c in _signals_raw.columns:
        if _signals_raw[_c].dtype in [np.float64, np.float32, np.int64, np.int32]:
            _price_col = _c
            break

print(f"\n  → Price column identified: '{_price_col}'")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Build PPO feature matrix from signals_log
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STEP 2 — Engineer PPO Feature Matrix from signals_log.csv")
print("=" * 70)

_df = _signals_raw.copy()

# Ensure numeric price column
_df[_price_col] = pd.to_numeric(_df[_price_col], errors="coerce")
_df = _df.dropna(subset=[_price_col]).reset_index(drop=True)

# 2a. Log returns
_df["log_return"] = np.log(_df[_price_col] / _df[_price_col].shift(1))

# 2b. Rolling volatility (std of log returns)
_df["vol_20"]  = _df["log_return"].rolling(window=20).std()   # 20-bar annualised-proxy
_df["vol_5"]   = _df["log_return"].rolling(window=5).std()    # 5-bar short-run vol

# 2c. Momentum signals (cumulative log-return over n bars)
_df["mom_5"]   = _df["log_return"].rolling(window=5).sum()    # 5-bar momentum
_df["mom_20"]  = _df["log_return"].rolling(window=20).sum()   # 20-bar momentum

# 2d. Forward return (1-step ahead) — used for labelling only, no leakage in features
_forward_ret = _df["log_return"].shift(-1)

# 2e. Discrete action label: 0=SELL, 1=HOLD, 2=BUY
#     BUY  → forward return > 0
#     SELL → forward return < 0
#     HOLD → forward return == 0 (edge case, rare)
_df["ppo_action"] = np.where(
    _forward_ret > 0, 2,
    np.where(_forward_ret < 0, 0, 1)
).astype(int)

# 2f. Drop rows with NaN (warm-up period from rolling windows) and last row (no forward label)
ppo_signals_matrix = _df.dropna(subset=["log_return", "vol_20", "vol_5",
                                         "mom_5", "mom_20"]).iloc[:-1].copy()
ppo_signals_matrix = ppo_signals_matrix.reset_index(drop=True)

# ── 3. Build PPO matrix from labeled_data (per-coin, production pipeline) ───
print("\n" + "=" * 70)
print("  STEP 3 — Build PPO Feature Matrix from labeled_data (all 8 coins)")
print("=" * 70)

PPO_FEATURE_COLS = [
    # Existing technical indicators (from compute_features / label_targets)
    "rsi_14", "macd_line", "macd_histogram",
    "bb_width", "bb_pct_b",
    "vol_delta", "body_size_norm", "roc_5",
    # New return-based features (added below)
    "log_return", "vol_20", "vol_5", "mom_5", "mom_20",
]

PPO_COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "NEIRO", "ZEREBRO"]

# Action mapping for PPO (distinct from GRU's -1/0/1 label scheme)
# Uses forward 1-step log-return sign → 0=SELL, 1=HOLD, 2=BUY
PPO_ACTION_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}

ppo_feature_matrices = {}

print(f"\n  PPO feature columns ({len(PPO_FEATURE_COLS)}): {PPO_FEATURE_COLS}")
print(f"  Action map : {PPO_ACTION_MAP}")
print()

for _coin in PPO_COINS:
    _src = labeled_data[_coin].copy()

    # ── Compute return-based features on top of existing indicators ──────────
    _src["log_return"] = np.log(_src["close"] / _src["close"].shift(1))
    _src["vol_20"]     = _src["log_return"].rolling(window=20).std()
    _src["vol_5"]      = _src["log_return"].rolling(window=5).std()
    _src["mom_5"]      = _src["log_return"].rolling(window=5).sum()
    _src["mom_20"]     = _src["log_return"].rolling(window=20).sum()

    # ── Forward 1-step log return for PPO labelling ─────────────────────────
    _fwd = _src["log_return"].shift(-1)

    _src["ppo_action"] = np.where(
        _fwd > 0, 2,
        np.where(_fwd < 0, 0, 1)
    ).astype(int)

    # ── Drop warm-up NaNs + last row (no forward label) ─────────────────────
    _src = _src.dropna(subset=["log_return", "vol_20", "vol_5",
                                "mom_5", "mom_20"]).iloc[:-1].copy()
    _src = _src.reset_index(drop=True)

    ppo_feature_matrices[_coin] = _src

    # ── Action distribution ──────────────────────────────────────────────────
    _total  = len(_src)
    _counts = _src["ppo_action"].value_counts().sort_index()
    _dist   = {PPO_ACTION_MAP[k]: f"{v:>8,} ({v / _total * 100:5.1f}%)"
               for k, v in _counts.items()}

    print(f"  {_coin:<8}  rows={_total:>8,}  features={len(PPO_FEATURE_COLS)}  "
          f"nans={_src[PPO_FEATURE_COLS + ['ppo_action']].isna().sum().sum()}")
    for _aname, _astat in _dist.items():
        print(f"           {_aname:>5}  {_astat}")

# ── 4. Full data summary ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STEP 4 — Full Data Summary")
print("=" * 70)

print(f"\n  [signals_log.csv]")
print(f"    Raw shape          : {_signals_raw.shape}")
print(f"    Columns            : {list(_signals_raw.columns)}")
print(f"    PPO matrix rows    : {len(ppo_signals_matrix):,}")
print(f"    PPO matrix columns : {list(ppo_signals_matrix.columns)}")
print(f"    PPO features       : log_return, vol_20, vol_5, mom_5, mom_20, ppo_action")

_action_dist = ppo_signals_matrix["ppo_action"].value_counts().sort_index()
print(f"    Action distribution:")
for _k, _v in _action_dist.items():
    print(f"      {PPO_ACTION_MAP[_k]:>4} ({_k}) : {_v:,}  ({_v/len(ppo_signals_matrix)*100:.1f}%)")

print(f"\n  [labeled_data → PPO feature matrices]")
print(f"    Coins              : {PPO_COINS}")
print(f"    Feature cols ({len(PPO_FEATURE_COLS)})  : {PPO_FEATURE_COLS}")
print(f"    Action col         : ppo_action  (0=SELL, 1=HOLD, 2=BUY)")

print(f"\n  Per-coin PPO matrix summary:")
print(f"  {'COIN':<10} {'ROWS':>9} {'FEATS':>6} {'NaNs':>6} {'SELL%':>7} {'HOLD%':>7} {'BUY%':>7}")
print(f"  {'-'*55}")
for _coin in PPO_COINS:
    _m  = ppo_feature_matrices[_coin]
    _n  = len(_m)
    _nc = _m[PPO_FEATURE_COLS].isna().sum().sum()
    _ac = _m["ppo_action"].value_counts().sort_index()
    _sell_pct = _ac.get(0, 0) / _n * 100
    _hold_pct = _ac.get(1, 0) / _n * 100
    _buy_pct  = _ac.get(2, 0) / _n * 100
    print(f"  {_coin:<10} {_n:>9,} {len(PPO_FEATURE_COLS):>6} {_nc:>6} "
          f"{_sell_pct:>7.1f} {_hold_pct:>7.1f} {_buy_pct:>7.1f}")

print(f"\n  Sample feature stats (BTC):")
_btc_ppo = ppo_feature_matrices["BTC"][PPO_FEATURE_COLS]
print(_btc_ppo.describe().to_string())

print(f"\n{'='*70}")
print(f"  ✅  ppo_feature_matrices ready — {len(ppo_feature_matrices)} coins")
print(f"  ✅  ppo_signals_matrix    ready — {len(ppo_signals_matrix):,} rows from signals_log.csv")
print(f"  ✅  PPO_FEATURE_COLS      ready — {PPO_FEATURE_COLS}")
print(f"  ✅  PPO_ACTION_MAP        ready — {PPO_ACTION_MAP}")
print(f"{'='*70}")
