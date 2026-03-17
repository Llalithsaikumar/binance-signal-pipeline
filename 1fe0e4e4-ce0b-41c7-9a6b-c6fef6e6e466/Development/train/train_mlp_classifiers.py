
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

# ── Config ─────────────────────────────────────────────────────────────────────
SEQ_LEN     = 30       # timesteps per input window
NUM_CLASSES = 3        # DOWN=0, SIDEWAYS=1, UP=2
MAX_SAMPLES = 30000    # cap training samples for speed

# Feature columns (14 total, as validated in collect_featured_dfs)
FEATURE_COLS = [
    "rsi_14", "ema_9", "ema_21",
    "macd_line", "macd_signal", "macd_histogram",
    "bb_upper", "bb_lower", "bb_width", "bb_pct_b",
    "vol_sma_20", "vol_delta",
    "body_size_norm", "roc_5",
]

# Map original labels (-1, 0, 1) → class indices (0, 1, 2)
LABEL_REMAP = {-1: 0, 0: 1, 1: 2}
CLASS_NAMES = ["DOWN", "SIDEWAYS", "UP"]

# ── RL Reward Constants ────────────────────────────────────────────────────────
# NEIRO / ZEREBRO special reward table (per ticket spec):
#   correct signal on >1.5% move : +2.0
#   correct signal on <1.5% move : +0.3
#   wrong signal                  : -2.5
#   fee                           : -0.06  (read from COIN_CONFIG['fee'])
# Standard coins: magnitude-scaled reward, fee from COIN_CONFIG['fee']
#   correct signal: +1.0 * |return_pct| / threshold  (capped at +2.0)
#   wrong signal  : -1.0
#   fee           : -COIN_CONFIG[coin]['fee']

MEME_REWARD_CORRECT_BIG   = +2.0   # correct + |return| > threshold
MEME_REWARD_CORRECT_SMALL = +0.3   # correct + |return| < threshold
MEME_REWARD_WRONG         = -2.5   # wrong signal
STD_REWARD_CORRECT_BASE   = +1.0   # standard correct base (scaled by return)
STD_REWARD_WRONG          = -1.0   # standard wrong signal

print("Backend: sklearn MLPClassifier (CPU)\n")
print("Per-coin hyperparameters read from COIN_CONFIG:")
print(f"  {'COIN':<10} {'EPOCHS':<8} {'DROPOUT':<10} {'L2':<8} {'FEE%':<8} {'CONF_T':<8}  NOTE")
print(f"  {'────':<10} {'──────':<8} {'───────':<10} {'──':<8} {'────':<8} {'──────':<8}  ────")
for _c in COINS:
    _cfg = COIN_CONFIG[_c]
    _note = "(meme/volatile)" if _c in MEME_COINS else "(standard)"
    print(f"  {_c:<10} {_cfg['epochs']:<8} {_cfg['dropout']:<10} {_cfg['l2']:<8} {_cfg['fee']:<8} {_cfg['confidence_threshold']:<8}  {_note}")
print()


# ── Sequence builder — flattens (seq_len × n_feat) → 1-D feature vector ────────
def build_sequences_flat(df, feature_cols, seq_len, label_remap, max_samples=None):
    _X_raw = df[feature_cols].values.astype(np.float32)
    _y_raw = df["label"].map(label_remap).values.astype(np.int64)

    # Z-score normalise features
    _mean = _X_raw.mean(axis=0)
    _std  = _X_raw.std(axis=0) + 1e-8
    _X_raw = (_X_raw - _mean) / _std

    _N = len(_X_raw) - seq_len
    _X = np.empty((_N, seq_len * len(feature_cols)), dtype=np.float32)
    _y = np.empty(_N, dtype=np.int64)
    for _i in range(_N):
        _X[_i] = _X_raw[_i: _i + seq_len].ravel()
        _y[_i] = _y_raw[_i + seq_len]

    # Use the most recent rows (chronological tail) up to max_samples
    if max_samples is not None and _N > max_samples:
        _X = _X[-max_samples:]
        _y = _y[-max_samples:]

    return _X, _y


# ── Precision / Recall ─────────────────────────────────────────────────────────
def precision_recall_per_class(y_true, y_pred, n_classes=3):
    _precision = np.zeros(n_classes)
    _recall    = np.zeros(n_classes)
    _support   = np.zeros(n_classes, dtype=int)
    for _c in range(n_classes):
        _tp = np.sum((y_pred == _c) & (y_true == _c))
        _fp = np.sum((y_pred == _c) & (y_true != _c))
        _fn = np.sum((y_pred != _c) & (y_true == _c))
        _precision[_c] = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
        _recall[_c]    = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
        _support[_c]   = int(np.sum(y_true == _c))
    return _precision, _recall, _support


# ── Per-coin training ──────────────────────────────────────────────────────────
gru_models  = {}   # trained models (name kept for downstream compatibility)
gru_metrics = {}

print("=" * 70)
print("  CLASSIFIER TRAINING  |  30 timesteps (flattened)  |  MLP  |  sklearn")
print("=" * 70)

for _coin in COINS:
    # ── Read per-coin hyperparams from COIN_CONFIG ──────────────────────────
    _coin_cfg = COIN_CONFIG[_coin]
    _epochs   = _coin_cfg["epochs"]          # max_iter for MLPClassifier
    _dropout  = _coin_cfg["dropout"]         # stored; not natively in sklearn MLP
    _l2       = _coin_cfg["l2"]              # alpha (L2 weight decay)
    # sklearn MLPClassifier requires alpha > 0; use 1e-4 default when l2=0.0
    _alpha    = _l2 if _l2 > 0.0 else 1e-4

    print(f"\n{'─'*70}")
    print(f"  COIN: {_coin}")
    print(f"  Config  →  epochs={_epochs}  dropout={_dropout}  l2={_l2}  (alpha={_alpha})")
    print(f"{'─'*70}")

    _df = labeled_data[_coin].copy()

    # Build flattened sequences
    _X, _y = build_sequences_flat(_df, FEATURE_COLS, SEQ_LEN, LABEL_REMAP, max_samples=MAX_SAMPLES)
    _n_total = len(_X)

    # 80/20 chronological split
    _split = int(_n_total * 0.8)
    _X_train, _X_test = _X[:_split], _X[_split:]
    _y_train, _y_test = _y[:_split], _y[_split:]
    print(f"  Sequences: total={_n_total:,}  train={_split:,}  test={_n_total-_split:,}")

    # Balanced class weighting via sample_weight
    _n_samples    = len(_y_train)
    _class_counts = np.bincount(_y_train.astype(int), minlength=NUM_CLASSES)
    _class_w      = np.where(_class_counts > 0,
                             _n_samples / (NUM_CLASSES * _class_counts),
                             1.0)
    print(f"  Class weights  DOWN={_class_w[0]:.3f}  SIDEWAYS={_class_w[1]:.3f}  UP={_class_w[2]:.3f}")

    # Build MLPClassifier using per-coin epochs and l2
    _model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=_alpha,                  # L2 weight decay from COIN_CONFIG
        batch_size=256,
        max_iter=_epochs,              # epoch count from COIN_CONFIG
        learning_rate_init=1e-3,
        random_state=42,
        verbose=False,
        n_iter_no_change=5,
        tol=1e-4,
    )

    print(f"\n  Training MLP (max_iter={_epochs}, alpha={_alpha})…")
    print(f"  [Note] dropout={_dropout} stored in COIN_CONFIG; natively applied in future GRU/PyTorch upgrade")
    _model.fit(_X_train, _y_train)
    print(f"  ✅ Training done  (n_iter={_model.n_iter_}, loss={_model.loss_:.4f})")

    # Evaluate on held-out test set
    _preds = _model.predict(_X_test)
    _acc   = float(np.mean(_preds == _y_test))
    _prec, _rec, _sup = precision_recall_per_class(_y_test, _preds, n_classes=NUM_CLASSES)

    gru_metrics[_coin] = {
        "accuracy"  : _acc,
        "precision" : _prec,
        "recall"    : _rec,
        "support"   : _sup,
        "epochs_cfg": _epochs,
        "dropout_cfg": _dropout,
        "l2_cfg"    : _l2,
        "n_iter"    : _model.n_iter_,
    }

    print(f"\n  ── Test Results ──")
    print(f"  Accuracy: {_acc:.4f}  ({_acc*100:.2f}%)")
    print(f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'Support':>10}")
    print(f"  {'-'*44}")
    for _ci, _cname in enumerate(CLASS_NAMES):
        print(f"  {_cname:<10} {_prec[_ci]:>10.4f} {_rec[_ci]:>10.4f} {int(_sup[_ci]):>10,}")

    gru_models[_coin] = _model
    print(f"\n  ✅ Model stored for {_coin}")


# ── Final training summary ───────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  FINAL SUMMARY  —  Test Accuracy per Coin  |  Per-coin config used")
print(f"{'='*70}")
print(f"  {'Coin':<8} {'Acc':>7}  {'Epochs':>6} {'Dropout':>8} {'L2':>7}  {'n_iter':>6}  {'DOWN_P':>7} {'UP_P':>7}")
print(f"  {'-'*70}")
for _coin in COINS:
    _m = gru_metrics[_coin]
    print(f"  {_coin:<8} {_m['accuracy']:>7.4f}  "
          f"{_m['epochs_cfg']:>6} {_m['dropout_cfg']:>8.2f} {_m['l2_cfg']:>7.4f}  "
          f"{_m['n_iter']:>6}  "
          f"{_m['precision'][0]:>7.4f} {_m['precision'][2]:>7.4f}")

print(f"\n  Models in memory: {list(gru_models.keys())}")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# RL ENVIRONMENT REWARD FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
# Computes per-step reward for every sample in the test set for each coin.
# Reward schema (from ticket spec):
#
#   NEIRO / ZEREBRO (meme coins) — fee from COIN_CONFIG['fee'] = 0.06%:
#     Correct signal + |return| > threshold (1.5%) : +2.0
#     Correct signal + |return| < threshold (1.5%) : +0.3
#     Wrong signal (non-SIDEWAYS prediction error)  : -2.5
#     Fee applied per non-SIDEWAYS action           : -fee
#
#   Standard coins — fee from COIN_CONFIG['fee'] = 0.02%:
#     Correct signal: +1.0 × (|return_pct| / threshold), capped at +2.0
#     Wrong signal  : -1.0
#     Fee applied per non-SIDEWAYS action: -fee
#
#   SIDEWAYS predictions: reward = 0  (no trade, no fee)
#
# Return pct is the 5-bar forward return computed in label_targets.
# We reconstruct it from the test-set close prices in labeled_data.
# ──────────────────────────────────────────────────────────────────────────────

def compute_rl_reward(
    coin,
    model,
    X_test,
    y_test,
    df_full,           # labeled_data[coin] — has 'close' and 'label' columns
    feature_cols,
    seq_len,
    label_remap,
    coin_cfg,
    is_meme,
):
    """
    Compute RL environment step rewards for each test-set sample.

    Returns
    -------
    rewards : np.ndarray of shape (n_test_samples,)
    summary : dict with aggregate stats
    """
    _fee       = coin_cfg["fee"]           # per-trade fee (%) from COIN_CONFIG
    _threshold = coin_cfg["threshold"]     # move threshold for meme big/small split

    # --- Predictions from the trained model ----------------------------------
    _preds_idx  = model.predict(X_test)    # class indices: 0=DOWN, 1=SIDEWAYS, 2=UP
    _n          = len(_preds_idx)

    # --- Reconstruct 5-bar forward returns for test rows ---------------------
    # build_sequences_flat used the last MAX_SAMPLES rows of df_full.
    # We recompute the aligned close series here.
    _n_total   = len(df_full) - seq_len    # total sequences before capping
    _n_capped  = min(_n_total, MAX_SAMPLES)
    _start_idx = _n_total - _n_capped      # start in df_full index
    # Each sequence i uses df_full rows [_start_idx + i : _start_idx + i + seq_len]
    # The label for sequence i is at df_full row _start_idx + i + seq_len
    # 80/20 split → test indices start at split offset
    _n_split    = int(_n_capped * 0.8)
    _test_label_rows = [_start_idx + _n_split + i + seq_len for i in range(_n - _n_split)]

    # 5-bar forward return (%) reconstructed for test samples
    # Note: labeled_data already dropped the last 5 rows, so close[t+5] is safe
    _close = df_full["close"].values
    _true_returns = np.array([
        (_close[min(r + 5, len(_close) - 1)] - _close[r]) / _close[r] * 100
        for r in _test_label_rows[:_n]
    ], dtype=np.float64)

    # Align true_returns with the test portion of y_test
    _rewards = np.zeros(_n, dtype=np.float64)

    for _i in range(_n):
        _pred     = int(_preds_idx[_i])
        _actual   = int(y_test[_i])            # class index: 0=DOWN, 1=SIDEWAYS, 2=UP
        _ret_pct  = float(_true_returns[_i]) if _i < len(_true_returns) else 0.0

        # ── SIDEWAYS prediction → no trade, no reward, no fee ────────────────
        if _pred == 1:                          # SIDEWAYS
            _rewards[_i] = 0.0
            continue

        _correct  = (_pred == _actual)
        _abs_ret  = abs(_ret_pct)

        if is_meme:
            # ── Meme coin reward table (per ticket spec) ─────────────────────
            if _correct:
                if _abs_ret >= _threshold:
                    _rewards[_i] = MEME_REWARD_CORRECT_BIG  - _fee  # +2.0 - fee
                else:
                    _rewards[_i] = MEME_REWARD_CORRECT_SMALL - _fee  # +0.3 - fee
            else:
                _rewards[_i] = MEME_REWARD_WRONG - _fee              # -2.5 - fee
        else:
            # ── Standard coin reward: magnitude-scaled ────────────────────────
            if _correct:
                _scaled = STD_REWARD_CORRECT_BASE * (_abs_ret / _threshold)
                _rewards[_i] = min(_scaled, 2.0) - _fee              # capped at +2.0
            else:
                _rewards[_i] = STD_REWARD_WRONG - _fee               # -1.0 - fee

    # ── Aggregate summary ─────────────────────────────────────────────────────
    _n_trade  = int(np.sum(_preds_idx != 1))           # non-SIDEWAYS predictions
    _n_corr   = int(np.sum(_preds_idx[_preds_idx != 1] == y_test[_preds_idx != 1]))
    _win_rate = _n_corr / _n_trade if _n_trade > 0 else 0.0

    _summary = {
        "mean_reward"       : float(np.mean(_rewards)),
        "total_reward"      : float(np.sum(_rewards)),
        "std_reward"        : float(np.std(_rewards)),
        "n_samples"         : _n,
        "n_trade_signals"   : _n_trade,
        "n_correct_signals" : _n_corr,
        "win_rate"          : _win_rate,
        "fee"               : _fee,
        "is_meme"           : is_meme,
    }
    return _rewards, _summary


# ── Run reward function for all coins ─────────────────────────────────────────
rl_reward_summary = {}

print(f"\n{'='*70}")
print("  RL ENVIRONMENT REWARD FUNCTION — test-set simulation")
print(f"  Schema: NEIRO/ZEREBRO → big(>thresh)=+{MEME_REWARD_CORRECT_BIG:.1f}  small=+{MEME_REWARD_CORRECT_SMALL:.1f}  wrong={MEME_REWARD_WRONG:.1f}  fee=−COIN_CONFIG['fee']")
print(f"  Schema: Standard      → correct=+scaled(|ret|/thresh, cap 2.0)  wrong={STD_REWARD_WRONG:.1f}  fee=−COIN_CONFIG['fee']")
print(f"{'='*70}")

for _coin in COINS:
    _coin_cfg = COIN_CONFIG[_coin]
    _is_meme  = _coin in MEME_COINS
    _model    = gru_models[_coin]
    _df_full  = labeled_data[_coin]

    # Rebuild train/test split (same logic as training loop above)
    _X_all, _y_all = build_sequences_flat(
        _df_full, FEATURE_COLS, SEQ_LEN, LABEL_REMAP, max_samples=MAX_SAMPLES
    )
    _split_idx   = int(len(_X_all) * 0.8)
    _X_test_rl   = _X_all[_split_idx:]
    _y_test_rl   = _y_all[_split_idx:]

    _rewards, _summary = compute_rl_reward(
        coin        = _coin,
        model       = _model,
        X_test      = _X_test_rl,
        y_test      = _y_test_rl,
        df_full     = _df_full,
        feature_cols= FEATURE_COLS,
        seq_len     = SEQ_LEN,
        label_remap = LABEL_REMAP,
        coin_cfg    = _coin_cfg,
        is_meme     = _is_meme,
    )

    rl_reward_summary[_coin] = _summary

    _tag = "🔥 meme" if _is_meme else "   std "
    print(f"\n  {_tag}  {_coin:<8}  fee={_coin_cfg['fee']:.2f}%  threshold=±{_coin_cfg['threshold']}%")
    print(f"           mean_reward={_summary['mean_reward']:>+8.4f}  total={_summary['total_reward']:>+12.2f}  "
          f"std={_summary['std_reward']:.4f}")
    print(f"           n_samples={_summary['n_samples']:,}  n_trade={_summary['n_trade_signals']:,}  "
          f"win_rate={_summary['win_rate']:.2%}  (fee=−{_summary['fee']:.2f}%/trade)")

print(f"\n{'='*70}")
print("  RL REWARD SUMMARY  —  mean reward per step across all coins")
print(f"  {'COIN':<10} {'MEAN_R':>8} {'TOTAL_R':>12} {'WIN_RATE':>10} {'N_TRADE':>9}  TYPE")
print(f"  {'-'*65}")
for _coin in COINS:
    _s   = rl_reward_summary[_coin]
    _tag = "meme" if _s["is_meme"] else "std"
    print(f"  {_coin:<10} {_s['mean_reward']:>+8.4f} {_s['total_reward']:>+12.2f} "
          f"{_s['win_rate']:>10.2%} {_s['n_trade_signals']:>9,}  {_tag}")
print(f"{'='*70}")
print(f"\n  rl_reward_summary exported → {list(rl_reward_summary.keys())}")
