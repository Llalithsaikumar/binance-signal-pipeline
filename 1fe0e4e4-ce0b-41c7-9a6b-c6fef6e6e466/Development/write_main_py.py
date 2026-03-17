
# ── Write the complete standalone main.py to the canvas filesystem ────────────
MAIN_PY = '''#!/usr/bin/env python3
"""
main.py — Unified Crypto Futures Trading Pipeline
=================================================
Covers every stage end-to-end:
  1. CLI argument parsing
  2. Data loading from Binance Futures (public FAPI)
  3. Feature engineering for MLP and PPO
  4. MLP signal engine training & inference
  5. PPO RL agent training loop (pure NumPy, all hyperparameters)
  6. Backtesting / execution pass
  7. Saving outputs: signals_log.csv, PNG charts

Usage
-----
  python main.py --mode both                          # train + run full pipeline
  python main.py --mode train --episodes 100000       # training only
  python main.py --mode run                           # inference / backtest only
  python main.py --mode train --coins BTC ETH SOL     # subset of coins
  python main.py --mode both --output-dir /tmp/out    # custom output directory
"""

# ══════════════════════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe in headless env
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — CLI ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified Crypto Futures Trading Pipeline (MLP + PPO)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "run", "both"],
        default="both",
        help="Pipeline mode: train (MLP+PPO training only), run (inference+backtest only), both (default)",
    )
    parser.add_argument(
        "--coins",
        nargs="+",
        default=["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "NEIRO", "ZEREBRO"],
        metavar="COIN",
        help="Coins to process (default: all 8)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100_000,
        help="Total PPO training timesteps (default: 100 000)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for output files: signals_log.csv, PNG charts (default: .)",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — COIN CONFIG & GLOBAL CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

INTERVAL  = "1m"
LIMIT     = 1000      # rows per Binance API request (max 1500)
BASE_URL  = "https://fapi.binance.com/fapi/v1/klines"

def _build_coin_config(coins: list) -> dict:
    """Return per-coin hyperparameter config for the requested coin list."""
    now_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_6m = int((datetime.now(timezone.utc) - timedelta(days=180)).timestamp() * 1000)
    neiro_start   = int(datetime(2024, 9, 17, tzinfo=timezone.utc).timestamp() * 1000)
    zerebro_start = int(datetime(2025,  1,  3, tzinfo=timezone.utc).timestamp() * 1000)

    ALL_CONFIG = {
        "BTC":    {"symbol": "BTCUSDT",     "start": start_6m,      "threshold": 0.3,  "epochs": 30, "dropout": 0.20, "l2": 0.0,   "fee": 0.02, "confidence_threshold": 0.65},
        "ETH":    {"symbol": "ETHUSDT",     "start": start_6m,      "threshold": 0.3,  "epochs": 30, "dropout": 0.20, "l2": 0.0,   "fee": 0.02, "confidence_threshold": 0.65},
        "SOL":    {"symbol": "SOLUSDT",     "start": start_6m,      "threshold": 0.3,  "epochs": 30, "dropout": 0.20, "l2": 0.0,   "fee": 0.02, "confidence_threshold": 0.65},
        "BNB":    {"symbol": "BNBUSDT",     "start": start_6m,      "threshold": 0.3,  "epochs": 30, "dropout": 0.20, "l2": 0.0,   "fee": 0.02, "confidence_threshold": 0.65},
        "XRP":    {"symbol": "XRPUSDT",     "start": start_6m,      "threshold": 0.3,  "epochs": 30, "dropout": 0.20, "l2": 0.0,   "fee": 0.02, "confidence_threshold": 0.65},
        "DOGE":   {"symbol": "DOGEUSDT",    "start": start_6m,      "threshold": 0.3,  "epochs": 30, "dropout": 0.20, "l2": 0.0,   "fee": 0.02, "confidence_threshold": 0.65},
        "NEIRO":  {"symbol": "NEIROUSDT",   "start": neiro_start,   "threshold": 1.5,  "epochs": 20, "dropout": 0.35, "l2": 0.001, "fee": 0.06, "confidence_threshold": 0.75},
        "ZEREBRO":{"symbol": "ZEREBROUSDT", "start": zerebro_start, "threshold": 1.5,  "epochs": 20, "dropout": 0.35, "l2": 0.001, "fee": 0.06, "confidence_threshold": 0.75},
    }
    return {c: ALL_CONFIG[c] for c in coins if c in ALL_CONFIG}, now_ms


MEME_COINS = {"NEIRO", "ZEREBRO"}

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
]

MLP_FEATURE_COLS = [
    "rsi_14", "ema_9", "ema_21",
    "macd_line", "macd_signal", "macd_histogram",
    "bb_upper", "bb_lower", "bb_width", "bb_pct_b",
    "vol_sma_20", "vol_delta",
    "body_size_norm", "roc_5",
]

PPO_FEATURE_COLS = [
    "rsi_14", "macd_line", "macd_histogram",
    "bb_width", "bb_pct_b",
    "vol_delta", "body_size_norm", "roc_5",
    "log_return", "vol_20", "vol_5", "mom_5", "mom_20",
]

PPO_ACTION_MAP   = {0: "SELL", 1: "HOLD", 2: "BUY"}
LABEL_MAP        = {1: "UP", 0: "SIDEWAYS", -1: "DOWN"}
MLP_LABEL_REMAP  = {-1: 0, 0: 1, 1: 2}
MLP_CLASS_NAMES  = ["DOWN", "SIDEWAYS", "UP"]

SEQ_LEN     = 30
MAX_SAMPLES = 30_000
NUM_CLASSES = 3

# Zerve design system colours
BG_COLOR    = "#1D1D20"
TEXT_COLOR  = "#fbfbff"
MUTED_COLOR = "#909094"
CHART_COLS  = ["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF",
               "#1F77B4", "#9467BD", "#8C564B"]
GOLD  = "#ffd400"
GREEN = "#17b26a"
RED   = "#f04438"


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def fetch_klines(symbol: str, start_ms: int, now_ms: int) -> pd.DataFrame:
    """Page through Binance FAPI 1-min klines from start_ms to now."""
    all_rows = []
    cursor   = start_ms

    while cursor < now_ms:
        params = urllib.parse.urlencode({
            "symbol":    symbol,
            "interval":  INTERVAL,
            "startTime": cursor,
            "endTime":   now_ms,
            "limit":     LIMIT,
        })
        with urllib.request.urlopen(f"{BASE_URL}?{params}", timeout=30) as resp:
            batch = json.loads(resp.read())

        if not batch:
            break
        all_rows.extend(batch)
        cursor = batch[-1][0] + 60_000
        if len(batch) < LIMIT:
            break

    df = pd.DataFrame(all_rows, columns=KLINE_COLUMNS)
    numeric_cols = ["open", "high", "low", "close", "volume",
                    "quote_volume", "taker_buy_base_vol", "taker_buy_quote_vol"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df.drop(columns=["ignore"], inplace=True)
    return df


def load_all_candles(coin_config: dict, now_ms: int) -> dict:
    """Fetch OHLCV for every coin in parallel. Returns {coin: DataFrame}."""
    print("\\n[1/7] Loading Binance Futures candles (parallel)…")
    raw_dfs = {}

    def _fetch(coin, cfg):
        return coin, fetch_klines(cfg["symbol"], cfg["start"], now_ms)

    with ThreadPoolExecutor(max_workers=len(coin_config)) as pool:
        futures = {pool.submit(_fetch, c, cfg): c for c, cfg in coin_config.items()}
        for fut in as_completed(futures):
            coin, df = fut.result()
            raw_dfs[coin] = df
            print(f"  {coin:<8}  {len(df):>8,} rows  "
                  f"{df['open_time'].iloc[0].date()} → {df['open_time'].iloc[-1].date()}")

    return raw_dfs


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_mlp_features(raw: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Compute MLP feature set (RSI, EMA, MACD, BB, volume, ROC)."""
    df     = raw.copy()
    close  = df["close"]
    vol    = df["volume"]
    open_  = df["open"]

    df["rsi_14"]  = _rsi(close, 14)
    df["ema_9"]   = close.ewm(span=9,  adjust=False).mean()
    df["ema_21"]  = close.ewm(span=21, adjust=False).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd_line"]      = ema12 - ema26
    df["macd_signal"]    = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd_line"] - df["macd_signal"]

    bb_mid           = close.rolling(20).mean()
    bb_std           = close.rolling(20).std(ddof=0)
    df["bb_upper"]   = bb_mid + 2 * bb_std
    df["bb_lower"]   = bb_mid - 2 * bb_std
    df["bb_width"]   = ((df["bb_upper"] - df["bb_lower"]) / bb_mid) * 100
    df["bb_pct_b"]   = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    df["vol_sma_20"]      = vol.rolling(20).mean()
    df["vol_delta"]       = vol - df["vol_sma_20"]
    df["body_size_norm"]  = (close - open_).abs() / close
    df["roc_5"]           = close.pct_change(periods=5) * 100
    df["label_threshold"] = threshold

    return df.dropna().reset_index(drop=True)


def add_labels(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Add 5-bar forward-return UP/SIDEWAYS/DOWN label."""
    threshold     = float(df["label_threshold"].iloc[0])
    close_ahead   = df["close"].shift(-horizon)
    future_return = (close_ahead - df["close"]) / df["close"] * 100
    df["label"]   = np.where(
        future_return >  threshold,  1,
        np.where(future_return < -threshold, -1, 0)
    )
    return df.iloc[:-horizon].copy()


def add_ppo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add return-based features for PPO (log returns, vol, momentum)."""
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["vol_20"]     = df["log_return"].rolling(20).std()
    df["vol_5"]      = df["log_return"].rolling(5).std()
    df["mom_5"]      = df["log_return"].rolling(5).sum()
    df["mom_20"]     = df["log_return"].rolling(20).sum()
    fwd              = df["log_return"].shift(-1)
    df["ppo_action"] = np.where(fwd > 0, 2, np.where(fwd < 0, 0, 1)).astype(int)
    return df.dropna(subset=["log_return", "vol_20", "vol_5", "mom_5", "mom_20"]).iloc[:-1].copy()


def build_all_features(raw_dfs: dict, coin_config: dict) -> tuple:
    """Run full feature engineering. Returns (mlp_data, ppo_matrices)."""
    print("\\n[2/7] Feature engineering…")
    mlp_data     = {}   # coin → labeled DataFrame with MLP features
    ppo_matrices = {}   # coin → DataFrame with PPO features

    for coin, df in raw_dfs.items():
        thresh         = coin_config[coin]["threshold"]
        featured       = compute_mlp_features(df, thresh)
        labeled        = add_labels(featured)
        mlp_data[coin] = labeled

        ppo_src            = add_ppo_features(labeled)
        ppo_matrices[coin] = ppo_src

        print(f"  {coin:<8}  MLP rows={len(labeled):>8,}  PPO rows={len(ppo_src):>8,}")

    return mlp_data, ppo_matrices


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — MLP SIGNAL ENGINE: TRAINING & INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def build_sequences_flat(df: pd.DataFrame, feature_cols: list,
                         seq_len: int, label_remap: dict,
                         max_samples: int = None):
    """Flatten (seq_len × n_feat) windows into 1-D vectors with z-score norm."""
    X_raw = df[feature_cols].values.astype(np.float32)
    y_raw = df["label"].map(label_remap).values.astype(np.int64)

    mean  = X_raw.mean(axis=0)
    std   = X_raw.std(axis=0) + 1e-8
    X_raw = (X_raw - mean) / std

    N  = len(X_raw) - seq_len
    X  = np.empty((N, seq_len * len(feature_cols)), dtype=np.float32)
    y  = np.empty(N, dtype=np.int64)
    for i in range(N):
        X[i] = X_raw[i: i + seq_len].ravel()
        y[i] = y_raw[i + seq_len]

    if max_samples is not None and N > max_samples:
        X = X[-max_samples:]
        y = y[-max_samples:]
    return X, y


def precision_recall(y_true, y_pred, n_classes=3):
    prec = np.zeros(n_classes)
    rec  = np.zeros(n_classes)
    sup  = np.zeros(n_classes, dtype=int)
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec[c]  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        sup[c]  = int(np.sum(y_true == c))
    return prec, rec, sup


def train_mlp_models(mlp_data: dict, coin_config: dict) -> tuple:
    """Train one MLPClassifier per coin. Returns (models, metrics)."""
    print("\\n[3/7] Training MLP classifiers…")
    models  = {}
    metrics = {}

    for coin in mlp_data:
        cfg      = coin_config[coin]
        epochs   = cfg["epochs"]
        alpha    = cfg["l2"] if cfg["l2"] > 0 else 1e-4

        X, y     = build_sequences_flat(mlp_data[coin], MLP_FEATURE_COLS,
                                        SEQ_LEN, MLP_LABEL_REMAP, MAX_SAMPLES)
        split    = int(len(X) * 0.8)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        # Balanced class weights
        n        = len(y_tr)
        counts   = np.bincount(y_tr.astype(int), minlength=NUM_CLASSES)
        weights  = np.where(counts > 0, n / (NUM_CLASSES * counts), 1.0)
        sample_w = weights[y_tr]

        model = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=alpha,
            batch_size=256,
            max_iter=epochs,
            learning_rate_init=1e-3,
            random_state=42,
            verbose=False,
            n_iter_no_change=5,
            tol=1e-4,
        )
        model.fit(X_tr, y_tr, sample_weight=sample_w)
        preds        = model.predict(X_te)
        acc          = float(np.mean(preds == y_te))
        prec, rec, sup = precision_recall(y_te, preds)

        models[coin]  = model
        metrics[coin] = {"accuracy": acc, "precision": prec, "recall": rec,
                         "support": sup, "n_iter": model.n_iter_}
        print(f"  {coin:<8}  acc={acc:.4f}  n_iter={model.n_iter_}  "
              f"DOWN_P={prec[0]:.3f}  UP_P={prec[2]:.3f}")

    return models, metrics


# ── Live-style inference (used in run / backtest mode) ───────────────────────

def _build_inference_vector(feat_df: pd.DataFrame) -> np.ndarray:
    """Last SEQ_LEN rows → z-scored flat vector (1, SEQ_LEN × 14)."""
    arr  = feat_df[MLP_FEATURE_COLS].values[-SEQ_LEN:].astype(np.float32)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0) + 1e-8
    return ((arr - mean) / std).ravel()


def run_mlp_inference(mlp_data: dict, models: dict,
                      coin_config: dict, output_dir: str) -> list:
    """
    Run MLP inference over the held-out test set for each coin.
    Uses the most recent SEQ_LEN rows of each coin\'s feature data as
    a single \'live\' snapshot — mirrors the live_signal_engine logic.
    Returns signals_list and writes signals_log.csv.
    """
    print("\\n[5/7] MLP inference & signal generation…")
    signals = []

    for coin, df in mlp_data.items():
        if coin not in models:
            print(f"  {coin:<8}  no model — skipped")
            continue

        model      = models[coin]
        cfg        = coin_config[coin]
        conf_thresh= cfg["confidence_threshold"]

        if len(df) < SEQ_LEN:
            print(f"  {coin:<8}  not enough rows ({len(df)}) — skipped")
            continue

        x_vec  = _build_inference_vector(df)
        probs_raw = model.predict_proba(x_vec.reshape(1, -1))[0]
        prob_full = np.zeros(NUM_CLASSES, dtype=np.float64)
        prob_full[model.classes_.astype(int)] = probs_raw

        class_idx  = int(np.argmax(prob_full))
        confidence = float(prob_full[class_idx])
        signal     = MLP_CLASS_NAMES[class_idx]
        price      = float(df["close"].iloc[-1])
        ts         = str(df.index[-1]) if "open_time" not in df.columns else str(df["open_time"].iloc[-1])
        emitted    = confidence > conf_thresh

        print(f"  {coin:<8}  signal={signal:<9}  conf={confidence:.1%}  "
              f"gate={conf_thresh:.0%}  price={price:,.4f}  "
              f"emit={'YES' if emitted else 'no'}")

        if emitted:
            signals.append({
                "timestamp": ts, "coin": coin, "signal": signal,
                "confidence": round(confidence, 4), "price": price,
            })

    # Write signals_log.csv
    log_path = os.path.join(output_dir, "signals_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "coin", "signal", "confidence", "price"])
        writer.writeheader()
        writer.writerows(signals)
    print(f"  Wrote {len(signals)} signal(s) → {log_path}")
    return signals


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — TRADING GYM ENVIRONMENT (pure NumPy, no gymnasium dependency)
# ══════════════════════════════════════════════════════════════════════════════

class _DiscreteSpace:
    def __init__(self, n):
        self.n   = n
        self._rng = np.random.default_rng()
    def sample(self): return int(self._rng.integers(0, self.n))
    def contains(self, x): return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n
    def seed(self, seed=None): self._rng = np.random.default_rng(seed)


class _BoxSpace:
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low   = np.full(shape, low,  dtype=dtype)
        self.high  = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype
        self._rng  = np.random.default_rng()
    def contains(self, x):
        x = np.asarray(x, dtype=self.dtype)
        return x.shape == self.shape and not np.any(np.isnan(x))
    def seed(self, seed=None): self._rng = np.random.default_rng(seed)


_POSITION_MAP = {0: -1.0, 1: 0.0, 2: 1.0}


class TradingEnv:
    """
    Gym-compatible crypto trading environment (no external gym required).
    State  : 13-dim feature vector
    Actions: 0=SELL | 1=HOLD | 2=BUY
    Reward : position-weighted log return − transaction_cost × |Δposition|
    """

    def __init__(self, df: pd.DataFrame, transaction_cost: float = 0.001, coin: str = "BTC"):
        missing = [c for c in PPO_FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"TradingEnv: missing columns {missing}")

        self.df               = df.reset_index(drop=True)
        self.transaction_cost = transaction_cost
        self.coin             = coin
        self._n               = len(self.df)
        self._obs             = np.nan_to_num(df[PPO_FEATURE_COLS].values.astype(np.float32))
        self._log_ret         = df["log_return"].values.astype(np.float64)

        self.observation_space = _BoxSpace(-np.inf, np.inf, (len(PPO_FEATURE_COLS),), np.float32)
        self.action_space      = _DiscreteSpace(3)

        self._step     = 0
        self._position = 0.0
        self._total_r  = 0.0
        self._ep_rewards = []

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.action_space.seed(seed)
        self._step, self._position, self._total_r, self._ep_rewards = 0, 0.0, 0.0, []
        return self._obs[0].copy(), {"step": 0, "coin": self.coin}

    def step(self, action: int):
        new_pos = _POSITION_MAP[int(action)]
        log_r   = self._log_ret[self._step]
        reward  = float(self._position * log_r - self.transaction_cost * abs(new_pos - self._position))
        self._position = new_pos
        self._total_r += reward
        self._ep_rewards.append(reward)
        self._step += 1
        done = self._step >= self._n
        obs  = self._obs[self._step].copy() if not done else np.zeros(len(PPO_FEATURE_COLS), np.float32)
        return obs, reward, done, False, {"step": self._step, "coin": self.coin}

    def close(self): pass


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — PPO AGENT (Pure NumPy, MlpPolicy equivalent)
# ══════════════════════════════════════════════════════════════════════════════

def _relu(x):   return np.maximum(0.0, x)
def _relu_d(x): return (x > 0).astype(np.float64)
def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)


class MLP:
    """Lightweight fully-connected net (He init, NumPy only)."""

    def __init__(self, layer_sizes: list, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.weights, self.biases, self._cache = [], [], []
        for i in range(len(layer_sizes) - 1):
            fin, fout = layer_sizes[i], layer_sizes[i + 1]
            self.weights.append(rng.standard_normal((fin, fout)) * np.sqrt(2.0 / fin))
            self.biases.append(np.zeros(fout))

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = []
        h = x.copy()
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ W + b
            self._cache.append((h, z))
            h = _relu(z) if i < len(self.weights) - 1 else z
        return h


class PPOAgent:
    """
    PPO with clipped surrogate objective.
    Hyperparameters mirror stable-baselines3 MlpPolicy defaults.

    Parameters
    ----------
    obs_dim      : observation vector dimension
    n_actions    : number of discrete actions
    lr           : Adam learning rate (default 3e-4)
    gamma        : discount factor (default 0.99)
    gae_lambda   : GAE lambda (default 0.95)
    clip_range   : PPO clip ε (default 0.2)
    ent_coef     : entropy coefficient (default 0.01)
    n_epochs     : PPO epochs per rollout (default 10)
    batch_size   : mini-batch size (default 64)
    n_steps      : rollout horizon (default 2048)
    """

    def __init__(self, obs_dim: int, n_actions: int, seed: int = 42,
                 lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_range: float = 0.2, ent_coef: float = 0.01,
                 n_epochs: int = 10, batch_size: int = 64, n_steps: int = 2048):
        hidden = [64, 64]
        self.actor  = MLP([obs_dim] + hidden + [n_actions], seed=seed)
        self.critic = MLP([obs_dim] + hidden + [1],         seed=seed + 1)

        self.lr, self.gamma, self.lam = lr, gamma, gae_lambda
        self.clip, self.ent_coef      = clip_range, ent_coef
        self.n_epochs, self.batch_size, self.n_steps = n_epochs, batch_size, n_steps
        self.n_actions = n_actions
        self.rng       = np.random.default_rng(seed)

        def _adam(net):
            sz = sum(W.size + b.size for W, b in zip(net.weights, net.biases))
            return np.zeros(sz), np.zeros(sz)

        self._m_a, self._v_a = _adam(self.actor)
        self._m_c, self._v_c = _adam(self.critic)
        self._t = 0

    def get_action_and_value(self, obs: np.ndarray):
        logits = self.actor.forward(obs[np.newaxis, :])[0]
        probs  = _softmax(logits)
        action = int(self.rng.choice(self.n_actions, p=probs))
        return action, float(np.log(probs[action] + 1e-12)), float(self.critic.forward(obs[np.newaxis, :])[0, 0])

    def predict(self, obs: np.ndarray) -> int:
        return int(np.argmax(self.actor.forward(obs[np.newaxis, :])[0]))

    def compute_gae(self, rewards, values, dones, last_value):
        n, adv, gae = len(rewards), np.zeros(len(rewards)), 0.0
        for t in reversed(range(n)):
            nxt   = last_value if t == n - 1 else values[t + 1]
            mask  = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * nxt * mask - values[t]
            gae   = delta + self.gamma * self.lam * mask * gae
            adv[t] = gae
        return adv, adv + values

    def _backprop(self, net: MLP, dout: np.ndarray) -> np.ndarray:
        grads, delta = [], dout
        for i in reversed(range(len(net.weights))):
            h, z = net._cache[i]
            if i < len(net.weights) - 1:
                delta = delta * _relu_d(z)
            grads.insert(0, (h.T @ delta, delta.sum(axis=0)))
            delta = delta @ net.weights[i].T
        return np.concatenate([np.concatenate([gW.ravel(), gb]) for gW, gb in grads])

    def _actor_grad(self, obs_b, acts_b, adv_b, old_lp_b):
        logits = self.actor.forward(obs_b)
        probs  = _softmax(logits)
        lp     = np.log(probs[np.arange(len(acts_b)), acts_b] + 1e-12)
        ratio  = np.exp(lp - old_lp_b)
        clipped = ((adv_b > 0) & (ratio > 1 + self.clip)) | ((adv_b < 0) & (ratio < 1 - self.clip))
        pg_coef  = -np.where(clipped, 0.0, adv_b)
        oh = np.zeros_like(probs); oh[np.arange(len(acts_b)), acts_b] = 1.0
        dlogits = (oh - probs) * pg_coef[:, np.newaxis]
        ent_g   = probs * (np.log(probs + 1e-12) + 1.0) - probs
        return self._backprop(self.actor, (dlogits - self.ent_coef * ent_g) / len(obs_b))

    def _critic_grad(self, obs_b, returns_b):
        v  = self.critic.forward(obs_b)[:, 0]
        dv = 2.0 * (v - returns_b) / len(obs_b)
        return self._backprop(self.critic, dv[:, np.newaxis])

    def _adam_step(self, net, grad, m, v, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        m[:] = beta1 * m + (1 - beta1) * grad
        v[:] = beta2 * v + (1 - beta2) * grad ** 2
        upd  = self.lr * (m / (1 - beta1 ** self._t)) / (np.sqrt(v / (1 - beta2 ** self._t)) + eps)
        idx  = 0
        for i in range(len(net.weights)):
            sz = net.weights[i].size; net.weights[i] -= upd[idx:idx + sz].reshape(net.weights[i].shape); idx += sz
            sz = net.biases[i].size;  net.biases[i]  -= upd[idx:idx + sz]; idx += sz

    def train(self, env, total_timesteps: int) -> list:
        log, obs, _ = [], np.array(env.reset(seed=42)[0], dtype=np.float64), None
        ep_rewards, ep_cur, steps = [], 0.0, 0
        print(f"    {\'Timestep\':>10}  {\'MeanEpReward\':>14}  {\'Episodes\':>9}")
        print(f"    {\'─\' * 40}")
        while steps < total_timesteps:
            n  = self.n_steps
            ob = np.zeros((n, obs.shape[0]))
            ac = np.zeros(n, dtype=np.int32)
            rw = np.zeros(n); vl = np.zeros(n); lp = np.zeros(n); dn = np.zeros(n, dtype=bool)
            for t in range(n):
                a, lpa, val = self.get_action_and_value(obs)
                ob[t], ac[t], lp[t], vl[t] = obs, a, lpa, val
                obs_n, r, term, trunc, _ = env.step(a)
                rw[t], dn[t] = r, term or trunc
                ep_cur += r; steps += 1
                if dn[t]:
                    ep_rewards.append(ep_cur); ep_cur = 0.0
                    obs, _ = env.reset(); obs = np.array(obs, dtype=np.float64)
                else:
                    obs = np.array(obs_n, dtype=np.float64)
            _, _, lv = self.get_action_and_value(obs)
            adv, ret = self.compute_gae(rw, vl, dn, lv)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            for _ in range(self.n_epochs):
                perm = self.rng.permutation(n)
                for s in range(0, n, self.batch_size):
                    b = perm[s:s + self.batch_size]
                    self._adam_step(self.actor,  self._actor_grad(ob[b], ac[b], adv[b], lp[b]), self._m_a, self._v_a)
                    self._adam_step(self.critic, self._critic_grad(ob[b], ret[b]), self._m_c, self._v_c)
            if ep_rewards:
                mr = float(np.mean(ep_rewards[-20:]))
                log.append((steps, mr))
                if len(log) % 5 == 0 or steps >= total_timesteps:
                    print(f"    {steps:>10,}  {mr:>+14.6f}  {len(ep_rewards):>9,}")
        env.close()
        return log


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — PPO TRAINING DRIVER
# ══════════════════════════════════════════════════════════════════════════════

def train_ppo_agent(ppo_matrices: dict, coin: str = "BTC",
                    total_timesteps: int = 100_000,
                    transaction_cost: float = 0.001) -> tuple:
    """Train PPO on the specified coin. Returns (agent, reward_log)."""
    print(f"\\n[4/7] Training PPO agent on {coin} ({total_timesteps:,} timesteps)…")

    required = PPO_FEATURE_COLS + ["ppo_action"]
    mat = (ppo_matrices[coin][required].dropna().reset_index(drop=True))
    n   = len(mat)
    split = int(n * 0.80)
    train_df = mat.iloc[:split].reset_index(drop=True)
    test_df  = mat.iloc[split:].reset_index(drop=True)

    print(f"  Total rows: {n:,}  |  Train: {len(train_df):,}  |  Test: {len(test_df):,}")
    print(f"  Features  : {len(PPO_FEATURE_COLS)}-dim  |  Tx cost: {transaction_cost*100:.1f}%")

    env   = TradingEnv(df=train_df, transaction_cost=transaction_cost, coin=coin)
    agent = PPOAgent(
        obs_dim    = len(PPO_FEATURE_COLS), n_actions = 3, seed = 42,
        lr         = 3e-4, gamma = 0.99, gae_lambda = 0.95,
        clip_range = 0.2,  ent_coef = 0.01,
        n_epochs   = 10,   batch_size = 64, n_steps = 2048,
    )
    reward_log = agent.train(env, total_timesteps=total_timesteps)
    print(f"  ✅ PPO training complete — {total_timesteps:,} timesteps")
    return agent, reward_log, test_df


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — BACKTESTING / EXECUTION PASS
# ══════════════════════════════════════════════════════════════════════════════

def backtest_ppo(agent: PPOAgent, test_df: pd.DataFrame,
                 coin: str = "BTC", transaction_cost: float = 0.001) -> dict:
    """Run the trained PPO agent over the test set. Returns metrics dict."""
    print(f"\\n[6/7] Backtesting PPO agent on {coin} test set…")
    env   = TradingEnv(df=test_df, transaction_cost=transaction_cost, coin=coin)
    obs, _ = env.reset(seed=0)
    obs    = np.array(obs, dtype=np.float64)

    step_rewards, actions = [], []
    done = False
    while not done:
        act       = agent.predict(obs)
        obs_n, r, term, trunc, _ = env.step(act)
        step_rewards.append(r); actions.append(act)
        done = term or trunc
        obs  = np.array(obs_n, dtype=np.float64)
    env.close()

    step_rewards = np.array(step_rewards, dtype=np.float64)
    actions      = np.array(actions, dtype=np.int32)
    cum_ppo      = np.cumsum(step_rewards)
    cum_bnh      = np.cumsum(test_df["log_return"].values[:len(step_rewards)])

    mean_r = np.mean(step_rewards)
    std_r  = np.std(step_rewards) + 1e-12
    sharpe = (mean_r / std_r) * np.sqrt(365 * 24 * 60)

    metrics = {
        "coin":           coin,
        "total_steps":    len(step_rewards),
        "total_return":   float(cum_ppo[-1]),
        "bnh_return":     float(cum_bnh[-1]),
        "sharpe":         float(sharpe),
        "win_rate":       float(np.mean(step_rewards > 0)),
        "action_counts":  {PPO_ACTION_MAP[i]: int(np.sum(actions == i)) for i in range(3)},
        "cum_ppo":        cum_ppo,
        "cum_bnh":        cum_bnh,
        "step_rewards":   step_rewards,
        "actions":        actions,
    }

    print(f"  Total Return  : {metrics[\'total_return\']:>+.6f}")
    print(f"  B&H  Return   : {metrics[\'bnh_return\']:>+.6f}")
    print(f"  Sharpe Ratio  : {metrics[\'sharpe\']:>+.4f}")
    print(f"  Win Rate      : {metrics[\'win_rate\']*100:.2f}%")
    print(f"  SELL/HOLD/BUY : "
          f"{metrics[\'action_counts\'][\'SELL\']}/{metrics[\'action_counts\'][\'HOLD\']}/{metrics[\'action_counts\'][\'BUY\']}")
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — CHART GENERATION & OUTPUT SAVING
# ══════════════════════════════════════════════════════════════════════════════

def _apply_zerve_style():
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR, "axes.facecolor":   BG_COLOR,
        "text.color":       TEXT_COLOR, "axes.labelcolor": TEXT_COLOR,
        "xtick.color":      TEXT_COLOR, "ytick.color":    TEXT_COLOR,
        "axes.edgecolor":   MUTED_COLOR,"grid.color":      "#33333a",
        "axes.titlesize":   13, "axes.labelsize": 11,
        "font.family":      "sans-serif",
    })


def save_charts(bt_metrics: dict, reward_log: list, output_dir: str):
    """Generate and save the three standard PPO output charts."""
    print(f"\\n[7/7] Saving charts to {output_dir}…")
    _apply_zerve_style()

    coin     = bt_metrics["coin"]
    cum_ppo  = bt_metrics["cum_ppo"]
    cum_bnh  = bt_metrics["cum_bnh"]
    actions  = bt_metrics["actions"]
    ac       = np.bincount(actions, minlength=3)
    nt       = ac.sum()

    # ── Chart 1: Cumulative Return vs Buy-and-Hold ───────────────────────────
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(cum_ppo, color=CHART_COLS[0], lw=1.8, label="PPO Agent",   alpha=0.95)
    ax1.plot(cum_bnh, color=CHART_COLS[1], lw=1.5, label="Buy & Hold",  alpha=0.85, ls="--")
    ax1.axhline(0, color=MUTED_COLOR, lw=0.6, ls=":")
    ax1.set_title(f"Cumulative Return — PPO vs Buy & Hold  ({coin} · 20% Test)",
                  color=TEXT_COLOR, pad=14)
    ax1.set_xlabel("Test Step (1-min bars)"); ax1.set_ylabel("Cumulative Log Return")
    ax1.legend(facecolor=BG_COLOR, edgecolor=MUTED_COLOR, labelcolor=TEXT_COLOR, fontsize=10)
    ax1.grid(alpha=0.25)
    plt.tight_layout()
    p1 = os.path.join(output_dir, "ppo_cumulative_return.png")
    fig1.savefig(p1, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig1)
    print(f"  Saved: {p1}")

    # ── Chart 2: Training Reward Curve ───────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    if len(reward_log) > 1:
        steps_log = [x[0] for x in reward_log]
        rews_log  = [x[1] for x in reward_log]
        ax2.plot(steps_log, rews_log, color=CHART_COLS[2], lw=1.8, alpha=0.9,
                 label="Mean Ep Reward (trailing 20)")
        w  = max(3, len(rews_log) // 5)
        sm = pd.Series(rews_log).rolling(w, min_periods=1).mean().values
        ax2.plot(steps_log, sm, color=GOLD, lw=2.4, ls="--", label=f"Trend ({w}-pt MA)")
        ax2.axhline(0, color=MUTED_COLOR, lw=0.6, ls=":")
        ax2.legend(facecolor=BG_COLOR, edgecolor=MUTED_COLOR, labelcolor=TEXT_COLOR, fontsize=10)
    else:
        ax2.text(0.5, 0.5, "No reward data recorded", transform=ax2.transAxes,
                 ha="center", va="center", color=MUTED_COLOR, fontsize=12)
    ax2.set_title(f"PPO Training Reward Curve  ({coin})", color=TEXT_COLOR, pad=14)
    ax2.set_xlabel("Timestep"); ax2.set_ylabel("Mean Episode Reward")
    ax2.grid(alpha=0.25)
    plt.tight_layout()
    p2 = os.path.join(output_dir, "ppo_training_curve.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig2)
    print(f"  Saved: {p2}")

    # ── Chart 3: Action Distribution ─────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(7, 7))
    ax3.pie(
        ac,
        labels=[f"SELL\\n{ac[0]:,} ({ac[0]/nt*100:.1f}%)",
                f"HOLD\\n{ac[1]:,} ({ac[1]/nt*100:.1f}%)",
                f"BUY\\n{ac[2]:,} ({ac[2]/nt*100:.1f}%)"],
        colors=[RED, MUTED_COLOR, GREEN],
        explode=[0.04, 0.04, 0.04], startangle=90,
        textprops={"color": TEXT_COLOR, "fontsize": 12, "fontweight": "bold"},
        wedgeprops={"edgecolor": BG_COLOR, "linewidth": 2},
    )
    ax3.set_title("PPO Agent — Action Distribution (Test Set)", color=TEXT_COLOR, pad=18, fontsize=13)
    plt.tight_layout()
    p3 = os.path.join(output_dir, "ppo_action_distribution.png")
    fig3.savefig(p3, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig3)
    print(f"  Saved: {p3}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Validate coin names
    valid_all = {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "NEIRO", "ZEREBRO"}
    coins     = [c.upper() for c in args.coins]
    invalid   = [c for c in coins if c not in valid_all]
    if invalid:
        print(f"ERROR: Unknown coins {invalid}. Valid: {sorted(valid_all)}", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 72)
    print("  CRYPTO FUTURES TRADING PIPELINE  (MLP + PPO)")
    print("=" * 72)
    print(f"  Mode       : {args.mode}")
    print(f"  Coins      : {coins}")
    print(f"  Episodes   : {args.episodes:,}")
    print(f"  Output dir : {args.output_dir}")
    print("=" * 72)

    coin_config, now_ms = _build_coin_config(coins)

    # ── TRAIN MODE ───────────────────────────────────────────────────────────
    mlp_models, mlp_metrics = {}, {}
    ppo_agent, reward_log, test_df = None, [], None
    mlp_data, ppo_matrices = {}, {}
    bt_metrics = {}

    if args.mode in ("train", "both"):
        raw_dfs            = load_all_candles(coin_config, now_ms)
        mlp_data, ppo_matrices = build_all_features(raw_dfs, coin_config)
        mlp_models, mlp_metrics = train_mlp_models(mlp_data, coin_config)

        # PPO trains on first coin in list (default BTC)
        ppo_coin    = coins[0]
        ppo_agent, reward_log, test_df = train_ppo_agent(
            ppo_matrices, coin=ppo_coin, total_timesteps=args.episodes,
        )

    # ── RUN MODE: data needed even if skipping training ──────────────────────
    if args.mode == "run" and not mlp_data:
        raw_dfs              = load_all_candles(coin_config, now_ms)
        mlp_data, ppo_matrices = build_all_features(raw_dfs, coin_config)

    # ── INFERENCE / BACKTEST ─────────────────────────────────────────────────
    if args.mode in ("run", "both"):
        signals = run_mlp_inference(mlp_data, mlp_models, coin_config, args.output_dir)

        if ppo_agent is not None and test_df is not None:
            bt_metrics = backtest_ppo(ppo_agent, test_df, coin=coins[0])
            save_charts(bt_metrics, reward_log, args.output_dir)
        else:
            print("\\n[6-7/7] Skipping PPO backtest & charts — no trained agent in \'run\' mode.")
            print("        Re-run with --mode both to train and evaluate.")

    # ── FINAL SUMMARY ────────────────────────────────────────────────────────
    print("\\n" + "=" * 72)
    print("  ✅  PIPELINE COMPLETE")
    print("=" * 72)

    if mlp_metrics:
        print("\\n  MLP Classifier Results:")
        print(f"  {\'COIN\':<10} {\'ACCURACY\':>9} {\'DOWN_P\':>8} {\'UP_P\':>8} {\'N_ITER\':>7}")
        print(f"  {\'-\'*50}")
        for c, m in mlp_metrics.items():
            print(f"  {c:<10} {m[\'accuracy\']:>9.4f} {m[\'precision\'][0]:>8.4f} "
                  f"{m[\'precision\'][2]:>8.4f} {m[\'n_iter\']:>7}")

    if bt_metrics:
        print("\\n  PPO Backtest Results:")
        print(f"  Coin          : {bt_metrics[\'coin\']}")
        print(f"  Total Return  : {bt_metrics[\'total_return\']:>+.6f}")
        print(f"  B&H  Return   : {bt_metrics[\'bnh_return\']:>+.6f}")
        print(f"  Sharpe Ratio  : {bt_metrics[\'sharpe\']:>+.4f}")
        print(f"  Win Rate      : {bt_metrics[\'win_rate\']*100:.2f}%")
        ac = bt_metrics["action_counts"]
        print(f"  SELL/HOLD/BUY : {ac[\'SELL\']}/{ac[\'HOLD\']}/{ac[\'BUY\']}")

    print(f"\\n  Output files in \'{args.output_dir}\':")
    for fn in ["signals_log.csv", "ppo_cumulative_return.png",
               "ppo_training_curve.png", "ppo_action_distribution.png"]:
        fp = os.path.join(args.output_dir, fn)
        if os.path.exists(fp):
            sz = os.path.getsize(fp)
            print(f"    ✅  {fn:<35} {sz:>8,} bytes")
        else:
            print(f"    ○   {fn:<35} (not generated in this mode)")

    print("=" * 72)


if __name__ == "__main__":
    main()
'''

# Write to canvas filesystem
with open("main.py", "w") as f:
    f.write(MAIN_PY)

import os
_size = os.path.getsize("main.py")
print(f"✅  main.py written successfully")
print(f"    Size: {_size:,} bytes  ({_size // 1024} KB)")
print(f"    Path: main.py (canvas filesystem current working directory)")
print()
print("Run with:")
print("  python main.py --mode both")
print("  python main.py --mode train --coins BTC ETH SOL --episodes 50000")
print("  python main.py --mode run")
print("  python main.py --mode both --output-dir /tmp/out")
