# 🤖 PPO Crypto Trading Agent

A full end-to-end reinforcement learning pipeline that trains a **PPO (Proximal Policy Optimisation) agent** to trade BTC perpetual futures on Binance — alongside a multi-coin **MLP signal engine** that generates live UP/DOWN/SIDEWAYS signals with Telegram alerts.

---

## 📋 Project Overview

This canvas implements a complete crypto trading research workflow:

1. **Data ingestion** — 6 months of 1-minute OHLCV candles for 8 coins from Binance Futures API
2. **Feature engineering** — 14 technical indicators (RSI, EMA, MACD, Bollinger Bands, ROC, etc.) computed in parallel via Zerve Fleet
3. **Label generation** — 5-bar forward return labels (UP / SIDEWAYS / DOWN) with per-coin thresholds
4. **Classifier training** — MLP classifiers (sklearn) trained per coin, 80/20 chronological split
5. **Live signal engine** — Real-time inference with per-coin confidence gates and ATR volatility warnings
6. **PPO feature matrix** — Return-based features (log returns, rolling vol, momentum) built for RL training
7. **Custom trading gym** — Gym-compatible `TradingEnv` (no gym dependency) with position-weighted rewards
8. **PPO agent training** — Pure NumPy MLP actor-critic PPO, 100K timesteps on BTC
9. **Telegram alerts** — Qualifying signals dispatched via Telegram Bot API

---

## 🗺️ Canvas Structure & Block Descriptions

The canvas has two parallel branches diverging after `label_targets`:

### 🔵 Branch 1 — MLP Signal Pipeline (top row)

| Block | Type | Description |
|---|---|---|
| `load_futures_candles` | Python | Fetches 1-min OHLCV from Binance FAPI for 8 perpetual futures. Defines `COIN_CONFIG` with per-coin thresholds and training hyperparameters. |
| `fan_out_coins` | Python | Uses `spread()` to fan out execution across 8 coins in parallel (Zerve Fleet). |
| `compute_features` | Python | Computes 14 technical indicators per coin: RSI(14), EMA 9/21, MACD(12,26,9), Bollinger Bands(20,2), volume delta, body size, ROC(5). |
| `collect_featured_dfs` | Aggregator | Collects all 8 parallel feature DataFrames, validates columns and NaN counts. |
| `label_targets` | Python | Applies 5-bar forward return labelling with per-coin thresholds: ±0.3% for standard coins, ±1.5% for NEIRO/ZEREBRO. |
| `train_gru_classifiers` | Python | Trains per-coin MLP classifiers (sklearn, 256→128 hidden) with balanced class weights. Computes RL reward simulation on test set. ⚠️ *Currently failing — see Notes.* |
| `live_signal_engine` | Python | Fetches live candles for all 8 coins in parallel, runs trained models, applies per-coin confidence gates (65%/75%) and ATR(14) volatility warnings. Logs signals to `signals_log.csv`. |
| `send_telegram_alerts` | Python | Dispatches UP/DOWN signals to Telegram via Bot API using `MarkdownV2` formatting. Reads credentials from environment variables. |

### 🟡 Branch 2 — PPO RL Pipeline (bottom row)

| Block | Type | Description |
|---|---|---|
| `build_ppo_feature_matrix` | Python | Loads `signals_log.csv`, engineers return-based PPO features (log returns, vol\_20, vol\_5, mom\_5, mom\_20), builds action labels (SELL/HOLD/BUY) for all 8 coins from `labeled_data`. |
| `trading_gym_env` | Python | Implements a Gym-compatible `TradingEnv` class (pure NumPy/Pandas, no gym dependency). 13-feature obs space, Discrete(3) action space, position-weighted reward minus transaction cost. |
| `ppo_agent_training` | Python | Trains a pure NumPy PPO agent (MLP 13→64→64→3 actor + critic) with GAE, Adam optimiser, clipped surrogate objective. Evaluates on 20% held-out test set and produces 3 output charts. |

---

## 🚀 How to Run

### Prerequisites
- No external ML libraries required for the PPO pipeline (pure NumPy)
- `sklearn`, `pandas`, `numpy`, `matplotlib` for the MLP pipeline
- Internet access for Binance FAPI (public endpoint — no API key needed for fetching candles)
- Optional: Telegram Bot token + chat ID for alerts

### Step-by-step Execution Order

```
1. load_futures_candles    ← start here (fetches ~6 months of data, takes a few minutes)
2. fan_out_coins           ← auto-runs after load
3. compute_features        ← runs in parallel for all 8 coins
4. collect_featured_dfs    ← aggregates Fleet results
5. label_targets           ← generates UP/SIDEWAYS/DOWN labels
         │
         ├── [Branch 1: MLP Signals]
         │   6a. train_gru_classifiers   ← trains 8 MLP models
         │   7a. live_signal_engine      ← live inference + CSV log
         │   8a. send_telegram_alerts    ← optional Telegram dispatch
         │
         └── [Branch 2: PPO Agent]
             6b. build_ppo_feature_matrix  ← PPO feature engineering
             7b. trading_gym_env           ← registers TradingEnv-v0
             8b. ppo_agent_training        ← trains & evaluates PPO agent
```

> 💡 **Tip:** Run blocks left-to-right. `label_targets` feeds both branches — run it before either `train_gru_classifiers` or `build_ppo_feature_matrix`.

### Telegram Alerts Setup
Set the following environment variables before running `send_telegram_alerts`:
```
TELEGRAM_BOT_TOKEN = <your bot token from @BotFather>
TELEGRAM_CHAT_ID   = <your chat or group ID>
```
If these are not set, the block runs without sending (safe to ignore).

---

## 💻 Standalone `main.py`

`main.py` is a fully self-contained version of this entire pipeline that runs from the command line — no Zerve canvas required. It replicates every stage: data loading, feature engineering, MLP training, PPO training, backtesting, and chart generation.

The file is already written to the canvas filesystem by the `write_main_py` block.

### Prerequisites

Install all dependencies with a single pip command:

```bash
pip install numpy pandas scikit-learn matplotlib requests
```

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥1.24 | PPO agent, feature math |
| `pandas` | ≥2.0 | DataFrames, feature engineering |
| `scikit-learn` | ≥1.3 | `MLPClassifier` for signal engine |
| `matplotlib` | ≥3.7 | Chart generation (PNG output) |
| `requests` | ≥2.31 | *(optional)* Telegram alerts |

> Python **3.9+** required. No `gymnasium`, `torch`, or `tensorflow` dependencies.

---

### Environment Variables

Set these before running if you want Telegram alerts to fire:

```bash
# Linux / macOS
export TELEGRAM_BOT_TOKEN="7123456789:AAFxxx..."
export TELEGRAM_CHAT_ID="-100123456789"

# Windows (PowerShell)
$env:TELEGRAM_BOT_TOKEN = "7123456789:AAFxxx..."
$env:TELEGRAM_CHAT_ID   = "-100123456789"
```

| Variable | Required | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Optional | Bot token from [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | Optional | Target chat / group / channel ID |

If neither is set, signal generation still runs and writes `signals_log.csv` — alerts are simply skipped.

---

### CLI Usage

#### ⚡ Quick-start (recommended)
```bash
python main.py
```
Runs the full pipeline — train MLP + PPO on all 8 coins, run inference, backtest, save charts. Equivalent to `--mode both`.

---

#### `--mode train` — Training only
```bash
# Train on all coins, 100 000 PPO timesteps (default)
python main.py --mode train

# Train on a subset of coins
python main.py --mode train --coins BTC ETH SOL

# Train with more PPO timesteps
python main.py --mode train --episodes 250000

# Train and save outputs to a custom directory
python main.py --mode train --coins BTC --episodes 50000 --output-dir ./results/btc_run1
```

---

#### `--mode run` — Inference & backtest only
```bash
# Run inference on all 8 coins (data fetched fresh, no models trained)
python main.py --mode run

# Run on a subset of coins
python main.py --mode run --coins BTC ETH

# Custom output directory
python main.py --mode run --output-dir ./signals
```

> ⚠️ In `run` mode, no PPO agent is available (it wasn't trained in this session), so PPO backtest and chart generation are skipped. Re-run with `--mode both` to train and evaluate.

---

#### `--mode both` — Full pipeline (train + run)
```bash
# Full pipeline, all coins
python main.py --mode both

# Full pipeline, specific coins, more timesteps, custom output
python main.py --mode both --coins BTC ETH SOL --episodes 200000 --output-dir ./out

# Meme coins only
python main.py --mode both --coins NEIRO ZEREBRO --episodes 100000
```

---

#### All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--mode` | `both` | Pipeline mode: `train`, `run`, or `both` |
| `--coins` | all 8 | Space-separated coin list: `BTC ETH SOL BNB XRP DOGE NEIRO ZEREBRO` |
| `--episodes` | `100000` | Total PPO training timesteps |
| `--output-dir` | `.` | Directory for output files (created if it doesn't exist) |

---

### Expected Output Files

After a full `--mode both` run, the following files are written to `--output-dir`:

| File | Mode | Description |
|---|---|---|
| `signals_log.csv` | `run`, `both` | MLP signal log. Columns: `timestamp`, `coin`, `signal`, `confidence`, `price`. One row per emitted signal. |
| `ppo_cumulative_return.png` | `both` | Cumulative log-return comparison: PPO agent vs Buy-and-Hold on the 20% BTC test set. |
| `ppo_training_curve.png` | `both` | PPO training reward curve — trailing 20-episode mean + moving-average trend line. |
| `ppo_action_distribution.png` | `both` | Pie chart of SELL / HOLD / BUY action distribution on the test set. |

> Files **not generated** in the current mode are reported as `○ (not generated)` in the final summary printed to stdout.

---

### Example Console Output

```
════════════════════════════════════════════════════════════════════════
  CRYPTO FUTURES TRADING PIPELINE  (MLP + PPO)
════════════════════════════════════════════════════════════════════════
  Mode       : both
  Coins      : ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'NEIRO', 'ZEREBRO']
  Episodes   : 100,000
  Output dir : .
════════════════════════════════════════════════════════════════════════

[1/7] Loading Binance Futures candles (parallel)…
  BTC          259,200 rows  2024-09-17 → 2025-03-17
  ETH          259,200 rows  2024-09-17 → 2025-03-17
  ...

[2/7] Feature engineering…
  BTC      MLP rows= 259,195  PPO rows= 259,174

[3/7] Training MLP classifiers…
  BTC      acc=0.5312  n_iter=30  DOWN_P=0.538  UP_P=0.541

[4/7] Training PPO agent on BTC (100,000 timesteps)…
     Timestep    MeanEpReward    Episodes
     ────────────────────────────────────────
        2,048      +0.000012            1
       ...

[5/7] MLP inference & signal generation…
  BTC      signal=UP         conf=72.3%  gate=65%  emit=YES

[6/7] Backtesting PPO agent on BTC test set…
  Total Return  : +0.001234
  Sharpe Ratio  : +1.4200
  Win Rate      : 51.23%

[7/7] Saving charts to .…
  Saved: ./ppo_cumulative_return.png
  Saved: ./ppo_training_curve.png
  Saved: ./ppo_action_distribution.png

════════════════════════════════════════════════════════════════════════
  ✅  PIPELINE COMPLETE
════════════════════════════════════════════════════════════════════════
```

---

## 📁 File Descriptions

### Current Folder Layout

```text
Development/
├── main.py
├── alerts/
│   └── send_telegram_alerts.py
├── data/
│   ├── fan_out_coins.py
│   └── load_futures_candles.py
├── features/
│   ├── build_ppo_feature_matrix.py
│   ├── collect_featured_dfs.text
│   └── compute_features.py
├── labels/
│   └── label_targets.py
├── live/
│   └── live_signal_engine.py
├── rl/
│   └── trading_gym_env.py
└── train/
  ├── train_mlp_classifiers.py
  └── train_ppo_agent.py
```

| File | Description |
|---|---|
| `main.py` | Standalone CLI script — complete pipeline (MLP + PPO) without Zerve. |
| `signals_log.csv` | Live signal log written by `live_signal_engine`. Columns: `timestamp`, `coin`, `signal`, `confidence`, `price`, `volatility_warning`, `atr_14`. Reset on each run. Also read by `build_ppo_feature_matrix` to seed PPO features. |
| `ppo_cumulative_return.png` | Chart comparing PPO agent cumulative log-return vs Buy-and-Hold on the BTC 20% test set. |
| `ppo_training_curve.png` | PPO training reward curve over 100K timesteps — mean episode reward (trailing 20 eps) with a moving-average trend line. |
| `ppo_action_distribution.png` | Pie chart of PPO agent action distribution (SELL / HOLD / BUY) on the test set. |

---

## ⚙️ Configuration

### Coin Config (`load_futures_candles`)
All per-coin parameters live in `COIN_CONFIG`:

| Param | Standard Coins | Meme Coins (NEIRO, ZEREBRO) |
|---|---|---|
| Label threshold | ±0.3% | ±1.5% |
| MLP epochs | 30 | 20 |
| Dropout (reference) | 0.20 | 0.35 |
| L2 / alpha | 0.0 (→ 1e-4) | 0.001 |
| Trade fee | 0.02% | 0.06% |
| Confidence gate | 65% | 75% |

### PPO Hyperparameters (`ppo_agent_training`)
```python
lr           = 3e-4
gamma        = 0.99
gae_lambda   = 0.95
clip_range   = 0.2
ent_coef     = 0.01
n_epochs     = 10       # PPO update epochs per rollout
batch_size   = 64
n_steps      = 2048     # rollout buffer size
total_timesteps = 100_000
transaction_cost = 0.1% # applied on each position change
```

### Feature Columns
**MLP model features (14):** `rsi_14`, `ema_9`, `ema_21`, `macd_line`, `macd_signal`, `macd_histogram`, `bb_upper`, `bb_lower`, `bb_width`, `bb_pct_b`, `vol_sma_20`, `vol_delta`, `body_size_norm`, `roc_5`

**PPO/TradingEnv features (13):** `rsi_14`, `macd_line`, `macd_histogram`, `bb_width`, `bb_pct_b`, `vol_delta`, `body_size_norm`, `roc_5`, `log_return`, `vol_20`, `vol_5`, `mom_5`, `mom_20`

---

## 📝 Notes & Known Issues

### ⚠️ `train_mlp_classifiers` — Currently Failing
The block is currently in a **failed** state (`expected str, bytes or os.PathLike object, not int`). This is a path-handling bug in the RL reward simulation section of the block. The PPO branch (`build_ppo_feature_matrix` → `trading_gym_env` → `ppo_agent_training`) runs independently and is fully functional.

### 🔄 Re-running the Pipeline
- **Data is fetched fresh** each time `load_futures_candles` runs — candle counts will update
- **`signals_log.csv` is reset** on each `live_signal_engine` run (not appended)
- PPO training is **non-deterministic** across runs unless `seed=42` is fixed end-to-end

### 🔧 Extending the Project
- **Add more coins:** Extend `COIN_CONFIG` in `load_futures_candles` and update the spread list in `fan_out_coins`
- **Swap MLP for GRU/LSTM:** The `train_mlp_classifiers` block is designed to be upgraded — `COIN_CONFIG` stores `dropout` and `l2` for a future PyTorch GRU implementation
- **Multi-coin PPO:** Currently PPO trains only on BTC. Extend `train_ppo_agent` to loop over `PPO_COINS` using `ppo_feature_matrices`
- **Scheduled runs:** Wire `load_futures_candles` to a Zerve scheduled job to refresh signals periodically
- **Deploy as API:** Use a Zerve deployment script with `from zerve import variable` to serve live signals over HTTP

---

*Built on Zerve · Pure NumPy PPO · Binance FAPI · sklearn MLP · No gym dependency*
