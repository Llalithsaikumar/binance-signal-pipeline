# Development Workspace

This folder is the working area for the exported Zerve canvas plus a standalone local runner. If you want to run the project outside Zerve, start with `main.py`.

## What is in this folder

```text
Development/
|-- README.md
|-- main.py
|-- layer.yaml
|-- alerts/
|   `-- send_telegram_alerts.py
|-- data/
|   |-- fan_out_coins.py
|   `-- load_futures_candles.py
|-- features/
|   |-- build_ppo_feature_matrix.py
|   |-- collect_featured_dfs.text
|   `-- compute_features.py
|-- labels/
|   `-- label_targets.py
|-- live/
|   `-- live_signal_engine.py
|-- rl/
|   `-- trading_gym_env.py
`-- train/
    |-- train_mlp_classifiers.py
    `-- train_ppo_agent.py
```

## Pipeline summary

The current implementation has two related paths:

1. An MLP signal pipeline for direction classification per coin.
2. A PPO reinforcement learning pipeline for a single coin at a time.

At a high level the workflow is:

1. Load Binance Futures 1-minute candles.
2. Compute technical indicators.
3. Label direction targets.
4. Train per-coin MLP classifiers on 30-candle windows.
5. Build PPO feature matrices and a custom trading environment.
6. Train and backtest a pure NumPy PPO agent.
7. Write logs, charts, and optionally dispatch Telegram alerts in the Zerve flow.

## Asset universe and thresholds

The project is configured for eight USDT perpetual futures symbols.

| Coin | Symbol | Label threshold | Confidence gate |
| --- | --- | ---: | ---: |
| BTC | BTCUSDT | 0.3% | 65% |
| ETH | ETHUSDT | 0.3% | 65% |
| SOL | SOLUSDT | 0.3% | 65% |
| BNB | BNBUSDT | 0.3% | 65% |
| XRP | XRPUSDT | 0.3% | 65% |
| DOGE | DOGEUSDT | 0.3% | 65% |
| NEIRO | NEIROUSDT | 1.5% | 75% |
| ZEREBRO | ZEREBROUSDT | 1.5% | 75% |

## Features used

### MLP classifier features

`rsi_14`, `ema_9`, `ema_21`, `macd_line`, `macd_signal`, `macd_histogram`, `bb_upper`, `bb_lower`, `bb_width`, `bb_pct_b`, `vol_sma_20`, `vol_delta`, `body_size_norm`, `roc_5`

### PPO environment features

`rsi_14`, `macd_line`, `macd_histogram`, `bb_width`, `bb_pct_b`, `vol_delta`, `body_size_norm`, `roc_5`, `log_return`, `vol_20`, `vol_5`, `mom_5`, `mom_20`

## Standalone runner

`main.py` is the recommended local entry point. It reproduces the project flow without requiring Zerve.

### Dependencies

Install the Python packages used by `main.py`:

```powershell
pip install numpy pandas scikit-learn matplotlib
```

Notes:

- Binance data loading uses `urllib` from the standard library, so no extra HTTP client is required.
- `main.py` does not send Telegram messages.
- No exchange API key is needed for market data downloads because the script uses public Binance Futures endpoints.

### CLI usage

Show help:

```powershell
python main.py --help
```

Train and run the full pipeline:

```powershell
python main.py --mode both
```

Train only:

```powershell
python main.py --mode train
python main.py --mode train --coins BTC ETH SOL
python main.py --mode train --episodes 200000
```

Run inference and backtest using previously saved models:

```powershell
python main.py --mode run
python main.py --mode run --coins BTC ETH
python main.py --mode run --model-dir .\models
```

Custom output and model directories:

```powershell
python main.py --mode both --output-dir .\out_run --model-dir .\models_run
```

### CLI flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--mode` | `both` | `train`, `run`, or `both` |
| `--coins` | all 8 coins | Coins to process |
| `--episodes` | `100000` | PPO training timesteps |
| `--output-dir` | `.` | Directory for signal logs and charts |
| `--model-dir` | `./models` | Directory for saved and loaded model pickle files |

### How `run` mode works

- `run` mode still downloads fresh market data and rebuilds features.
- MLP models are loaded from `--model-dir` as `mlp_<COIN>.pkl`.
- The PPO agent is loaded from `--model-dir` as `ppo_<COIN>.pkl`.
- If no saved model exists for a coin, that coin is skipped.
- If no PPO agent is found, PPO backtest charts are skipped.

### Important PPO behavior

The standalone script trains PPO only for the first coin in the `--coins` list:

- `python main.py --mode both` -> PPO trains on `BTC`
- `python main.py --mode both --coins ETH SOL` -> PPO trains on `ETH`

The PPO backtest and chart generation also use that first coin.

## Output files

After a successful `--mode both` run, you should expect files like:

| File | Produced by | Description |
| --- | --- | --- |
| `signals_log.csv` | `run`, `both` | Emitted MLP signals with `timestamp`, `coin`, `signal`, `confidence`, `price` |
| `ppo_cumulative_return.png` | `both` | PPO vs buy-and-hold cumulative return chart |
| `ppo_training_curve.png` | `both` | PPO reward trend over training |
| `ppo_action_distribution.png` | `both` | PPO action mix on the test set |
| `models/mlp_<COIN>.pkl` | `train`, `both` | Saved scikit-learn MLP per coin |
| `models/ppo_<COIN>.pkl` | `train`, `both` | Saved PPO agent for the PPO coin |

## Zerve-specific files

The subfolders mirror the exported canvas blocks:

- `data/`, `features/`, `labels/`, `train/`, `rl/`, `live/`, and `alerts/` contain block source files.
- `layer.yaml` describes the Development layer.
- `../canvas.yaml` describes the full canvas export.

Two points are worth calling out:

- The block name `train_gru_classifiers` is legacy naming. The actual implementation in `train/train_mlp_classifiers.py` uses `scikit-learn` `MLPClassifier`.
- `alerts/send_telegram_alerts.py` is separate from `main.py`. Telegram is only part of the Zerve workflow.

### Telegram alerts in the Zerve flow

If you use the alert block, set:

```powershell
$env:TELEGRAM_BOT_TOKEN = "<bot token>"
$env:TELEGRAM_CHAT_ID = "<chat id>"
```

If either variable is missing, the alert block safely skips sending.

## Current limitations

- This is research code, not a production trading system.
- There is no live order execution, portfolio management, or risk engine.
- The standalone pipeline fetches data through REST calls, not WebSockets.
- Several files still use older names and comments from earlier iterations of the project.
