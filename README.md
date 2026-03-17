# Binance Signal Pipeline

This repository contains a crypto futures research project built around Binance Futures market data. It has two main parts:

- a Zerve canvas export in `1fe0e4e4-ce0b-41c7-9a6b-c6fef6e6e466/`
- a standalone Python CLI pipeline in `1fe0e4e4-ce0b-41c7-9a6b-c6fef6e6e466/Development/main.py`

The current codebase does not execute trades. It fetches market data, builds features, trains models, writes signal logs, saves models, and generates PPO backtest charts.

## What the project currently does

- Downloads 1-minute Binance Futures candles from the public REST API for `BTC`, `ETH`, `SOL`, `BNB`, `XRP`, `DOGE`, `NEIRO`, and `ZEREBRO`
- Trains a per-coin direction classifier using `scikit-learn` `MLPClassifier`
- Builds a pure NumPy PPO agent for one coin at a time
- Writes `signals_log.csv` plus PPO evaluation charts
- Supports optional Telegram alert dispatch inside the Zerve workflow

## Important scope notes

- Despite some legacy names such as `train_gru_classifiers`, the implementation in this repo is MLP-based, not GRU-based.
- The standalone script trains PPO only for the first symbol in the `--coins` list. With the default coin order, that is `BTC`.
- Binance market data is fetched from public endpoints, so API keys are not required for data ingestion.
- The standalone script does not place orders and does not send Telegram alerts.

## Repository layout

```text
binance-signal-pipeline/
|-- README.md
|-- LICENSE
`-- 1fe0e4e4-ce0b-41c7-9a6b-c6fef6e6e466/
    |-- canvas.yaml
    `-- Development/
        |-- README.md
        |-- main.py
        |-- layer.yaml
        |-- data/
        |-- features/
        |-- labels/
        |-- train/
        |-- rl/
        |-- live/
        `-- alerts/
```

## Quick start

The easiest local entry point is the standalone script:

```powershell
cd 1fe0e4e4-ce0b-41c7-9a6b-c6fef6e6e466\Development
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install numpy pandas scikit-learn matplotlib
python main.py --mode both
```

This will:

- fetch fresh Binance Futures data
- train MLP models for the selected coins
- train a PPO agent for the first selected coin
- save model files under `.\models`
- write outputs such as `signals_log.csv` and PPO charts

For detailed usage, model behavior, output files, and the Zerve block layout, see [Development/README.md](./1fe0e4e4-ce0b-41c7-9a6b-c6fef6e6e466/Development/README.md).

## Generated artifacts

Some folders under `Development/` such as `models/`, `out/`, `out_both/`, `out_small/`, and similar directories are generated outputs from prior runs, not source code.

## License

MIT. See [LICENSE](./LICENSE).
