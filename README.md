# 🚀 Binance Signal Pipeline

A personal crypto trading signal system that uses **GRU neural networks** and **PPO reinforcement learning** to predict short-term price direction on Binance Futures. Signals are delivered via **Telegram** and a **live web dashboard** — no auto-execution, manual trading only.

> ⚠️ **Always test on [Binance Testnet](https://testnet.binancefuture.com) before using real funds.**

---

## 📌 What It Does

Every 1 minute, the pipeline:
1. Pulls live 1-minute candle data from Binance Futures
2. Computes technical indicators (RSI, EMA, MACD, Bollinger Bands, Volume delta)
3. Runs a trained GRU model → predicts UP / DOWN / SIDEWAYS for the next 5 minutes
4. Passes the signal to a PPO RL agent → confirms or skips the signal
5. If both agree with high confidence → fires a Telegram alert + updates dashboard

---

## 🪙 Coins Tracked

| Coin | Type | Confidence Threshold |
|------|------|----------------------|
| BTC/USDT | Standard | 65% |
| ETH/USDT | Standard | 65% |
| SOL/USDT | Standard | 65% |
| BNB/USDT | Standard | 65% |
| XRP/USDT | Standard | 65% |
| XLM/USDT | Standard | 65% |
| NEIRO/USDT | Meme / Small Cap | 75% |
| ZEREBRO/USDT | Meme / Small Cap | 75% |

Meme coins use a higher prediction threshold (1.5% move vs 0.3%) due to extreme volatility.

---

## 🏗️ Architecture

```
Binance Futures WebSocket (1m candles)
            │
            ▼
  Feature Engineering
  RSI · EMA · MACD · Bollinger Bands · Volume Delta · ATR
            │
            ▼
    GRU Classifier (PyTorch)
    Predicts: UP / DOWN / SIDEWAYS
    Output: direction + confidence score
            │
       confidence > threshold?
            │
            ▼
    PPO RL Agent (stable-baselines3)
    Decides: ACT or SKIP
    Learns from simulated trade outcomes
            │
         ACT?
        /     \
      YES       NO
       │         │
       ▼       silent
  Telegram Alert
  Live Dashboard update
  Signal logged to CSV
```

---

## 📲 Telegram Signal Format

```
🚨 SIGNAL — BTCUSDT
Direction  : 📈 LONG (UP)
GRU Conf   : 74%
RL Decision: ✅ CONFIRMED
Price      : $83,240
Timeframe  : Next 5 mins
⚠️ Manual trade — always use stop-loss
```

For meme coins:
```
🚨 SIGNAL — NEIROUSDT
Type       : 🎰 Meme / Small Cap
Direction  : 📉 SHORT (DOWN)
GRU Conf   : 78% (threshold: 75%)
RL Decision: ✅ CONFIRMED
Price      : $0.00142
Timeframe  : Next 5 mins
⚠️ HIGH VOLATILITY — smaller position size recommended
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| Data | `python-binance` WebSocket |
| Indicators | `ta` library |
| ML Model | `PyTorch` GRU |
| RL Agent | `stable-baselines3` PPO |
| Backtesting | `vectorbt` |
| API Server | `FastAPI` |
| Alerts | `python-telegram-bot` |
| Hosting | Railway / Zerve |

---

## 📁 Project Structure

```
binance-signal-pipeline/
├── block_1_data_ingestion.py       # Pull 6 months historical candles
├── block_2_feature_engineering.py  # Compute all technical indicators
├── block_3_label_generation.py     # Label UP / DOWN / SIDEWAYS
├── block_4_model_training.py       # Train GRU classifier per coin
├── block_5_live_signal_engine.py   # Real-time inference (runs every 1 min)
├── block_6_telegram_alert.py       # Send Telegram notifications
├── block_7_dashboard.py            # FastAPI live dashboard
├── block_8_gym_environment.py      # Custom OpenAI Gym trading env
├── block_9_ppo_training.py         # Train PPO RL agent per coin
├── block_10_backtest.py            # Backtest GRU vs GRU+RL vs hold
├── block_11_updated_live_engine.py # Combined GRU + PPO live pipeline
├── block_12_weekly_retrain.py      # Weekly model update (Sunday 2AM IST)
├── config.py                       # COIN_CONFIG dict — all settings
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Configuration

All per-coin settings live in one place in `config.py`:

```python
COIN_CONFIG = {
    "BTCUSDT":     {"threshold": 0.3, "epochs": 30, "dropout": 0.2, "confidence": 0.65, "fee": 0.04},
    "ETHUSDT":     {"threshold": 0.3, "epochs": 30, "dropout": 0.2, "confidence": 0.65, "fee": 0.04},
    "SOLUSDT":     {"threshold": 0.3, "epochs": 30, "dropout": 0.2, "confidence": 0.65, "fee": 0.04},
    "BNBUSDT":     {"threshold": 0.3, "epochs": 30, "dropout": 0.2, "confidence": 0.65, "fee": 0.04},
    "XRPUSDT":     {"threshold": 0.3, "epochs": 30, "dropout": 0.2, "confidence": 0.65, "fee": 0.04},
    "XLMUSDT":     {"threshold": 0.3, "epochs": 30, "dropout": 0.2, "confidence": 0.65, "fee": 0.04},
    "NEIROUSDT":   {"threshold": 1.5, "epochs": 20, "dropout": 0.35, "confidence": 0.75, "fee": 0.06},
    "ZEREBROUSDT": {"threshold": 1.5, "epochs": 20, "dropout": 0.35, "confidence": 0.75, "fee": 0.06},
}
```

To add a new coin in the future — just add one line to this dict.

---

## 🚀 Setup

### 1. Clone

```bash
git clone https://github.com/Llalithsaikumar/binance-signal-pipeline.git
cd binance-signal-pipeline
```

### 2. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set Environment Variables

Create a `.env` file (use `.env.example` as template):

```bash
# Binance
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret

# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

> Get your Telegram bot token from [@BotFather](https://t.me/botfather).
> Get your chat ID from [@userinfobot](https://t.me/userinfobot).

### 4. Run Training Pipeline

```bash
# Step 1: Pull historical data
python block_1_data_ingestion.py

# Step 2–4: Features + Labels + Train GRU
python block_4_model_training.py

# Step 8–9: Build Gym env + Train PPO
python block_9_ppo_training.py

# Step 10: Backtest all strategies
python block_10_backtest.py
```

### 5. Go Live

```bash
# Start the live signal engine
python block_11_updated_live_engine.py

# Start the dashboard (http://localhost:8000)
uvicorn block_7_dashboard:app --host 0.0.0.0 --port 8000
```

---

## 📊 Model Details

### GRU Classifier

| Parameter | Standard Coins | Meme Coins |
|-----------|---------------|------------|
| Layers | 2 | 2 |
| Hidden size | 128 | 128 |
| Dropout | 0.2 | 0.35 |
| Sequence length | 30 timesteps | 30 timesteps |
| Epochs | 30 | 20 |
| Output | UP / DOWN / SIDEWAYS | UP / DOWN / SIDEWAYS |

### PPO Agent

| Parameter | Value |
|-----------|-------|
| Policy | MlpPolicy |
| Learning rate | 0.0003 |
| Total timesteps | 500,000 |
| Batch size | 64 |
| Reward (correct) | +1.5 |
| Reward (wrong) | -2.0 |
| Fee penalty | per coin config |

### Weekly Retraining

Every **Sunday at 2:00 AM IST**, the pipeline automatically:
- Fine-tunes GRU on the past 7 days of new data
- Retrains PPO for 100k additional timesteps
- Backtests new vs old model
- Keeps old model if new one performs worse
- Sends a Telegram summary report

---

## 📈 Backtest Strategies Compared

| Strategy | Description |
|----------|-------------|
| A — GRU Only | Trade every GRU signal above confidence threshold |
| B — GRU + PPO | Trade only when PPO agent confirms the signal |
| C — Buy & Hold | Baseline comparison |

Results saved as PNG charts per coin after running `block_10_backtest.py`.

---

## 🐛 Common Issues

**WebSocket disconnects**
The live engine has auto-reconnect with try/except. If it keeps disconnecting, check your network or Binance API rate limits.

**Low model accuracy on NEIRO/ZEREBRO**
These coins have limited history (listed late 2024/early 2025). Accuracy will improve over time as more data is collected. Treat their signals with extra caution.

**Telegram not sending**
Verify your token: `curl https://api.telegram.org/botYOUR_TOKEN/getMe`
Make sure you've sent `/start` to your bot at least once.

**Out of memory during training**
Reduce `GRU_BATCH_SIZE` to 16 or 8 in `config.py`. Or train one coin at a time instead of parallel.

---

## ⚠️ Disclaimer

This project is for **personal and educational use only**.

- Not financial advice — signals are ML predictions, not guarantees
- Crypto trading involves substantial risk of loss
- Backtested accuracy does not predict live performance
- Always test on Binance Testnet before using real money
- Start with the smallest possible position size
- The author assumes no responsibility for trading losses

**Trade at your own risk.**

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built for personal use. Learning by doing.* 🚀
