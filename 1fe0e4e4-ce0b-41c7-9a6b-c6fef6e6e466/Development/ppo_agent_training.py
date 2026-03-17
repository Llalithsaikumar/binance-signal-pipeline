
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  Zerve Design System
# ══════════════════════════════════════════════════════════════════════════════
BG_COLOR    = "#1D1D20"
TEXT_COLOR  = "#fbfbff"
MUTED_COLOR = "#909094"
COLORS = ["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF",
          "#1F77B4", "#9467BD", "#8C564B"]
GOLD  = "#ffd400"
GREEN = "#17b26a"
RED   = "#f04438"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR, "axes.facecolor": BG_COLOR,
    "text.color": TEXT_COLOR,     "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,    "ytick.color": TEXT_COLOR,
    "axes.edgecolor": MUTED_COLOR,"grid.color": "#33333a",
    "axes.titlesize": 13, "axes.labelsize": 11, "font.family": "sans-serif",
})

# ══════════════════════════════════════════════════════════════════════════════
#  Re-declare module-level constants from trading_gym_env that are private (_)
#  and therefore not propagated across blocks
# ══════════════════════════════════════════════════════════════════════════════
# This mirrors _POSITION_MAP from the TradingEnv block — must be in scope when
# TradingEnv.step() executes inside this block's runtime.
_POSITION_MAP = {0: -1.0, 1: 0.0, 2: 1.0}   # SELL=-1, HOLD=0, BUY=+1
_REQUIRED_COLS = TRADING_FEATURE_COLS + ["ppo_action"]

# ══════════════════════════════════════════════════════════════════════════════
#  Pure NumPy PPO Implementation
#  Actor-Critic with MLP policy — no external RL libraries needed
#  Hyperparameters mirror stable-baselines3 MlpPolicy defaults
# ══════════════════════════════════════════════════════════════════════════════

def _relu(x):   return np.maximum(0.0, x)
def _relu_d(x): return (x > 0).astype(np.float64)

def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)


class MLP:
    """Lightweight fully-connected network (NumPy only, He initialisation)."""

    def __init__(self, layer_sizes: list, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases  = []
        self._cache  = []
        for i in range(len(layer_sizes) - 1):
            fan_in  = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            self.weights.append(rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in))
            self.biases.append(np.zeros(fan_out))

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
    PPO with clipped surrogate objective (Schulman et al., 2017).
    Two separate MLPs for actor and critic — no shared trunk.
    Adam optimiser with manual backprop.
    """

    def __init__(self, obs_dim: int, n_actions: int, seed: int = 42,
                 lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_range: float = 0.2, ent_coef: float = 0.01,
                 n_epochs: int = 10, batch_size: int = 64, n_steps: int = 2048):
        hidden  = [64, 64]
        self.actor  = MLP([obs_dim] + hidden + [n_actions], seed=seed)
        self.critic = MLP([obs_dim] + hidden + [1],         seed=seed + 1)

        self.lr         = lr
        self.gamma      = gamma
        self.lam        = gae_lambda
        self.clip       = clip_range
        self.ent_coef   = ent_coef
        self.n_epochs   = n_epochs
        self.batch_size = batch_size
        self.n_steps    = n_steps
        self.n_actions  = n_actions
        self.rng        = np.random.default_rng(seed)

        def _adam_buf(net):
            sz = sum(W.size + b.size for W, b in zip(net.weights, net.biases))
            return np.zeros(sz), np.zeros(sz)

        self._m_a, self._v_a = _adam_buf(self.actor)
        self._m_c, self._v_c = _adam_buf(self.critic)
        self._t = 0

    # ── Inference ─────────────────────────────────────────────────────────────

    def get_action_and_value(self, obs: np.ndarray):
        logits = self.actor.forward(obs[np.newaxis, :])[0]
        probs  = _softmax(logits)
        action = int(self.rng.choice(self.n_actions, p=probs))
        log_p  = float(np.log(probs[action] + 1e-12))
        value  = float(self.critic.forward(obs[np.newaxis, :])[0, 0])
        return action, log_p, value

    def predict(self, obs: np.ndarray) -> int:
        """Deterministic greedy action."""
        return int(np.argmax(self.actor.forward(obs[np.newaxis, :])[0]))

    # ── GAE ───────────────────────────────────────────────────────────────────

    def compute_gae(self, rewards: np.ndarray, values: np.ndarray,
                    dones: np.ndarray, last_value: float):
        n   = len(rewards)
        adv = np.zeros(n, dtype=np.float64)
        gae = 0.0
        for t in reversed(range(n)):
            nxt = last_value if t == n - 1 else values[t + 1]
            mask  = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * nxt * mask - values[t]
            gae   = delta + self.gamma * self.lam * mask * gae
            adv[t] = gae
        return adv, adv + values

    # ── Manual backprop ───────────────────────────────────────────────────────

    def _backprop(self, net: MLP, dout: np.ndarray) -> np.ndarray:
        grads = []
        delta = dout
        for i in reversed(range(len(net.weights))):
            h, z = net._cache[i]
            if i < len(net.weights) - 1:
                delta = delta * _relu_d(z)
            grads.insert(0, (h.T @ delta, delta.sum(axis=0)))
            delta = delta @ net.weights[i].T
        return np.concatenate([np.concatenate([gW.ravel(), gb]) for gW, gb in grads])

    def _actor_grad(self, obs_b, actions_b, adv_b, old_lp_b):
        logits = self.actor.forward(obs_b)
        probs  = _softmax(logits)
        lp     = np.log(probs[np.arange(len(actions_b)), actions_b] + 1e-12)
        ratio  = np.exp(lp - old_lp_b)

        # Clipped PPO objective gradient mask
        clipped = ((adv_b > 0) & (ratio > 1 + self.clip)) | \
                  ((adv_b < 0) & (ratio < 1 - self.clip))
        pg_coef = -np.where(clipped, 0.0, adv_b)

        oh = np.zeros_like(probs)
        oh[np.arange(len(actions_b)), actions_b] = 1.0
        dlogits  = (oh - probs) * pg_coef[:, np.newaxis]         # policy gradient
        ent_grad = probs * (np.log(probs + 1e-12) + 1.0) - probs # ∂H/∂logits
        dlogits  = (dlogits - self.ent_coef * ent_grad) / len(obs_b)

        return self._backprop(self.actor, dlogits)

    def _critic_grad(self, obs_b, returns_b):
        v  = self.critic.forward(obs_b)[:, 0]
        dv = 2.0 * (v - returns_b) / len(obs_b)
        return self._backprop(self.critic, dv[:, np.newaxis])

    def _adam_step(self, net, grad, m, v, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        m[:]   = beta1 * m + (1 - beta1) * grad
        v[:]   = beta2 * v + (1 - beta2) * grad ** 2
        mh     = m / (1 - beta1 ** self._t)
        vh     = v / (1 - beta2 ** self._t)
        update = self.lr * mh / (np.sqrt(vh) + eps)
        idx = 0
        for i in range(len(net.weights)):
            sz = net.weights[i].size
            net.weights[i] -= update[idx:idx + sz].reshape(net.weights[i].shape); idx += sz
            sz = net.biases[i].size
            net.biases[i]  -= update[idx:idx + sz]; idx += sz

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, env, total_timesteps: int):
        reward_log  = []
        obs, _      = env.reset(seed=42)
        obs         = obs.astype(np.float64)
        ep_rewards  = []
        ep_cur      = 0.0
        total_steps = 0

        print(f"    {'Timestep':>10}  {'Mean Ep Reward':>15}  {'Eps':>6}")
        print(f"    {'─' * 38}")

        while total_steps < total_timesteps:
            n = self.n_steps
            obs_buf  = np.zeros((n, obs.shape[0]))
            act_buf  = np.zeros(n, dtype=np.int32)
            rew_buf  = np.zeros(n)
            val_buf  = np.zeros(n)
            lp_buf   = np.zeros(n)
            done_buf = np.zeros(n, dtype=bool)

            for t in range(n):
                action, log_p, value = self.get_action_and_value(obs)
                obs_buf[t] = obs
                act_buf[t] = action
                lp_buf[t]  = log_p
                val_buf[t] = value

                obs_next, reward, term, trunc, _ = env.step(action)
                rew_buf[t]  = reward
                done_buf[t] = term or trunc
                ep_cur     += reward
                total_steps += 1

                if done_buf[t]:
                    ep_rewards.append(ep_cur)
                    ep_cur = 0.0
                    obs, _ = env.reset()
                    obs    = obs.astype(np.float64)
                else:
                    obs = obs_next.astype(np.float64)

            _, _, last_v = self.get_action_and_value(obs)
            adv, returns = self.compute_gae(rew_buf, val_buf, done_buf, last_v)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            for _ in range(self.n_epochs):
                perm = self.rng.permutation(n)
                for start in range(0, n, self.batch_size):
                    b = perm[start:start + self.batch_size]
                    a_grad = self._actor_grad(obs_buf[b], act_buf[b], adv[b], lp_buf[b])
                    self._adam_step(self.actor,  a_grad, self._m_a, self._v_a)
                    c_grad = self._critic_grad(obs_buf[b], returns[b])
                    self._adam_step(self.critic, c_grad, self._m_c, self._v_c)

            if ep_rewards:
                mean_rew = float(np.mean(ep_rewards[-20:]))
                reward_log.append((total_steps, mean_rew))
                if len(reward_log) % 5 == 0 or total_steps >= total_timesteps:
                    print(f"    {total_steps:>10,}  {mean_rew:>+15.6f}  {len(ep_rewards):>6,}")

        env.close()
        return reward_log


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN / TEST SPLIT  (80/20 by time — BTC)
# ══════════════════════════════════════════════════════════════════════════════
COIN             = "BTC"
TRANSACTION_COST = 0.001
TOTAL_TIMESTEPS  = 100_000

_btc_mat = (ppo_feature_matrices[COIN][_REQUIRED_COLS]
            .dropna()
            .reset_index(drop=True))

_n        = len(_btc_mat)
_split    = int(_n * 0.80)
_train_df = _btc_mat.iloc[:_split].reset_index(drop=True)
_test_df  = _btc_mat.iloc[_split:].reset_index(drop=True)

print(f"{'═' * 70}")
print(f"  PPO AGENT (Pure NumPy MlpPolicy) — BTC Trading Environment")
print(f"{'═' * 70}")
print(f"  Total rows   : {_n:>10,}")
print(f"  Train rows   : {len(_train_df):>10,}  (80%)")
print(f"  Test rows    : {len(_test_df):>10,}  (20%)")
print(f"  Features     : {len(TRADING_FEATURE_COLS)}-dim  {TRADING_FEATURE_COLS}")
print(f"  Actions      : 0=SELL | 1=HOLD | 2=BUY")
print(f"  Tx cost      : {TRANSACTION_COST*100:.1f}%")
print(f"  Timesteps    : {TOTAL_TIMESTEPS:,}")
print(f"  Architecture : MLP [13 → 64 → 64 → 3] actor + [13 → 64 → 64 → 1] critic")
print(f"{'═' * 70}\n")

# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════
_train_env = TradingEnv(df=_train_df, transaction_cost=TRANSACTION_COST, coin=COIN)

ppo_agent = PPOAgent(
    obs_dim=len(TRADING_FEATURE_COLS), n_actions=3,
    seed=42, lr=3e-4, gamma=0.99, gae_lambda=0.95,
    clip_range=0.2, ent_coef=0.01, n_epochs=10,
    batch_size=64, n_steps=2048,
)

print("  Training PPO agent …\n")
_reward_log = ppo_agent.train(_train_env, total_timesteps=TOTAL_TIMESTEPS)

ppo_reward_curve_steps   = [x[0] for x in _reward_log]
ppo_reward_curve_rewards = [x[1] for x in _reward_log]

print(f"\n  ✅  Training complete — {TOTAL_TIMESTEPS:,} timesteps")
print(f"  Reward curve checkpoints : {len(ppo_reward_curve_steps)}\n")

# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION on TEST SET
# ══════════════════════════════════════════════════════════════════════════════
print("  Evaluating on test set …")

_test_env = TradingEnv(df=_test_df, transaction_cost=TRANSACTION_COST, coin=COIN)
_obs, _   = _test_env.reset(seed=0)
_obs      = _obs.astype(np.float64)
_done     = False

ppo_test_step_rewards   = []
ppo_test_step_positions = []
ppo_test_actions_taken  = []
_POS_MAP = {0: -1.0, 1: 0.0, 2: 1.0}
_total_test_reward = 0.0

while not _done:
    _act = ppo_agent.predict(_obs)
    _obs_next, _rew, _term, _trunc, _ = _test_env.step(_act)
    ppo_test_step_rewards.append(_rew)
    ppo_test_actions_taken.append(_act)
    ppo_test_step_positions.append(_POS_MAP[_act])
    _total_test_reward += _rew
    _done = _term or _trunc
    _obs  = _obs_next.astype(np.float64)

_test_env.close()

ppo_test_step_rewards   = np.array(ppo_test_step_rewards,   dtype=np.float64)
ppo_test_step_positions = np.array(ppo_test_step_positions, dtype=np.float64)
ppo_test_actions_taken  = np.array(ppo_test_actions_taken,  dtype=np.int32)

ppo_cumulative_return = np.cumsum(ppo_test_step_rewards)
_test_log_rets        = _test_df["log_return"].values
bnh_cumulative_return = np.cumsum(_test_log_rets)

_mean_r        = np.mean(ppo_test_step_rewards)
_std_r         = np.std(ppo_test_step_rewards)
_bars_per_year = 365 * 24 * 60
ppo_sharpe     = (_mean_r / (_std_r + 1e-12)) * np.sqrt(_bars_per_year)
ppo_total_return = float(ppo_cumulative_return[-1])
ppo_win_rate     = float(np.mean(ppo_test_step_rewards > 0))

print(f"\n{'═' * 70}")
print(f"  PPO AGENT — TEST SET EVALUATION")
print(f"{'═' * 70}")
print(f"  Test steps       : {len(ppo_test_step_rewards):>10,}")
print(f"  Total Reward     : {_total_test_reward:>+14.6f}")
print(f"  Sharpe Ratio     : {ppo_sharpe:>+14.4f}")
print(f"  Total Return     : {ppo_total_return:>+14.6f}")
print(f"  Win Rate         : {ppo_win_rate * 100:>13.2f}%")
print(f"  B&H Return       : {bnh_cumulative_return[-1]:>+14.6f}")
print(f"{'═' * 70}\n")

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 1 — Cumulative Portfolio Return vs Buy-and-Hold
# ══════════════════════════════════════════════════════════════════════════════
_n_test = len(ppo_cumulative_return)

fig1, ax1 = plt.subplots(figsize=(12, 5))
fig1.patch.set_facecolor(BG_COLOR)
ax1.set_facecolor(BG_COLOR)

ax1.plot(ppo_cumulative_return, color=COLORS[0], linewidth=1.8, label="PPO Agent", alpha=0.95)
ax1.plot(bnh_cumulative_return, color=COLORS[1], linewidth=1.5, label="Buy & Hold",
         linestyle="--", alpha=0.85)
ax1.axhline(0, color=MUTED_COLOR, linewidth=0.6, linestyle=":")
ax1.set_title("Cumulative Return — PPO Agent vs Buy & Hold  (BTC · 20% Test Set)",
              color=TEXT_COLOR, pad=14)
ax1.set_xlabel("Test Step (1-min bars)", color=TEXT_COLOR)
ax1.set_ylabel("Cumulative Log Return", color=TEXT_COLOR)
ax1.legend(facecolor=BG_COLOR, edgecolor=MUTED_COLOR, labelcolor=TEXT_COLOR, fontsize=10)
ax1.tick_params(colors=TEXT_COLOR)
ax1.grid(alpha=0.25)

_off = 18 if ppo_cumulative_return[-1] >= bnh_cumulative_return[-1] else -18
ax1.annotate(f"PPO: {ppo_cumulative_return[-1]:+.4f}",
             xy=(_n_test - 1, ppo_cumulative_return[-1]),
             xytext=(-100, _off), textcoords="offset points",
             color=COLORS[0], fontsize=9, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=COLORS[0], lw=1.2))
ax1.annotate(f"B&H: {bnh_cumulative_return[-1]:+.4f}",
             xy=(_n_test - 1, bnh_cumulative_return[-1]),
             xytext=(-100, -_off), textcoords="offset points",
             color=COLORS[1], fontsize=9, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=COLORS[1], lw=1.2))

plt.tight_layout()
plt.savefig("ppo_cumulative_return.png", dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
plt.show()
print("  📈 Plot 1: Cumulative return vs Buy-and-Hold — rendered")

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 2 — Training Reward Curve
# ══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(12, 4))
fig2.patch.set_facecolor(BG_COLOR)
ax2.set_facecolor(BG_COLOR)

if len(ppo_reward_curve_steps) > 1:
    ax2.plot(ppo_reward_curve_steps, ppo_reward_curve_rewards,
             color=COLORS[2], linewidth=1.8, alpha=0.9, label="Mean Ep. Reward (trailing 20)")
    _w  = max(3, len(ppo_reward_curve_rewards) // 5)
    _sm = pd.Series(ppo_reward_curve_rewards).rolling(_w, min_periods=1).mean().values
    ax2.plot(ppo_reward_curve_steps, _sm, color=GOLD, linewidth=2.4, linestyle="--",
             label=f"Trend ({_w}-pt MA)")
    ax2.axhline(0, color=MUTED_COLOR, linewidth=0.6, linestyle=":")
    ax2.legend(facecolor=BG_COLOR, edgecolor=MUTED_COLOR, labelcolor=TEXT_COLOR, fontsize=10)
else:
    ax2.text(0.5, 0.5, "Not enough rollout data", transform=ax2.transAxes,
             ha="center", va="center", color=MUTED_COLOR, fontsize=12)

ax2.set_title("PPO Training Reward Curve  (BTC · 100 K timesteps)", color=TEXT_COLOR, pad=14)
ax2.set_xlabel("Timestep", color=TEXT_COLOR)
ax2.set_ylabel("Mean Episode Reward", color=TEXT_COLOR)
ax2.tick_params(colors=TEXT_COLOR)
ax2.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("ppo_training_curve.png", dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
plt.show()
print("  📈 Plot 2: Training reward curve — rendered")

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT 3 — Action Distribution Pie Chart
# ══════════════════════════════════════════════════════════════════════════════
_ac = np.bincount(ppo_test_actions_taken, minlength=3)
_nt = _ac.sum()

fig3, ax3 = plt.subplots(figsize=(7, 7))
fig3.patch.set_facecolor(BG_COLOR)
ax3.set_facecolor(BG_COLOR)

ax3.pie(
    _ac,
    labels=[f"SELL\n{_ac[0]:,}  ({_ac[0]/_nt*100:.1f}%)",
            f"HOLD\n{_ac[1]:,}  ({_ac[1]/_nt*100:.1f}%)",
            f"BUY\n{_ac[2]:,}  ({_ac[2]/_nt*100:.1f}%)"],
    colors=[RED, MUTED_COLOR, GREEN],
    explode=[0.04, 0.04, 0.04], startangle=90,
    textprops={"color": TEXT_COLOR, "fontsize": 12, "fontweight": "bold"},
    wedgeprops={"edgecolor": BG_COLOR, "linewidth": 2},
)
ax3.set_title("PPO Agent — Action Distribution on Test Set",
              color=TEXT_COLOR, pad=18, fontsize=13)
plt.tight_layout()
plt.savefig("ppo_action_distribution.png", dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
plt.show()
print("  📈 Plot 3: Action distribution pie — rendered\n")

# ══════════════════════════════════════════════════════════════════════════════
#  FINAL KEY METRICS
# ══════════════════════════════════════════════════════════════════════════════
print(f"{'═' * 70}")
print(f"  ✅  PPO AGENT — FINAL KEY METRICS")
print(f"{'═' * 70}")
print(f"  Sharpe Ratio    : {ppo_sharpe:>+.4f}  (annualised, 1-min bars)")
print(f"  Total Return    : {ppo_total_return:>+.6f}  (cumulative log-return)")
print(f"  Win Rate        : {ppo_win_rate * 100:>.2f}%")
print(f"  B&H Return      : {bnh_cumulative_return[-1]:>+.6f}")
print(f"  SELL / HOLD / BUY : {_ac[0]:,} / {_ac[1]:,} / {_ac[2]:,}")
print(f"{'═' * 70}")
