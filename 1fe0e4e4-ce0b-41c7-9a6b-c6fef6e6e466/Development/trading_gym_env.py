
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  TradingEnv — Custom Gym-Compatible Trading Environment (pure NumPy/Pandas)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Built without gymnasium/gym dependency — implements the full Gym API:
#    reset() → (obs, info)
#    step(action) → (obs, reward, terminated, truncated, info)
#
#  State   : 13-dim feature vector at timestep t (float32 np.ndarray)
#  Actions : 0=SELL | 1=HOLD | 2=BUY
#  Reward  : position-weighted log return  −  transaction_cost * |Δposition|
#  Episode : runs from bar 0 to the final bar in the supplied DataFrame
# ══════════════════════════════════════════════════════════════════════════════

# NOTE: 'log_return' is both a feature AND used for reward — it appears once
TRADING_FEATURE_COLS = [
    "rsi_14", "macd_line", "macd_histogram",
    "bb_width", "bb_pct_b",
    "vol_delta", "body_size_norm", "roc_5",
    "log_return", "vol_20", "vol_5", "mom_5", "mom_20",   # 13 features
]

# Columns to pull from the ppo_feature_matrices df
# log_return is already in TRADING_FEATURE_COLS so no duplication
_REQUIRED_COLS = TRADING_FEATURE_COLS + ["ppo_action"]

TRADING_ACTION_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}

# Position encoding: action → scalar position for reward computation
_POSITION_MAP = {0: -1.0, 1: 0.0, 2: 1.0}  # SELL=-1, HOLD=0, BUY=+1


# ── Minimal Space implementations (no gym dependency) ──────────────────────

class _DiscreteSpace:
    """Minimal Discrete space — mirrors gym.spaces.Discrete API."""
    def __init__(self, n: int):
        self.n = n
        self._rng = np.random.default_rng()

    def sample(self) -> int:
        return int(self._rng.integers(0, self.n))

    def contains(self, x) -> bool:
        return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)

    def __repr__(self):
        return f"Discrete({self.n})"


class _BoxSpace:
    """Minimal Box space — mirrors gym.spaces.Box API with infinite bounds support."""
    def __init__(self, low, high, shape: tuple, dtype=np.float32):
        self.low   = np.full(shape, low,  dtype=dtype)
        self.high  = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype
        self._rng  = np.random.default_rng()

    def sample(self) -> np.ndarray:
        _lo = np.where(np.isinf(self.low),  -1e6, self.low)
        _hi = np.where(np.isinf(self.high),  1e6, self.high)
        return self._rng.uniform(_lo, _hi).astype(self.dtype)

    def contains(self, x) -> bool:
        x = np.asarray(x, dtype=self.dtype)
        if x.shape != self.shape:
            return False
        if np.any(np.isnan(x)):
            return False
        _lo_ok = np.all(np.isneginf(self.low)  | (x >= self.low))
        _hi_ok = np.all(np.isposinf(self.high) | (x <= self.high))
        return bool(_lo_ok and _hi_ok)

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)

    def __repr__(self):
        return f"Box(-inf, inf, {self.shape}, {self.dtype})"


# ── TradingEnv ────────────────────────────────────────────────────────────────

class TradingEnv:
    """
    Crypto trading environment wrapping the PPO engineered feature matrix.
    Implements the Gymnasium-compatible API without any external gym dependency.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain all columns in TRADING_FEATURE_COLS (including 'log_return').
    transaction_cost : float
        Fractional cost applied on each position *change* (default 0.001 = 0.1%).
    coin : str
        Identifier used for display / metadata (default 'BTC').
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, transaction_cost: float = 0.001, coin: str = "BTC"):
        # ── Validate ───────────────────────────────────────────────────────────
        _missing = [c for c in TRADING_FEATURE_COLS if c not in df.columns]
        if _missing:
            raise ValueError(f"[TradingEnv] Missing feature columns: {_missing}")

        self.df               = df.reset_index(drop=True)
        self.feature_cols     = TRADING_FEATURE_COLS
        self.transaction_cost = transaction_cost
        self.coin             = coin
        self._n_steps_total   = len(self.df)

        # Pre-extract ONLY the 13 feature columns as float32 numpy array (NaN→0)
        self._obs_array   = self.df[self.feature_cols].values.astype(np.float32)
        self._obs_array   = np.nan_to_num(self._obs_array, nan=0.0)
        self._log_returns = self.df["log_return"].values.astype(np.float64)

        # ── Gym spaces ────────────────────────────────────────────────────────
        n_features = len(self.feature_cols)            # 13
        # Unbounded obs space — financial features are unnormalised by design
        self.observation_space = _BoxSpace(
            low=-np.inf, high=np.inf,
            shape=(n_features,),
            dtype=np.float32,
        )
        self.action_space = _DiscreteSpace(n=3)

        # Internal episode state
        self._current_step:     int   = 0
        self._current_position: float = 0.0  # starts as HOLD
        self._total_reward:     float = 0.0
        self._episode_rewards:  list  = []
        self._rng = np.random.default_rng()

    # ── Core Gym API ─────────────────────────────────────────────────────────

    def _get_info(self) -> dict:
        return {
            "step":         self._current_step,
            "position":     self._current_position,
            "total_reward": self._total_reward,
            "coin":         self.coin,
        }

    def reset(self, *, seed=None, options=None):
        """Reset to beginning of episode. Returns (obs, info)."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.action_space.seed(seed)
        self._current_step      = 0
        self._current_position  = 0.0
        self._total_reward      = 0.0
        self._episode_rewards   = []
        return self._obs_array[0].copy(), self._get_info()

    def step(self, action: int):
        """
        Execute one timestep.
        Returns (obs, reward, terminated, truncated, info).
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}. Valid: {{0=SELL, 1=HOLD, 2=BUY}}")

        new_position = _POSITION_MAP[int(action)]
        log_ret      = self._log_returns[self._current_step]

        # Reward: position-weighted log return − transaction cost on position change
        position_reward = self._current_position * log_ret
        delta_position  = abs(new_position - self._current_position)
        cost_penalty    = self.transaction_cost * delta_position
        reward          = float(position_reward - cost_penalty)

        self._current_position  = new_position
        self._total_reward     += reward
        self._episode_rewards.append(reward)
        self._current_step     += 1

        terminated = self._current_step >= self._n_steps_total
        truncated  = False

        obs = (self._obs_array[self._current_step].copy()
               if not terminated
               else np.zeros(len(self.feature_cols), dtype=np.float32))

        return obs, reward, terminated, truncated, self._get_info()

    def render(self, mode="human"):
        print(f"  [{self.coin}] step={self._current_step:>6,} | "
              f"pos={self._current_position:>+.0f} | "
              f"cumulative_reward={self._total_reward:>+.6f}")

    def close(self):
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  Lightweight Env Registry  (gym.register / gym.make pattern)
# ══════════════════════════════════════════════════════════════════════════════

_ENV_REGISTRY: dict = {}

def register_env(env_id: str, entry_point):
    _ENV_REGISTRY[env_id] = entry_point

def make_env(env_id: str, **kwargs) -> TradingEnv:
    if env_id not in _ENV_REGISTRY:
        raise KeyError(f"Env '{env_id}' not registered. Available: {list(_ENV_REGISTRY.keys())}")
    return _ENV_REGISTRY[env_id](**kwargs)

register_env("TradingEnv-v0", TradingEnv)
print(f"  ✅  Registered 'TradingEnv-v0' → {TradingEnv.__name__}")
print(f"      obs_space  : Box(-inf, inf, shape=({len(TRADING_FEATURE_COLS)},), float32)")
print(f"      act_space  : Discrete(3)  →  0=SELL | 1=HOLD | 2=BUY")
print(f"      features   : {TRADING_FEATURE_COLS}")


# ══════════════════════════════════════════════════════════════════════════════
#  Sanity-Check: Random Agent on BTC PPO Feature Matrix
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  SANITY CHECK — Random Agent on BTC PPO Feature Matrix")
print("═" * 70)

# Select only the required columns (no duplicate log_return)
_btc_matrix = (ppo_feature_matrices["BTC"][_REQUIRED_COLS]
               .dropna()
               .reset_index(drop=True))

print(f"\n  BTC feature matrix : {_btc_matrix.shape[0]:,} rows × {_btc_matrix.shape[1]} cols")
print(f"  Columns selected   : {list(_btc_matrix.columns)}")
print(f"  Transaction cost   : 0.1%  |  n_features = {len(TRADING_FEATURE_COLS)}")

# ── Instantiate ───────────────────────────────────────────────────────────────
_env = make_env("TradingEnv-v0", df=_btc_matrix, transaction_cost=0.001, coin="BTC")
_obs, _info = _env.reset(seed=42)

print(f"\n  obs shape          : {_obs.shape}   (expected: ({len(TRADING_FEATURE_COLS)},))")
print(f"  obs dtype          : {_obs.dtype}")
print(f"  obs (first 5 vals) : {_obs[:5]}")
print(f"  obs in space       : {_env.observation_space.contains(_obs)}")
print(f"  action space       : {_env.action_space}")
print(f"  n_actions          : {_env.action_space.n}")

assert _obs.shape == (len(TRADING_FEATURE_COLS),), \
    f"❌ obs shape {_obs.shape} ≠ ({len(TRADING_FEATURE_COLS)},)"
assert _obs.dtype == np.float32,                   "❌ obs dtype ≠ float32"
assert _env.observation_space.contains(_obs),      "❌ obs not in observation_space!"
assert _env.action_space.n == 3,                   "❌ Expected 3 actions!"
print(f"\n  ✅  All space assertions passed.")

# ── Random agent episode ─────────────────────────────────────────────────────
print("\n  Running random agent episode …")

_obs, _info = _env.reset(seed=0)
_done        = False
_total_rew   = 0.0
_step_count  = 0
_action_hist = {0: 0, 1: 0, 2: 0}

while not _done:
    _action = _env.action_space.sample()
    _obs, _reward, _terminated, _truncated, _info = _env.step(_action)
    _total_rew  += _reward
    _step_count += 1
    _action_hist[_action] += 1
    _done = _terminated or _truncated

_env.close()

_rewards_arr = np.array(_env._episode_rewards)
_reward_min  = float(np.min(_rewards_arr))
_reward_max  = float(np.max(_rewards_arr))
_reward_mean = float(np.mean(_rewards_arr))
_reward_std  = float(np.std(_rewards_arr))

print(f"\n  ─── Random Agent Episode Results ─────────────────────────────────")
print(f"  Steps completed    : {_step_count:,}")
print(f"  Total reward       : {_total_rew:>+.6f}")
print(f"  Mean reward/step   : {_reward_mean:>+.8f}")
print(f"  Std  reward/step   : {_reward_std:>+.8f}")
print(f"  Reward range       : [{_reward_min:>+.8f},  {_reward_max:>+.8f}]")
print(f"  Action histogram   : SELL={_action_hist[0]:,}  HOLD={_action_hist[1]:,}  BUY={_action_hist[2]:,}")
print(f"  Reward type check  : {'PASS ✅' if isinstance(_total_rew, float) else 'FAIL ❌'}")

# ── Per-coin validation ───────────────────────────────────────────────────────
print(f"\n  ─── Per-Coin Env Instantiation & Random Agent Check ──────────────")
print(f"  {'COIN':<10} {'ROWS':>9} {'OBS_OK':>7} {'STEPS':>9} {'REWARD':>14}")
print(f"  {'-'*56}")

trading_env_stats = {}

for _coin in PPO_COINS:
    _mat = (ppo_feature_matrices[_coin][_REQUIRED_COLS]
            .dropna().reset_index(drop=True))
    _e   = TradingEnv(df=_mat, transaction_cost=0.001, coin=_coin)
    _o, _ = _e.reset(seed=7)
    _obs_ok  = _e.observation_space.contains(_o)
    _r_total = 0.0
    _n_steps = 0
    _d = False
    while not _d:
        _a = _e.action_space.sample()
        _o, _r, _term, _trunc, _ = _e.step(_a)
        _r_total += _r
        _n_steps += 1
        _d = _term or _trunc
    _e.close()
    trading_env_stats[_coin] = {"rows": len(_mat), "reward": _r_total, "steps": _n_steps}
    print(f"  {_coin:<10} {len(_mat):>9,}  {'✅' if _obs_ok else '❌':>7} "
          f"{_n_steps:>9,} {_r_total:>+14.6f}")

print(f"\n{'═'*70}")
print(f"  ✅  TradingEnv instantiates and steps without error")
print(f"  ✅  Observation space : Box(-inf, inf, shape=({len(TRADING_FEATURE_COLS)},), float32)")
print(f"  ✅  Action space      : Discrete(3) → 0=SELL | 1=HOLD | 2=BUY")
print(f"  ✅  Random agent total reward (BTC): {_total_rew:>+.6f}  over {_step_count:,} steps")
print(f"  ✅  Reward range      : [{_reward_min:>+.8f},  {_reward_max:>+.8f}]")
print(f"  ✅  All {len(PPO_COINS)} coins validated  — registry: {list(_ENV_REGISTRY.keys())}")
print(f"{'═'*70}")

# ── Export for downstream use ────────────────────────────────────────────────
trading_env_class        = TradingEnv
trading_env_feature_cols = TRADING_FEATURE_COLS
trading_env_registry     = _ENV_REGISTRY
