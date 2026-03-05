"""RL agents for energy trading (numpy-based, no torch/tensorflow required)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class RandomAgent:
    """Baseline random policy."""
    def act(self, state: np.ndarray) -> float:
        return float(np.random.uniform(-1, 1))

    def learn(self, *args) -> None:
        pass


class RuleBasedAgent:
    """Mean-reversion rule agent: buy when z < -z_entry, sell when z > z_entry."""

    def __init__(self, window: int = 20, z_entry: float = 2.0, z_exit: float = 0.5):
        self._window = window
        self._z_entry = z_entry
        self._z_exit = z_exit
        self._history: List[float] = []
        self._position: float = 0.0

    def act(self, state: np.ndarray) -> float:
        price = float(state[-1]) if len(state) > 0 else 0.0
        self._history.append(price)
        if len(self._history) < self._window:
            return 0.0
        hist = np.array(self._history[-self._window:])
        z = (hist[-1] - hist.mean()) / (hist.std() + 1e-8)
        if z < -self._z_entry:
            self._position = 1.0
        elif z > self._z_entry:
            self._position = -1.0
        elif abs(z) < self._z_exit:
            self._position = 0.0
        return self._position

    def learn(self, *args) -> None:
        pass


class LinearQAgent:
    """Linear function approximation Q-learning agent."""

    def __init__(self, state_dim: int, n_actions: int = 3,
                 lr: float = 0.01, gamma: float = 0.99, epsilon: float = 0.1):
        self.state_dim = state_dim
        self.n_actions = n_actions  # discrete: -1, 0, +1
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        # Q(s, a) = w_a . s + b_a
        self._W = np.zeros((n_actions, state_dim))
        self._b = np.zeros(n_actions)
        self._actions = [-1.0, 0.0, 1.0][:n_actions]

    def act(self, state: np.ndarray) -> float:
        if np.random.rand() < self.epsilon:
            return float(np.random.choice(self._actions))
        q_vals = self._W @ state + self._b
        return self._actions[int(np.argmax(q_vals))]

    def learn(self, state: np.ndarray, action: float, reward: float,
              next_state: np.ndarray, done: bool) -> None:
        a_idx = self._actions.index(float(action)) if float(action) in self._actions else 1
        q_current = float(self._W[a_idx] @ state + self._b[a_idx])
        if done:
            q_target = reward
        else:
            q_next = float(np.max(self._W @ next_state + self._b))
            q_target = reward + self.gamma * q_next
        td_error = q_target - q_current
        self._W[a_idx] += self.lr * td_error * state
        self._b[a_idx] += self.lr * td_error

    def update_epsilon(self, episode: int, decay: float = 0.995,
                        min_eps: float = 0.01) -> None:
        self.epsilon = max(min_eps, self.epsilon * decay)


class PolicyGradientAgent:
    """REINFORCE with baseline (numpy)."""

    def __init__(self, state_dim: int, action_dim: int = 1,
                 hidden_dim: int = 32, lr: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        # Two-layer policy network
        scale = np.sqrt(2.0 / (state_dim + hidden_dim))
        self._W1 = np.random.randn(hidden_dim, state_dim) * scale
        self._b1 = np.zeros(hidden_dim)
        self._W2 = np.random.randn(action_dim, hidden_dim) * 0.01
        self._b2 = np.zeros(action_dim)
        # Baseline (value function) — linear
        self._Wv = np.zeros(state_dim)
        self._bv = 0.0

    def _forward(self, s: np.ndarray) -> Tuple[np.ndarray, float]:
        h = _relu(self._W1 @ s + self._b1)
        mu = np.tanh(self._W2 @ h + self._b2)  # action mean in [-1,1]
        log_std = np.full_like(mu, -1.0)  # fixed log std
        return mu, float(log_std[0])

    def act(self, state: np.ndarray) -> Tuple[float, float]:
        mu, log_std = self._forward(state)
        std = np.exp(log_std)
        action = float(np.clip(mu[0] + std * np.random.randn(), -1.0, 1.0))
        log_prob = float(-0.5 * ((action - mu[0]) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi))
        return action, log_prob

    def update(self, episode_states: List[np.ndarray],
               episode_actions: List[float],
               episode_rewards: List[float]) -> None:
        """REINFORCE update."""
        T = len(episode_rewards)
        if T == 0:
            return
        # Compute returns
        G = np.zeros(T)
        gamma = 0.99
        g = 0.0
        for t in reversed(range(T)):
            g = episode_rewards[t] + gamma * g
            G[t] = g
        G = (G - G.mean()) / (G.std() + 1e-8)

        for t in range(T):
            s = episode_states[t]
            a = episode_actions[t]
            mu, log_std = self._forward(s)
            std = np.exp(log_std)
            baseline = float(self._Wv @ s + self._bv)
            advantage = G[t] - baseline
            # Policy gradient
            h = _relu(self._W1 @ s + self._b1)
            d_log_pi_dmu = (a - mu[0]) / (std ** 2)
            d_mu_dW2 = h
            self._W2 += self.lr * advantage * d_log_pi_dmu * d_mu_dW2
            self._b2 += self.lr * advantage * d_log_pi_dmu
            # Baseline update
            self._Wv += self.lr * 0.1 * advantage * s
            self._bv += self.lr * 0.1 * advantage


class PPOAgent:
    """Simplified PPO with linear policy (numpy)."""

    def __init__(self, state_dim: int, clip_eps: float = 0.2,
                 lr: float = 0.001, gamma: float = 0.99,
                 lambda_gae: float = 0.95):
        self.state_dim = state_dim
        self.clip_eps = clip_eps
        self.lr = lr
        self.gamma = gamma
        self.lam = lambda_gae
        # Linear policy: action = W @ state + b, with Gaussian noise
        self._W = np.zeros(state_dim)
        self._b = 0.0
        self._log_std = np.array(-1.0)
        self._Wv = np.zeros(state_dim)
        self._bv = 0.0

    def act(self, state: np.ndarray) -> Tuple[float, float]:
        mu = float(self._W @ state + self._b)
        std = float(np.exp(self._log_std))
        action = float(np.clip(mu + std * np.random.randn(), -1.0, 1.0))
        log_prob = float(-0.5 * ((action - mu) / std) ** 2 - self._log_std
                         - 0.5 * np.log(2 * np.pi))
        return action, log_prob

    def update(self, states: np.ndarray, actions: np.ndarray,
               rewards: np.ndarray, log_probs_old: np.ndarray,
               n_epochs: int = 4) -> Dict[str, float]:
        """PPO update step."""
        T = len(rewards)
        # Compute advantages via GAE
        values = states @ self._Wv + self._bv
        advantages = np.zeros(T)
        last_gae = 0.0
        for t in reversed(range(T)):
            v_next = values[t + 1] if t < T - 1 else 0.0
            delta = rewards[t] + self.gamma * v_next - values[t]
            last_gae = delta + self.gamma * self.lam * last_gae
            advantages[t] = last_gae
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        for _ in range(n_epochs):
            for t in range(T):
                s = states[t]
                a = actions[t]
                mu = float(self._W @ s + self._b)
                std = float(np.exp(self._log_std))
                log_prob = float(-0.5 * ((a - mu) / std) ** 2 - self._log_std
                                  - 0.5 * np.log(2 * np.pi))
                ratio = np.exp(log_prob - log_probs_old[t])
                clipped_ratio = np.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                policy_loss = -min(ratio * advantages[t], clipped_ratio * advantages[t])
                total_policy_loss += policy_loss
                # Update
                self._W -= self.lr * policy_loss * s
                self._b -= self.lr * policy_loss
            # Value function update
            v_pred = states @ self._Wv + self._bv
            returns = advantages + values
            v_loss = returns - v_pred
            self._Wv += self.lr * states.T @ v_loss / T
            self._bv += self.lr * v_loss.mean()

        return {"policy_loss": float(total_policy_loss / T)}


class AgentEvaluator:
    """Evaluates and compares RL agents."""

    def evaluate(self, agent: Any, env: Any, n_episodes: int = 10) -> Dict:
        total_rewards = []
        for _ in range(n_episodes):
            state = env.reset()
            episode_reward = 0.0
            done = False
            returns_list = []
            while not done:
                if hasattr(agent, "act"):
                    result = agent.act(state)
                    action = result[0] if isinstance(result, tuple) else result
                else:
                    action = 0.0
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                returns_list.append(reward)
            total_rewards.append(episode_reward)

        rewards_arr = np.array(total_rewards)
        return {
            "mean_reward": float(rewards_arr.mean()),
            "std_reward": float(rewards_arr.std()),
            "sharpe": float(rewards_arr.mean() / (rewards_arr.std() + 1e-8) * np.sqrt(n_episodes)),
            "max_reward": float(rewards_arr.max()),
            "min_reward": float(rewards_arr.min()),
        }

    def compare_agents(self, agents_dict: Dict[str, Any], env: Any,
                        n_episodes: int = 10) -> pd.DataFrame:
        rows = []
        for name, agent in agents_dict.items():
            metrics = self.evaluate(agent, env, n_episodes)
            rows.append({"agent": name, **metrics})
        return pd.DataFrame(rows)
