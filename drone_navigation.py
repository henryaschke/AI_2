"""
Assignment 2 – Question 1
Tile-Coded Approximate Q-Learning for Drone Navigation

This script maps each sub-question or sub-part of Question 1 to specific sections:

Sub-question (a): Feature representation & update equation - See TileCoder and update method
Sub-question (b): Implementation (environment + agent) - DroneWindEnv and ApproxQLearner classes  
Sub-question (c): Hyper-parameter tuning - Hyperparameters section
Sub-question (d): Performance evaluation - Evaluation section
"""

from typing import Tuple
from dataclasses import dataclass
import numpy as np

# ============================================================================
# A. Sub-question (a): Feature Representation & Update Equation
# ============================================================================

"""
We discretise the continuous position (x,y) using a uniform grid  
implemented by TileCoder.  
The linear value function for a given action a is  

Q̂(s,a) = w_a^T φ(s)

where φ(s) is a one-hot vector with 1 at the active tile.

TD update used in ApproxQLearner.update:

δ = {
    r - Q(s,a)                        if s' is terminal,
    r + γ max_a' Q(s',a') - Q(s,a)   otherwise.
}

w_a ← w_a + α δ φ(s)

Below is the code for the tile coder and comments highlighting
where this update is implemented.
"""

@dataclass
class TileCoder:
    """Uniform grid over (x,y) – wind dimension ignored for features."""
    nx: int
    ny: int
    x_range: Tuple[float, float] = (0.0, 1.0)
    y_range: Tuple[float, float] = (0.0, 1.0)

    def n_tiles(self) -> int:
        return self.nx * self.ny

    def tile_index(self, state: np.ndarray) -> int:
        """Map (x,y) to a single tile id."""
        x, y = state[0], state[1]

        # Compute discrete bin indices
        ix = ### fill up here
        iy = ### fill up here

        # Row-major flatten
        return iy * self.nx + ix

# ============================================================================
# B. Sub-question (b): Implementation (Environment & Agent)
# ============================================================================

"""
The next two classes provide:
* DroneWindEnv — the windy-field environment  
* ApproxQLearner — the agent implementing the TD update from Section A  
  Key lines are commented as "Sub-Q1 b".
"""

class DroneWindEnv:
    """Continuous 2-D field with global east–west wind."""

    def __init__(self, max_steps: int = 200, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.action_space = 4  # 0=N,1=S,2=E,3=W
        self.state: np.ndarray | None = None
        self._step_count = 0

    # -- physics
    @staticmethod
    def _next_state(state: np.ndarray, action: int) -> np.ndarray:
        x, y, w = state
        base_move = 0.05

        if action == 0:   # North
            dx, dy = 0.0, base_move
        elif action == 1: # South
            dx, dy = 0.0, -base_move
        elif action == 2: # East (head-wind)
            dx, dy = base_move * (1.0 - 0.5 * w), 0.0
        elif action == 3: # West (tail-wind)
            dx, dy = -base_move * (1.0 + 0.5 * w), 0.0
        else:
            raise ValueError("Bad action")

        new_x = np.clip(x + dx, 0.0, 1.0)
        new_y = np.clip(y + dy, 0.0, 1.0)
        new_w = np.clip(w + 0.01 * (np.random.rand() - 0.5), 0.0, 1.0)

        return np.array([new_x, new_y, new_w], dtype=np.float32)

    # -- Gym-like API
    def reset(self) -> np.ndarray:
        self.state = np.array([0.05, 0.05, self.rng.random()], dtype=np.float32)
        self._step_count = 0
        return self.state.copy()

    def step(self, action: int):
        self._step_count += 1
        next_state = self._next_state(self.state, action)

        # +10 in charging zone (NE corner), else -1 per step
        reward = 10.0 if (next_state[0] > 0.9 and next_state[1] > 0.9) else -1.0
        done = reward == 10.0 or self._step_count >= self.max_steps

        self.state = next_state
        return next_state.copy(), reward, done, {}


class ApproxQLearner:
    """Linear Approximate Q-learning with a single tiling."""

    def __init__(self,
                 tile_coder: TileCoder,
                 n_actions: int,
                 alpha: float = 0.1,
                 gamma: float = 0.99,
                 epsilon: float = 0.1,
                 seed: int | None = None):
        self.tc = tile_coder
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

        # Weight matrix w[a, tile]
        self.weights = np.zeros((n_actions, self.tc.n_tiles()), dtype=np.float32)

    # -- helpers
    def _phi(self, state: np.ndarray) -> int:
        return self.tc.tile_index(state)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        tid = self._phi(state)
        return self.weights[:, tid]

    def select_action(self, state: np.ndarray) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.integers(self.n_actions)
        return int(np.argmax(self.q_values(state)))

    # -- TD update implementing Sub-question (a) formula
    def update(self, s: np.ndarray, a: int, r: float, s_next: np.ndarray, done: bool):
        tid = self._phi(s)
        q_sa = self.weights[a, tid]                # current estimate

        target = ### fill up here
        td_error = ### fill up here                  # δ
        self.weights[a, tid] += ### fill up here  # gradient step

    # -- training loop
    def train(self, env: DroneWindEnv, episodes: int = 1000, max_steps: int = 200):
        history = []
        ### fill up here
        return history

    # -- evaluation
    def evaluate(self, env: DroneWindEnv, episodes: int = 100, max_steps: int = 200):
        total = 0.0
        ### fill up here
        return total / episodes

# ============================================================================
# C. Sub-question (c): Hyper-parameter Tuning
# ============================================================================

"""
Adjust the parameters below and re-run training to observe their impact.
Document your chosen values in the assignment write-up (pdf file).
"""

# --- Hyper-parameters (Sub-Q1 c) ---
NX, NY = 10, 10        # tile resolution
ALPHA   = 0.2          # learning rate
GAMMA   = 0.95         # discount factor
EPSILON = 0.2          # initial exploration rate
SEED    = 42

def run_training():
    """Run the training with the specified hyperparameters."""
    env   = DroneWindEnv(max_steps=200, seed=SEED)
    tc    = TileCoder(nx=NX, ny=NY)
    agent = ApproxQLearner(tc, n_actions=env.action_space,
                           alpha=ALPHA, gamma=GAMMA,
                           epsilon=EPSILON, seed=SEED)

    print("Training...")
    history = agent.train(env, episodes=500)
    return env, agent, history

# ============================================================================
# D. Sub-question (d): Performance Evaluation
# ============================================================================

"""
Run the evaluation below to compute the average reward over 100 evaluation
episodes and fill in the rubric table in your PDF solution.
"""

def run_evaluation(env, agent):
    """Run evaluation and print results."""
    avg_reward = agent.evaluate(env, episodes=100)
    print(f"Average reward over 100 evaluation episodes: {avg_reward:.2f}")
    return avg_reward

# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Run training
    env, agent, history = run_training()
    
    # Run evaluation
    avg_reward = run_evaluation(env, agent) 