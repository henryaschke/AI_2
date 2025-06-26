"""
Assignment 2 – Question 2
Deep Q-Learning for Pacman

This script implements a DQN agent for playing Pacman. The code is organized into distinct sections
that map to the assignment sub-questions:

Sub-question (a): Improve Algorithm - Enhance training speed through algorithmic improvements
Sub-question (b): Hyperparameter Tuning - Fine-tune at least 2 hyperparameters  
Sub-question (c): Performance Analysis - Evaluate and compare results

Code Structure Overview:
1. Configuration - All hyperparameters and reward values (MODIFY for sub-question b)
2. Game Environment - DO NOT MODIFY (just run and skip)
3. DQN Core Algorithm - MODIFY for sub-question (a) if you want to improve algorithm
4. Training Loop - MODIFY for sub-question (a) if you want to improve training strategy
5. Testing & Evaluation - MODIFY for sub-question (c)
"""

import math
import random
import time
from collections import deque
from typing import Dict, Any

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ============================================================================
# 1. Configuration (Hyperparameters & Rewards)
# ============================================================================

"""
For sub-question (b): Tune hyperparameters here.
For sub-question (a): You'll modify the algorithm implementation in the DQN classes, not here.
"""

# ===== REWARD VALUES =====
# These define the reward structure of the game
STEP_PENALTY = -0.1  # Penalty for each time step
WALL_PENALTY = -1  # Additional penalty for hitting walls
PELLET_REWARD = 10  # Reward for eating a normal pellet
POWER_PELLET_REWARD = 50  # Reward for eating a power pellet
GHOST_REWARD = 200  # Reward for eating a ghost (when frightened)
DEATH_PENALTY = -100  # Penalty for being caught by a ghost
VICTORY_REWARD = 200  # Reward for clearing all pellets

# ===== HYPERPARAMETERS (Sub-question b: Tune these) =====
# Exploration parameters
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_END = 0.01  # Final exploration rate
EPSILON_DECAY = 0.999  # Decay rate per episode

# Learning parameters
LEARNING_RATE = 0.00025  # Neural network learning rate
BATCH_SIZE = 128  # Batch size for training
GAMMA = 0.99  # Discount factor for future rewards

# Network update parameters
TARGET_UPDATE_STEPS = 1000  # Steps between target network updates

# Training duration
NUM_EPISODES = 1000  # Total training episodes

# ===== DEFAULT ALGORITHM SETTINGS =====
# These are baseline settings. For sub-question (a), you'll modify
# the actual implementation in the DQN classes, not just these values.
FRAME_STACK_SIZE = 2  # Number of frames to stack
REPLAY_BUFFER_CAPACITY = 10000  # Size of experience replay buffer
LEARN_EVERY_N_STEPS = 2  # How often to perform learning update

# ===== GAME PARAMETERS (Do not modify) =====
SCREEN_WIDTH, SCREEN_HEIGHT = 280, 316
GRID_SIZE = 28
FRIGHTENED_DURATION = 10

# --- Color Definitions ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
PINK = (255, 182, 193)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)

# ============================================================================
# 2. Game Environment (DO NOT MODIFY)
# ============================================================================

"""
This section contains the Pacman game environment. 
DO NOT MODIFY - Just run this and skip to the DQN algorithm section.
"""

class PacmanGame:
    """
    Environment class that encapsulates Pacman game logic and rendering.
    All reward values are defined in the Configuration section.
    """

    def __init__(self):
        pygame.init()
        self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        #pygame.display.set_caption("DQN Pac-Man")
        self.clock = pygame.time.Clock()

        self.font = pygame.font.Font(None, 20)
        # Game map (1: wall, 0: empty, 2: pellet, 3: power pellet)
        self.level_map = [
            "1111111111",
            "1223222221",
            "1211221121",
            "1222222221",
            "1211221121",
            "1222222221",
            "1111111111",
        ]
        self.width = len(self.level_map[0])
        self.height = len(self.level_map)
        self.action_space = 4  # 0:Up, 1:Down, 2:Left, 3:Right
        self.training_info = {}
        self.hit_wall = False
        self.frightened_timer = 0
        self.frightened_duration = FRIGHTENED_DURATION
        self.reset()

    def reset(self):
        """Reset game state to the initial configuration."""
        self.pacman_pos = [1, 3]  # Pacman starting position
        self.pacman_direction = (
            3  # Pacman facing direction (0:Up, 1:Down, 2:Left, 3:Right)
        )
        self.ghosts = {
            "blinky": {"pos": [1, 1], "color": RED, "direction": 3},
            "pinky": {"pos": [8, 5], "color": PINK, "direction": 2},
        }
        self.score = 0
        self.pellets = []
        self.power_pellets = []
        self.walls = []
        for r, row in enumerate(self.level_map):
            for c, char in enumerate(row):
                if char == "1":
                    self.walls.append(
                        pygame.Rect(
                            c * GRID_SIZE, r * GRID_SIZE, GRID_SIZE, GRID_SIZE
                        )
                    )
                elif char == "2":
                    self.pellets.append(
                        pygame.Rect(
                            c * GRID_SIZE + GRID_SIZE // 2,
                            r * GRID_SIZE + GRID_SIZE // 2,
                            4,
                            4,
                        )
                    )
                elif char == "3":
                    self.power_pellets.append(
                        pygame.Rect(
                            c * GRID_SIZE + GRID_SIZE // 2,
                            r * GRID_SIZE + GRID_SIZE // 2,
                            8,
                            8,
                        )
                    )
        self.done = False
        self.hit_wall = False
        self.frightened_timer = 0
        self._render_game_state()  # Initial render
        return self._get_state()

    def _get_state(self):
        """Get current game screen pixels as state."""
        frame = pygame.surfarray.array3d(self.screen)
        # Convert to grayscale to reduce state dimensions
        frame = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
        frame = frame.T  # Transpose to match (Height, Width)
        frame = frame.astype(np.uint8)
        # Add channel dimension to match CNN input format (C, H, W)
        frame = np.expand_dims(frame, axis=0)
        return frame

    def _move(self, pos, direction):
        """Move position based on direction, with screen wrapping."""
        if direction == 0:
            pos[1] -= 1  # Up
        elif direction == 1:
            pos[1] += 1  # Down
        elif direction == 2:
            pos[0] -= 1  # Left
        elif direction == 3:
            pos[0] += 1  # Right
        # Handle wrapping around the screen
        if pos[0] < 0:
            pos[0] = self.width - 1
        if pos[0] >= self.width:
            pos[0] = 0
        return pos

    def _check_collision(self, pos):
        """Check if a given grid position collides with a wall."""
        rect = pygame.Rect(
            pos[0] * GRID_SIZE, pos[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE
        )
        for wall in self.walls:
            if rect.colliderect(wall):
                return True
        return False

    def _get_valid_moves(self, pos):
        """Get all valid (non-wall) move directions from a given position."""
        valid_moves = []
        for direction in range(4):  # 0:Up, 1:Down, 2:Left, 3:Right
            potential_pos = self._move(list(pos), direction)
            if not self._check_collision(potential_pos):
                valid_moves.append(direction)
        return valid_moves

    def step(self, action):
        """Execute one action step, return next_state, reward, done."""
        # Use reward values from Configuration section
        reward = STEP_PENALTY

        # Decrement frightened timer if active
        if self.frightened_timer > 0:
            self.frightened_timer -= 1

        # Move Pacman
        new_pos = list(self.pacman_pos)
        potential_pos = self._move(list(new_pos), action)
        if not self._check_collision(potential_pos):
            self.pacman_pos = potential_pos
            self.pacman_direction = action  # Update Pacman's facing direction
            self.hit_wall = False
        else:
            self.hit_wall = True
            reward += WALL_PENALTY

        pacman_rect = pygame.Rect(
            self.pacman_pos[0] * GRID_SIZE,
            self.pacman_pos[1] * GRID_SIZE,
            GRID_SIZE,
            GRID_SIZE,
        )

        # Check rewards from eating pellets
        eaten_pellet = pacman_rect.collidelist(self.pellets)
        if eaten_pellet != -1:
            self.pellets.pop(eaten_pellet)
            self.score += PELLET_REWARD
            reward += PELLET_REWARD

        eaten_power_pellet = pacman_rect.collidelist(self.power_pellets)
        if eaten_power_pellet != -1:
            self.power_pellets.pop(eaten_power_pellet)
            self.score += POWER_PELLET_REWARD
            reward += POWER_PELLET_REWARD
            # Activate frightened state for ghosts
            self.frightened_timer = self.frightened_duration

        # Collision detection with ghosts (check before ghosts move)
        for g in self.ghosts.values():
            ghost_rect = pygame.Rect(
                g["pos"][0] * GRID_SIZE,
                g["pos"][1] * GRID_SIZE,
                GRID_SIZE,
                GRID_SIZE,
            )
            if pacman_rect.colliderect(ghost_rect):
                if self.frightened_timer > 0:
                    self.score += GHOST_REWARD
                    reward += GHOST_REWARD
                    g["pos"] = [1, 1]  # Send eaten ghost back to start
                else:
                    self.done = True
                    reward = DEATH_PENALTY
                    self._render_game_state()
                    return self._get_state(), reward, self.done

        # Move ghosts
        for g in self.ghosts.values():
            valid_moves = self._get_valid_moves(g["pos"])
            # Avoid immediate reversal unless there is no other choice
            if len(valid_moves) > 1:
                reverse_map = {0: 1, 1: 0, 2: 3, 3: 2}
                reverse_direction = reverse_map.get(g["direction"])
                if reverse_direction in valid_moves:
                    valid_moves.remove(reverse_direction)

            if valid_moves:
                ghost_action = random.choice(valid_moves)
                g["pos"] = self._move(list(g["pos"]), ghost_action)
                g["direction"] = ghost_action

        # Check game over conditions (caught by a ghost or all pellets eaten)
        for g in self.ghosts.values():
            ghost_rect = pygame.Rect(
                g["pos"][0] * GRID_SIZE,
                g["pos"][1] * GRID_SIZE,
                GRID_SIZE,
                GRID_SIZE,
            )
            if pacman_rect.colliderect(ghost_rect):
                if self.frightened_timer > 0:
                    self.score += GHOST_REWARD
                    reward += GHOST_REWARD
                    g["pos"] = [1, 1]  # Send eaten ghost back to start
                else:
                    self.done = True
                    reward = DEATH_PENALTY
                    break

        if not self.pellets and not self.power_pellets:
            self.done = True
            reward = VICTORY_REWARD

        # Render the game state and get the new state
        self._render_game_state()
        next_state = self._get_state()

        return next_state, reward, self.done

    def _render_game_state(self):
        """Render the game world to the screen surface without updating the display."""
        self.screen.fill(BLACK)
        for wall in self.walls:
            pygame.draw.rect(self.screen, BLUE, wall)
        for pellet in self.pellets:
            pygame.draw.rect(self.screen, WHITE, pellet)
        for p_pellet in self.power_pellets:
            pygame.draw.rect(self.screen, WHITE, p_pellet)

        self._draw_pacman()
        for g in self.ghosts.values():
            self._draw_ghost(g["pos"], g["color"])
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))

    def _draw_pacman(self):
        """Draw Pacman with its mouth oriented based on its direction."""
        pos_x = self.pacman_pos[0] * GRID_SIZE + GRID_SIZE // 2
        pos_y = self.pacman_pos[1] * GRID_SIZE + GRID_SIZE // 2
        radius = GRID_SIZE // 2
        center = (pos_x, pos_y)

        # Define mouth angles for each direction
        angles = {
            0: (math.pi / 4, 3 * math.pi / 4),  # Up
            1: (5 * math.pi / 4, 7 * math.pi / 4),  # Down
            2: (3 * math.pi / 4, 5 * math.pi / 4),  # Left
            3: (-math.pi / 4, math.pi / 4),  # Right
        }

        # Correct for Pygame's inverted Y-axis. The visual representation for "up"
        # and "down" movement needs to be swapped.
        visual_correction_angles = {
            0: angles[1],
            1: angles[0],
            2: angles[2],
            3: angles[3],
        }
        start_angle, end_angle = visual_correction_angles[self.pacman_direction]

        # Draw body and mouth
        pygame.draw.circle(self.screen, YELLOW, center, radius)
        points = [center]
        for n in range(16):
            theta = start_angle + (end_angle - start_angle) * n / 15
            points.append(
                (
                    center[0] + radius * math.cos(theta),
                    center[1] + radius * math.sin(theta),
                )
            )
        pygame.draw.polygon(self.screen, BLACK, points)

    def _draw_ghost(self, pos, color):
        """Draw a ghost shape."""
        # Change ghost color if in frightened state
        current_color = GRAY if self.frightened_timer > 0 else color

        pos_x = pos[0] * GRID_SIZE
        pos_y = pos[1] * GRID_SIZE
        radius = GRID_SIZE // 2
        body_rect = pygame.Rect(pos_x, pos_y + radius, GRID_SIZE, radius)

        # Body
        pygame.draw.rect(
            self.screen,
            current_color,
            body_rect,
            border_bottom_left_radius=3,
            border_bottom_right_radius=3,
        )
        # Head
        pygame.draw.circle(
            self.screen, current_color, (pos_x + radius, pos_y + radius), radius
        )
        # Eyes
        eye_radius = GRID_SIZE // 8
        pupil_radius = GRID_SIZE // 16
        pygame.draw.circle(
            self.screen, WHITE, (pos_x + radius - 5, pos_y + radius - 2), eye_radius
        )
        pygame.draw.circle(
            self.screen,
            BLACK,
            (pos_x + radius - 5, pos_y + radius - 2),
            pupil_radius,
        )
        pygame.draw.circle(
            self.screen, WHITE, (pos_x + radius + 5, pos_y + radius - 2), eye_radius
        )
        pygame.draw.circle(
            self.screen,
            BLACK,
            (pos_x + radius + 5, pos_y + radius - 2),
            pupil_radius,
        )

    def render(self, training_info={}):
        """Render the complete game interface, including training info."""
        # Clear bottom info area to prevent old text from lingering
        info_area_rect = pygame.Rect(
            0,
            self.height * GRID_SIZE,
            SCREEN_WIDTH,
            SCREEN_HEIGHT - self.height * GRID_SIZE,
        )
        self.screen.fill(BLACK, info_area_rect)

        # Separate warning messages from regular info for display
        info_to_render = dict(training_info)
        warn_message = info_to_render.pop("WARN", None)

        # Display training info
        if info_to_render:
            items = list(info_to_render.items())
            mid_point = (len(items) + 1) // 2
            # First column
            y_offset = self.height * GRID_SIZE + 10
            for key, value in items[:mid_point]:
                info_text = self.font.render(f"{key}: {value}", True, WHITE)
                self.screen.blit(info_text, (10, y_offset))
                y_offset += 28
            # Second column
            y_offset = self.height * GRID_SIZE + 10
            x_offset = SCREEN_WIDTH // 2
            for key, value in items[mid_point:]:
                info_text = self.font.render(f"{key}: {value}", True, WHITE)
                self.screen.blit(info_text, (x_offset, y_offset))
                y_offset += 28

        # Render warning message at the bottom
        if warn_message:
            warn_text = self.font.render(f"Status: {warn_message}", True, YELLOW)
            warn_rect = warn_text.get_rect(bottomleft=(10, SCREEN_HEIGHT - 10))
            self.screen.blit(warn_text, warn_rect)

        # Update the full display
        pygame.display.flip()
        if self.clock:
            self.clock.tick(30)  # Control game frame rate

# ============================================================================
# 3. DQN Core Algorithm (MODIFY for sub-question a)
# ============================================================================

"""
For sub-question (a): MODIFY THIS SECTION to improve the algorithm.

Possible improvements to consider:
- Implement a warm-up phase in ReplayBuffer before starting to learn
- Modify the loss function (MSE vs Huber)
- Implement prioritized experience replay
- Add reward shaping in the agent
- Modify the network architecture
- Implement Double DQN or Dueling DQN

You are also ENCOURAGED TO TRY YOUR OWN IDEAS
"""

class ReplayBuffer:
    """Experience replay buffer for DQN.

    Sub-question (a): Consider implementing improvements such as:
    - Warm-up phase: Don't start learning until buffer has minimum samples
    - Prioritized replay: Sample important experiences more frequently
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store one experience tuple."""
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor([action], dtype=torch.int64)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.bool)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Randomly sample a batch of experiences."""
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            torch.stack(list(state)),
            torch.cat(action),
            torch.cat(reward),
            torch.stack(list(next_state)),
            torch.cat(done),
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Neural network for approximating Q-values.

    Sub-question (a): Consider modifying the architecture for better performance.
    """

    def __init__(self, input_shape, num_actions):
        super(QNetwork, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(self._get_conv_out(input_shape), 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _get_conv_out(self, shape):
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x / 255.0  # Normalize
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQNAgent:
    """DQN Agent that learns to play Pacman.

    Sub-question (a): Key methods to modify for algorithm improvements:
    - learn(): Implement Double DQN, different loss functions, etc.
    - select_action(): Implement better exploration strategies
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        capacity=REPLAY_BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.memory = ReplayBuffer(capacity)

        # Two networks for stability
        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Sub-question (a): Choose loss function
        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        self.last_loss = 0.0

    def select_action(self, state, epsilon):
        """Epsilon-greedy action selection.

        Sub-question (a): Consider implementing better exploration strategies.
        """
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.max(1)[1].item()

    def learn(self):
        """Update the policy network.

        Sub-question (a): This is where you implement algorithm improvements like:
        - Double DQN: Use policy net to select actions, target net to evaluate
        - Different TD targets
        - Gradient clipping
        """
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )

        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            next_q_values[dones] = 0.0
            target_q_values = rewards + (self.gamma * next_q_values)

        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values.unsqueeze(1))
        self.last_loss = loss.item()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """Copy weights from policy network to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ============================================================================
# 4. Training Loop (MODIFY for sub-question a)
# ============================================================================

"""
For sub-question (a): MODIFY THIS SECTION for training strategies.

Possible improvements:
- Implement different frame stacking strategies
- Add warm-up phase before learning starts
- Modify learning frequency
- Implement reward shaping
- Add training tricks like gradient clipping
"""

def train_agent():
    """Main training loop for the DQN agent."""
    # --- Initialize Environment and Agent ---
    env = PacmanGame()

    # Sub-question (a): Modify frame stacking strategy
    # Currently stacking 2 frames - consider different numbers or methods
    frame_stack_size = FRAME_STACK_SIZE

    # Get state and action dimensions
    state_dim = (frame_stack_size, SCREEN_HEIGHT, SCREEN_WIDTH)
    action_dim = env.action_space

    # Create the DQN agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
    )

    # Initialize training variables
    epsilon = EPSILON_START
    total_steps = 0
    training_info = {}

    # Sub-question (a): Consider implementing a warm-up phase
    # For example: min_buffer_size = 1000
    # Don't start learning until buffer has enough samples

    # --- Start Training Loop ---
    print("--- Training Started ---")
    start_time = time.time()

    for episode in range(NUM_EPISODES):
        # Reset environment and initialize frame stack
        frame = env.reset()
        frame_stack = deque([frame] * frame_stack_size, maxlen=frame_stack_size)
        state = np.concatenate(frame_stack, axis=0)
        episode_reward = 0

        while True:
            total_steps += 1

            # Select and execute action
            action = agent.select_action(state, epsilon)
            next_frame, reward, done = env.step(action)

            # Sub-question (a): Add reward shaping here if desired
            # For example: shaped_reward = reward + custom_bonus

            # Update frame stack
            frame_stack.append(next_frame)
            next_state = np.concatenate(frame_stack, axis=0)

            # Store experience
            agent.memory.push(state, action, reward, next_state, done)

            # Update state
            state = next_state
            episode_reward += reward

            # Sub-question (a): Modify learning frequency
            # Currently learning every step - consider different strategies
            if total_steps % LEARN_EVERY_N_STEPS == 0:
                agent.learn()

            # Render UI
            training_info = {
                "Episode": f"{episode + 1}/{NUM_EPISODES}",
                "Total Steps": f"{total_steps}",
                "Epsilon": f"{epsilon:.4f}",
                "Reward": f"{episode_reward:.1f}",
                "Loss": f"{agent.last_loss:.4f}" if agent.last_loss > 0 else "N/A",
                "Buffer": f"{len(agent.memory)}/{agent.memory.buffer.maxlen}",
            }
            if env.hit_wall:
                training_info["WARN"] = "Hit Wall!"

            env.render(training_info)

            # Update target network
            if total_steps % TARGET_UPDATE_STEPS == 0:
                agent.update_target_network()
                print(f"--- Step {total_steps}: Target Network Updated! ---")
                MODEL_PATH = f"pacman_dqn_model_{total_steps}.pth"
                torch.save(agent.policy_net.state_dict(), MODEL_PATH)
                print(f"Model saved to {MODEL_PATH}")

            if done:
                break

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        print(
            f"Episode: {episode + 1}/{NUM_EPISODES}, Score: {episode_reward:.2f}, Epsilon: {epsilon:.2f}"
        )

    # --- Training Complete ---
    end_time = time.time()
    print("--- Training Complete ---")
    print(f"Total time: {(end_time - start_time) / 60:.2f} minutes")

    # Save final model
    MODEL_PATH = "pacman_dqn_model.pth"
    torch.save(agent.policy_net.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    pygame.quit()
    return agent

# ============================================================================
# 5. Testing & Evaluation (MODIFY for sub-question c)
# ============================================================================

"""
For sub-question (c): Use this section to evaluate your agent's performance.
Run this after training to collect metrics for your report.
"""

def test_agent(model_path="pacman_dqn_model.pth", num_test_episodes=10):
    """Test the trained agent and return performance metrics."""
    # --- Initialize Test Environment and Agent ---
    test_env = PacmanGame()
    state_dim = (FRAME_STACK_SIZE, SCREEN_HEIGHT, SCREEN_WIDTH)
    action_dim = test_env.action_space
    test_agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)

    # --- Load Trained Model ---
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_agent.policy_net.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        test_agent.policy_net.to(device)
        test_agent.policy_net.eval()
        print(f"\nModel loaded successfully from {model_path}!")
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found. Please run training first.")
        pygame.quit()
        return None

    # --- Run Test Episodes ---
    test_scores = []

    print("\n--- Testing Started ---")
    running_test = True

    for episode in range(num_test_episodes):
        if not running_test:
            break

        frame = test_env.reset()
        frame_stack = deque([frame] * FRAME_STACK_SIZE, maxlen=FRAME_STACK_SIZE)
        state = np.concatenate(frame_stack, axis=0)
        episode_reward = 0
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    running_test = False

            if not running_test:
                break

            # Greedy action selection (no exploration)
            action = test_agent.select_action(state, epsilon=0.0)
            next_frame, reward, game_is_done = test_env.step(action)

            frame_stack.append(next_frame)
            next_state = np.concatenate(frame_stack, axis=0)
            state = next_state
            episode_reward += reward

            test_info = {
                "Test Episode": f"{episode + 1}/{num_test_episodes}",
                "Score": f"{episode_reward:.1f}",
                "Mode": "Testing (Greedy)",
            }
            test_env.render(test_info)

            if game_is_done:
                done = True

        if running_test:
            test_scores.append(episode_reward)
            print(
                f"Test Episode: {episode + 1}/{num_test_episodes}, Score: {episode_reward:.2f}"
            )

    # --- Print Results ---
    if test_scores:
        print("\n--- Testing Complete ---")
        print(f"Average Score: {np.mean(test_scores):.2f}")
        print(f"Best Score: {np.max(test_scores):.2f}")
        print(f"Lowest Score: {np.min(test_scores):.2f}")
        print(f"Standard Deviation: {np.std(test_scores):.2f}")
        
        results = {
            "average_score": np.mean(test_scores),
            "best_score": np.max(test_scores),
            "lowest_score": np.min(test_scores),
            "std_deviation": np.std(test_scores),
            "all_scores": test_scores
        }
    else:
        results = None

    print("Testing complete. Closing Pygame window.")
    pygame.quit()
    return results

# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Uncomment the section you want to run:
    
    # Train the agent
    print("Starting training...")
    trained_agent = train_agent()
    
    # Test the agent (uncomment after training)
    # print("Starting testing...")
    # test_results = test_agent()
    # if test_results:
    #     print(f"Final Average Score: {test_results['average_score']:.2f}") 