import gymnasium as gym
import math
import pygame
import numpy as np
# Initialize Pygame
pygame.init()

# Screen dimensions
width, height = 800, 600
x_cart_scale, pendulum_len = 150, 200
cart_width, cart_height = 100, 5
y_cart = height - 100

# Set up the display
scrn = pygame.display.set_mode((width, height))

# Create the CartPole-v1 environment
env = gym.make("CartPole-v1", render_mode="rgb_array")


n_bins = 20
state_bounds = [
    [-4.8, 4.8],  # Cart position
    [-4, 4],      # Cart velocity
    [-0.418, 0.418],  # Pole angle (about 24 degrees)
    [-4, 4]       # Pole angular velocity
]

# Create bins for each state dimension
state_bins = [np.linspace(b[0], b[1], n_bins) for b in state_bounds]


def discretize_state(state, state_bins):
    """Discretizes the continuous state into a tuple of bin indices."""
    return tuple(np.digitize(s, bins) for s, bins in zip(state, state_bins))


# Initialize Q-table
n_actions = env.action_space.n
q_table = np.zeros([n_bins] * len(state_bounds) + [n_actions])

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995  # Epsilon decay
min_epsilon = 0.01

# Q-learning algorithm
n_episodes = 10000
max_steps = 500

for episode in range(n_episodes):
    state, _ = env.reset()
    state = discretize_state(state, state_bins)
    done = False
    total_reward = 0

    for step in range(max_steps):
        scrn.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Choose action using ε-greedy policy

        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit

        # Take action and observe the next state and reward
        next_state, reward, done, _, _ = env.step(action)

        x_cart = int(next_state[0] * x_cart_scale) + \
            width // 2  # Scale and centre the cart
        ang = next_state[2]

        # Calculate pendulum position
        x_p = x_cart + pendulum_len * math.sin(ang)
        y_p = y_cart - pendulum_len * math.cos(ang)

        # Draw the pendulum and cart
        pygame.draw.line(scrn, (0, 0, 255), (x_p, y_p), (x_cart, y_cart), 2)
        pygame.draw.circle(scrn, (0, 255, 0), (int(x_p), int(y_p)), 15)
        pygame.draw.rect(scrn, (255, 0, 0), pygame.Rect(
            x_cart - cart_width // 2, y_cart, cart_width, cart_height))

        next_state = discretize_state(next_state, state_bins)

        pygame.display.update()

        # Update Q-value
        q_table[state][action] += alpha * \
            (reward + gamma *
             np.max(q_table[next_state]) - q_table[state][action])

        state = next_state
        total_reward += reward

        if done:
            break
        if episode > 8000:
            pygame.time.wait(50)

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if episode % 100 == 0:
        print(f"Episode {episode}: Total Reward: {total_reward}")

env.close()
pygame.quit()
