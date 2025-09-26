# train.py
import numpy as np
import random
from collections import deque
import time
from tensorflow.keras.optimizers import Adam
from neural_network import PongEnv, build_q_network

# --- Hyperparametry ---
ALPHA = 0.001
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.05
BATCH_SIZE = 64
MEMORY_SIZE = 50000
NUM_EPISODES = 5000
MAX_STEPS = 500

# --- Inicjalizacja ---
env = PongEnv()
state_size = 8  # <- jeśli masz 6 elementów stanu: (x, y, dx, dy, r_paddle, l_paddle)
num_actions = 3

q_network = build_q_network(state_size, num_actions)
target_q_network = build_q_network(state_size, num_actions)

optimizer = Adam(learning_rate=ALPHA)
q_network.compile(optimizer=optimizer, loss="mse")
target_q_network.compile(optimizer=optimizer, loss="mse")

target_q_network.set_weights(q_network.get_weights())
memory = deque(maxlen=MEMORY_SIZE)


# --- epsilon-greedy ---
def get_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(num_actions)
    q_values = q_network(np.expand_dims(state, axis=0))
    return np.argmax(q_values[0].numpy())


# --- replay memory ---
def replay():
    if len(memory) < BATCH_SIZE:
        return None
    batch = random.sample(memory, BATCH_SIZE)
    states, targets = [], []
    for state, action, reward, next_state, done in batch:
        target = q_network(np.expand_dims(state, axis=0))[0].numpy()
        if done:
            target[action] = reward
        else:
            q_future = np.max(target_q_network(np.expand_dims(next_state, axis=0))[0].numpy())
            target[action] = reward + GAMMA * q_future
        states.append(state)
        targets.append(target)
    loss = q_network.train_on_batch(np.array(states), np.array(targets))
    return loss


# --- Trening ---
start = time.time()
epsilon = EPSILON
rewards_history = []
losses_history = []

for episode in range(NUM_EPISODES):
    state = env.reset()
    total_reward = 0
    step_count = 0
    episode_losses = []

    for step in range(MAX_STEPS):
        action = get_action(state, epsilon)
        next_state, reward, done = env.step(action)

        memory.append((state, action, reward, next_state, done))

        if step_count % 5 == 0:
            loss = replay()
            if loss is not None:
                episode_losses.append(loss)

        step_count += 1
        if step_count % 100 == 0:
            target_q_network.set_weights(q_network.get_weights())

        state = next_state
        total_reward += reward
        if done:
            break

    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    # --- logowanie ---
    rewards_history.append(total_reward)
    if episode_losses:
        losses_history.append(np.mean(episode_losses))

    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(rewards_history[-100:])
        avg_loss = np.mean(losses_history[-100:]) if losses_history else 0
        print(f"Episode {episode+1}, Avg Reward (last 100): {avg_reward:.2f}, "
              f"Avg Loss (last 100): {avg_loss:.4f}, Epsilon: {epsilon:.2f}", 
              f"Total Reward: {total_reward:.2f}")

        q_network.save(f" pong_ai_ep{episode+1}.keras")
        print(f" Zapisano checkpoint: pong_ai_ep{episode+1}.keras")

q_network.save("pong_ai.keras")
print(" Model zapisany jako pong_ai.keras")
print(f" Całkowity czas: {time.time()-start:.2f}s")
