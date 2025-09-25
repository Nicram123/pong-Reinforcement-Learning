# train.py
import numpy as np
import random
from collections import deque
import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from neural_network import PongEnv, build_q_network

# --- Hyperparametry ---
ALPHA = 0.001
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.05
BATCH_SIZE = 64
MEMORY_SIZE = 50000
NUM_EPISODES = 5000 # 1000 # 5000
MAX_STEPS = 500 # 500

# --- Inicjalizacja ---
env = PongEnv()
state_size = 6
num_actions = 3

q_network = build_q_network(state_size, num_actions) # caly czas aktualizowana 
target_q_network = build_q_network(state_size, num_actions) # aktualizowana co 500 timestempow i wylicza przy okazji y_target

optimizer = Adam(learning_rate=ALPHA)
q_network.compile(optimizer=optimizer, loss="mse")
target_q_network.compile(optimizer=optimizer, loss="mse")


target_q_network.set_weights(q_network.get_weights())

memory = deque(maxlen=MEMORY_SIZE)

# return 0 - stay lub 1 - up lub 2 - down  
def get_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(num_actions)
    q_values = q_network(np.expand_dims(state, axis=0))
    return np.argmax(q_values[0].numpy())

def replay():
    if len(memory) < BATCH_SIZE:
        return
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
    q_network.train_on_batch(np.array(states), np.array(targets))

# --- Trening ---
start = time.time()
epsilon = EPSILON
for episode in range(NUM_EPISODES):
    state = env.reset() # poczatkowy stan (ball_x, ball_y, ball_dx, ball_dy, r_paddle, l_paddle) czyli po resecie wyzerowane prawie wszystko 
    total_reward = 0 # po 1 epizodzie total reward 
    step_count = 0 # 
    for step in range(MAX_STEPS): # timestemp zeby ograniczyc trwanie epizodu 
        action = get_action(state, epsilon)
        next_state, reward, done = env.step(action)
        # mały minus za każdy krok (opcjonalne)
        #reward -= 0.01
        memory.append((state, action, reward, next_state, done))
        if step_count % 5 == 0:
            replay()
        step_count += 1
        if step_count % 500 == 0:
            target_q_network.set_weights(q_network.get_weights())
        state = next_state
        total_reward += reward
        if done:
            break
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    #if (episode+1) % 50 == 0:
    #    target_q_network.set_weights(q_network.get_weights())
    #    print(f"Episode {episode+1}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
    if (episode+1) % 100 == 0:
        avg_reward = np.mean([m[2] for m in memory][-100:])
        print(f"Episode {episode+1}, Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {epsilon:.2f}")
        q_network.save(f"pong_ai_ep{episode+1}.h5")
        print(f" Zapisano checkpoint: pong_ai_ep{episode+1}.h5")

q_network.save("pong_ai.h5")
print("✅ Model zapisany jako pong_ai.h5")
print(f"Całkowity czas: {time.time()-start:.2f}s")