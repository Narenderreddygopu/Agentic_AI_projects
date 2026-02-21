import numpy as np

# Simple 1D grid world: states 0..4, goal at 4
# actions: 0=left, 1=right
n_states = 5
n_actions = 2
goal = 4

Q = np.zeros((n_states, n_actions))
alpha = 0.1
gamma = 0.9
eps = 0.2

def step(state, action):
    next_state = state + (1 if action == 1 else -1)
    next_state = max(0, min(goal, next_state))
    reward = 1 if next_state == goal else 0
    done = (next_state == goal)
    return next_state, reward, done

for episode in range(500):
    s = 0
    while True:
        # epsilon-greedy
        if np.random.rand() < eps:
            a = np.random.randint(n_actions)
        else:
            a = np.argmax(Q[s])

        ns, r, done = step(s, a)

        # Q-learning update
        Q[s, a] = Q[s, a] + alpha * (r + gamma * np.max(Q[ns]) - Q[s, a])

        s = ns
        if done:
            break

print("Learned Q-table:\n", Q)
print("Best policy (0=left,1=right):", np.argmax(Q, axis=1))