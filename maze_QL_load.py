import numpy as np


class MyEnvironmentSimulator():
    def __init__(self):
        self._maze = np.loadtxt('maze7x7.txt', delimiter=',', dtype='int32')
        self.reset()

    def reset(self):
        self._state = [1, 1]
        return np.array(self._state)

    def step(self, action):
        reward = 0
        if action == 0:
            self._state[0] = self._state[0] - 1
        elif action == 1:
            self._state[1] = self._state[1] + 1
        elif action == 2:
            self._state[0] = self._state[0] + 1
        else:
            self._state[1] = self._state[1] - 1
        b = self._maze[self._state[0], self._state[1]]
        if b == 0:
            reward = -1
        elif b == 1:
            reward = 0
        elif b == 2:
            reward = 1
        return np.array(self._state), reward


class MyQTable():
    def __init__(self):
        # self._Qtable = np.zeros((4, 7, 7))
        qt = np.loadtxt('Qvalue.txt')
        self._Qtable = qt.reshape(4, 7, 7)

    def get_action(self, state, epsilon):
        if epsilon > np.random.uniform(0, 1):
            next_action = np.random.choice([0, 3])
        else:
            a = np.where(self._Qtable[:, state[0], state[1]] == self._Qtable[:, state[0], state[1]].max())[0]
            next_action = np.random.choice(a)
        return next_action

    def update_Qtable(self, state, action, reward, next_state):
        gamma = 0.9
        alpha = 0.5
        next_maxQ = max(self._Qtable[:, next_state[0], next_state[1]])
        self._Qtable[action, state[0], state[1]] = (1 - alpha) * self._Qtable[action, state[0], state[1]] + alpha * (reward + gamma * next_maxQ)
        return self._Qtable


def main():
    num_episodes = 1000
    max_number_of_steps = 100
    epsilon = np.linspace(start=0.0, stop=0.0, num=num_episodes)
    env = MyEnvironmentSimulator()
    tab = MyQTable()
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for t in range(max_number_of_steps):
            action = tab.get_action(state, epsilon[episode])
            next_state, reward = env.step(action)
            q_table = tab.update_Qtable(state, action, reward, next_state)
            state = next_state
            if reward != 0:
                break
        print(f'Episode:{episode:4.0f}, Step:{t:3.0f}, R:{reward:3.0f}')
    np.savetxt('Qvalue.txt', tab._Qtable.reshape(4*7*7))

    state = [1, 1]
    maze = np.loadtxt('maze7x7.txt', delimiter=',', dtype='int32')
    for t in range(100):
        maze[state[0], state[1]] = 3
        action = np.where(tab._Qtable[:, state[0], state[1]] == tab._Qtable[:, state[0], state[1]].max())[0]
        
        print(t+1, state, action)
        if action == 0:
            state[0] = state[0] - 1
        elif action == 1:
            state[1] = state[1] + 1
        elif action == 2:
            state[0] = state[0] + 1
        else:
            state[1] = state[1] - 1
        if maze[state[0], state[1]] == 2:
            break
    print(maze)


if __name__ == "__main__":
    main()

