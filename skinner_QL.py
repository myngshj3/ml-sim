import numpy as np


class MyEnvironmentSimulator():

    def __init__(self):
        self.reset()

    def reset(self):
        self._state = 0
        return self._state

    def step(self, action):
        reward = 0
        if self._state == 0:
            if action == 0:
                self._state = 1
            else:
                self._state = 0
        else:
            if action == 0:
                self._state = 0
            else:
                self._state = 1
                reward = 1
        return self._state, reward


class MyQTable():

    def __init__(self):
        self._Qtable = np.zeros((2, 2))

    def get_action(self, state, epsilon):
        if epsilon > np.random.uniform(0, 1):
            next_action = np.random.choice([0, 1])
        else:
            a = np.where(self._Qtable[state] == self._Qtable[state].max())[0]
            next_action = np.random.choice(a)
        return next_action


    def update_Qtable(self, state, action, reward, next_state):
        gamma = 0.9
        alpha = 0.5
        next_maxQ = max(self._Qtable[next_state])
        self._Qtable[state, action] = (1 - alpha) * self._Qtable[state, action] + alpha * (reward + gamma * next_maxQ)
        return self._Qtable


def main():
    num_episodes = 10
    max_number_of_steps = 5
    epsilon = np.linspace(start=1.0, stop=0.0, num=num_episodes)
    env = MyEnvironmentSimulator()
    tab = MyQTable()

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for t in range(max_number_of_steps):
            action = tab.get_action(state, epsilon[episode])
            next_state, reward = env.step(action)
            print(state, action, reward)
            q_table = tab.update_Qtable(state, action, reward, next_state)
            state = next_state
            episode_reward += reward
        print(f'Episode:{episode+1:4.0f}, R:{episode_reward:3.0f}')
        print(q_table)
    np.savetxt('Qvalue.txt', tab._Qtable)


if __name__ == "__main__":
    main()

