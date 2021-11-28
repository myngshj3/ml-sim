import gym
import numpy as np
import time


class MyQTable():
    def __init__(self, num_action):
        self._Qtable = np.random.uniform(low=-1, high=1, size=(num_digitized**4, num_action))

    def get_action(self, state, epsilon):
        if epsilon > np.random.uniform(0, 1):
            next_action = np.random.choice([0, 1])
        else:
            a = np.where(self._Qtable[state] == self._Qtable[state].max())[0]
            next_action = np.random.choice(a)
        return next_action

    def update_Qtable(self, state, action, reward, next_state):
        gamma = 0.99
        alpha = 0.5
        next_maxQ = max(self._Qtable[next_state])
        self._Qtable[state, action] = (1 - alpha) * self._Qtable[state, action] + alpha * (reward + gamma * next_maxQ)
        return self._Qtable


num_digitized = 6
def digitize_state(observation):
    p, v, a, w = observation
    d = num_digitized
    pn = np.digitize(p, np.linspace(-2.4, 2.4, d+1)[1:-1:])
    vn = np.digitize(v, np.linspace(-3.0, 3.0, d+1)[1:-1])
    an = np.digitize(a, np.linspace(-0.5, 0.5, d+1)[1:-1])
    wn = np.digitize(w, np.linspace(-2.0, 2.0, d+1)[1:-1])
    return pn + vn*d + an*d**2 + wn*d**3


def main():
    num_episodes = 1000
    max_number_of_steps = 200
    env = gym.make('CartPole-v0')
    tab = MyQTable(env.action_space.n)
    for episode in range(num_episodes):
        observation = env.reset()
        state = digitize_state(observation)
        episode_reward = 0
        for t in range(max_number_of_steps):
            action = tab.get_action(state, epsilon = 0.5 * (1/(episode + 1)))
            observation, reward, done, info = env.step(action)
            if episode % 10 == 0:
                env.render()
            if done and t < max_number_of_steps-1:
                reward -= max_number_of_steps
            next_state = digitize_state(observation)
            q_table = tab.update_Qtable(state, action, reward, next_state)
            state = next_state
            episode_reward += reward
            if done:
                break
        print(f'Episode:{episode:4.0f}, R:{episode_reward:4.0f}')
    np.savetxt('Qvalue.txt', tab._Qtable)


if __name__ == "__main__":
    main()

    
