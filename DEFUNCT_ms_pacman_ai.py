import gym
import numpy as np
import tensorflow as tf
from collections import defaultdict
"""Ms. Pacman AI"""

env = gym.make('MsPacman-ram-v0')


env.reset()

# class DiscreteState(gym.ObservationWrapper):
#     "Converts state to integer"
#
#     def __init__(self, env, n_bins=10, low=None, high=None):
#         super().__init__(env)
#         assert isinstance(env.observation_space, gym.spaces.Box)
#
#         low = self.observation_space.low if low is None else low
#         high = self.observation_space.high if high is None else high
#
#         self.n_bins = n_bins
#         self.val_bins = [np.linspace(l, h, n_bins + 1) for l, h in zip(low.flatten(), high.flatten())]
#
#         self.observation_space = gym.spaces.Discrete(n_bins ** low.flatten().shape[0])
#
#     def _convert_to_one_number(self, digits):
#         return sum([d * ((self.n_bins + 1) ** i) for i, d in enumerate(digits)])
#
#     def observation(self, observation):
#         digits = [np.digitize([x], bins)[0]
#                   for x, bins in zip(observation.flatten(), self.val_bins)]
#         return self._convert_to_one_number(digits)

# print("init?")
# env = DiscreteState(env, 3, [-2.4, -2.0, -0.42, -3.5], [2.4, 2.0, 0.42, 3.5])
# print("init!")

Q = defaultdict(float)
actions = range(9)

gamma = 0.99
alpha = 0.5
epsilon = 0.2
steps = 10000000000

def update_Q(state, reward, action, next_state, done):
    max_q_next = max([Q[next_state, a] for a in actions])

    Q[state, action] += alpha * (reward+ gamma * max_q_next * (1.0 - done) - Q[state, action])





def getAction(state):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    qvals = {a: Q[state, a] for a in actions}
    max_q = max(qvals.values())

    actions_with_max_q = [a for a, q in qvals.items() if q == max_q]
    return np.random.choice(actions_with_max_q)



state = env.reset()
state = state.tobytes()

env.render()


rewards = []
reward = 0.0
iteration = 1


for step in range(steps):
    action = getAction(state)
    next_state, currReward, done, info = env.step(action)
    next_state = next_state.tobytes()
    env.render()
    update_Q(state, currReward, action, next_state, done)
    reward += currReward
    if done:
        rewards.append(reward)
        reward = 0.0
        avg = sum(rewards)/len(rewards)
        print("\n----------------------------------------------\n")
        print("•There have been {} iterations\n•Step Number: {}".format(iteration, step))
        print("Average Reward is: {}".format(avg))
        if len(rewards) > 5:
            run_avg = sum(rewards[len(rewards)-5:])/5
            print("•Running Reward Average is: {}".format(run_avg))
            print("•Recent attempts are averaging {} increased reward".format(run_avg-avg))
        state = env.reset()
        state = state.tobytes()
        iteration += 1
    else:
        state = next_state


env.close()
