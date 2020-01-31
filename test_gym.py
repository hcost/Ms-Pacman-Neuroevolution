import gym
import numpy as np

"""Simple model that does cartpole"""

env = gym.make('CartPole-v1')

def runEpisode(env, paramaters):
    state = env.reset()
    env.render()
    totalReward = 0
    numSteps = 0
    done = False
    while numSteps < 3000 and not done:
        action = 0 if np.matmul(paramaters, state) < 0 else 1
        state, reward, done, info = env.step(action)
        env.render()
        totalReward += reward
        numSteps += 1
    return totalReward

paramaters = np.random.rand(4) * 2 -1

reward = runEpisode(env, paramaters)
print("Iteration: 0, Reward is: {}".format(reward))

alpha = 0.05
iter = 0
while reward != 500:
    iter += 1
    newParamaters = paramaters + (np.random.rand(4)*2 -1)*alpha
    newReward = runEpisode(env, newParamaters)
    print("Iteration: {}, Reward is: {}, Old reward is: {}, alpha is: {}".format(iter, newReward, reward, alpha))
    if newReward > reward:
        paramaters = newParamaters
        reward = newReward
        alpha -= 0.025
    else:
        alpha += .025
    if iter % 50 == 0:
        print("New Base")
        paramaters = np.random.rand(4) * 2 -1
        alpha = 0.05
        reward = runEpisode(env, paramaters)

print ("Reached 500 in {} iterations".format(iter)) if reward == 500 else print("Not this time :(")

env.close()
