import gym
import numpy as np
import tensorflow as tf
from statistics import median as med
from statistics import variance as var
from statistics import mean
import datetime
from time import process_time
import pandas as pd
wkday = datetime.datetime.today().strftime('%A')

env = gym.make('MsPacman-v0')



def runEpisode():
    state = env.reset()
    env.render()
    totalReward = 0
    done = False
    action = 0
    while not done:
        print("\nAction number {}".format(len(data)))
        action = getAction(action)
        data.append((clean_state, action))
        state, reward, done, info = env.step(action)
        print("Reward: {}".format(reward))
        env.render()
        totalReward += reward
        print("Cumulative Reward {}".format(totalReward))

def takeAction():
    mapping = {'n':0, 'u':1, 'd': 4, 'l':3, 'r':2, 'ul':6, 'ur':5, 'dl': 8, 'dr':7}
    action = input("Enter action (n, u, d, l, r, ul, ur, dl, dr): ")
    state, reward, done, info = env.step(mapping[action])
    env.render()
    return state, reward, done, info





def getAction(lastAction = 0):
    inp = input("Enter action (n, u, d, l, r, ul, ur, dl, dr): ")
    mapping = {'n':0, 'u':1, 'd': 4, 'l':3, 'r':2, 'ul':6, 'ur':5, 'dl': 8, 'dr':7}
    if inp:
        return mapping[inp]
    return lastAction

state = env.reset()
env.render()
