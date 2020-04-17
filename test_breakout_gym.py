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

env = gym.make('Breakout-v0')



def runEpisode():
    state = env.reset()
    env.render()
    totalReward = 0
    done = False
    action = 0
    while not done:
        action = getAction(action)
        state, reward, done, info = env.step(action)
        print("Reward: {}".format(reward))
        env.render()
        totalReward += reward
        print("Cumulative Reward {}".format(totalReward))

def takeAction():
    mapping = {'n':0, 'press':1, 'r': 2, 'l':3}
    action = input("Enter action (n, press, r, l): ")
    state, reward, done, info = env.step(mapping[action])
    env.render()
    return state, reward, done, info





def getAction(lastAction = 0):
    inp = input("Enter action (n, press, r, l): ")
    mapping = {'n':0, 'press':1, 'r': 2, 'l':3}
    if inp:
        return mapping[inp]
    return lastAction

state = env.reset()
env.render()
