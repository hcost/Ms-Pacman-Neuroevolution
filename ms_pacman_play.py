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

def cleanState(state):
    return np.reshape(state, [1, 210, 160, 3])

def runEpisode(env, games=1):
    data = []
    while games:
        state = env.reset()
        clean_state = cleanState(state)
        env.render()
        totalReward = 0
        numSteps = 0
        done = False
        action = 0
        while not done:
            print("\nAction number {}".format(len(data)))
            action = getAction(action)
            data.append((clean_state, action))
            state, reward, done, info = env.step(action)
            print("Reward: {}".format(reward))
            clean_state = cleanState(state)
            env.render()
            totalReward += reward
            numSteps += 1
        games -= 1
    arr = np.array(data)
    return arr


def make_data(fp, games):
    filepath = "/Users/harrison/gamerAI/data/"+fp+".csv"
    pd.DataFrame(runEpisode(env, games)).to_csv(filepath)


def getAction(lastAction = 0):
    inp = input("Enter action (n, u, d, l, r, ul, ur, dl, dr): ")
    mapping = {'n':0, 'u':1, 'd': 4, 'l':3, 'r':2, 'ul':6, 'ur':5, 'dl': 8, 'dr':7}
    if inp:
        return mapping[inp]
    return lastAction


fp = input("Enter .csv filename: ")
games = int(input("How many games would you like to play? "))
make_data(fp, games)
