import gym
import numpy as np
import tensorflow as tf
from statistics import median as med
from time import sleep

"""Simple model that does cartpole"""

env = gym.make('CartPole-v1')


def build_model(me_llamo_es):
    model = tf.keras.models.Sequential(name=me_llamo_es)
    model.add(tf.keras.layers.Dense(4, input_shape=(4,), activation='relu', name="input")) #Input layer
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid', name="output")) #Output layer
    return model

def cleanState(state):
    return np.reshape(state, [1, 4])

def homo_erectus(num):
    population = []
    total = num
    while num:
        population.append(build_model("cave_man_{}".format(total - num)))
        num -= 1
    return population


def runEpisode(env, person):
    state = env.reset()
    env.render()
    totalReward = 0
    numSteps = 0
    done = False
    while numSteps < 3000 and not done:
        action = person.predict(cleanState(state))
        state, reward, done, info = env.step(int(round(action[0][0])))
        env.render()
        totalReward += reward
        numSteps += 1
    return totalReward



def run_generation(population, attempts):
    performance = {}
    for person in population:
        rewards = []
        for i in range(attempts):
            rewards.append(runEpisode(env, person))
        performance[person] = med(rewards)
    return performance


def the_fittest(population, survivors):
    the_fit = []
    while survivors:
        best = max(population, key=population.get)
        the_fit.append(best)
        del population[best]
        survivors -= 1
    return the_fit

def survival_of_the_fittest(population, individuals, survivors, gen):
    parents = the_fittest(population, survivors)
    children = mating_season(parents, individuals, survivors, gen)
    return children

def mating_season(parents, individuals, survivors, gen):
    num_kids = int((individuals - survivors)/survivors)
    gene_pool = []
    kid_num = 0
    for parent in parents:
        gene_pool.append(parent)
        parent_dna = parent.get_weights()
        for _ in range(num_kids):
            prelim_kid = make_child(parent_dna, gen)
            kid = birth(prelim_kid, gen, kid_num)
            gene_pool.append(kid)
            kid_num += 1
    return gene_pool

def make_child(dna, gen):
    #Takes in model weights and slightly modifies it, returning new weights
    max_change = int(10000/(1.5*gen))
    for layer in dna:
        for theta in layer:
            if np.random.randint(0,2) and type(theta) == np.ndarray:
                alterations = np.random.randint(0, len(theta))
                for _ in range(alterations):
                    index = np.random.randint(0, len(theta))
                    change = np.random.randint(0, max_change)/10000
                    if np.random.randint(0,2):
                        theta[index] += change
                    else:
                        theta[index] -= change
    return dna

def birth(dna, gen, num):
    #creates model with given weights
    model = build_model("gen_{}_num_{}".format(gen, num))
    model.set_weights(dna)
    return model





def evolve(generations=20, individuals=10, attempts=10, survivors=2):
    population = homo_erectus(individuals)
    for gen in range(1, generations+1):
        print("Generation {}".format(gen))
        population = run_generation(population, attempts)
        population = survival_of_the_fittest(population, individuals, survivors, gen)
    population = run_generation(population, attempts)
    return the_fittest(population, 1)[0]


the_champion = evolve()
print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("Now presenting your CHAMPION!")
print("\n")
print(the_champion.summary())
