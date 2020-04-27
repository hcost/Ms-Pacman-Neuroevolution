import gym
import numpy as np
import tensorflow as tf
from statistics import median as med
from statistics import variance as var
from statistics import mean
import datetime
from time import process_time
from PIL import Image
from collections import Counter

"""2D CNN neuroevolution for Ms Pacman, now with less colors!"""


def build_model(name="bob"):
    #bigger model size--gen116 architecture
    # model = tf.keras.models.Sequential(name=name)
    # model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=3, strides=1, input_shape=(210, 160, 4), activation='relu', name="input")) #Input layer
    # model.add(tf.keras.layers.Conv2D(filters=9, kernel_size=3, strides=1, activation='relu', name='conv_2'))
    # model.add(tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=5, activation='relu', name='conv_3'))
    # model.add(tf.keras.layers.Flatten(name='flat'))
    # model.add(tf.keras.layers.Dense(10, activation='sigmoid', name='dense1'))
    # model.add(tf.keras.layers.Dense(4, activation='softmax', name="output")) #Output layer


    #smaller arch size--experimental/current
    model = tf.keras.models.Sequential(name=name)
    model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=3, strides=1, input_shape=(210, 160, 4), activation='relu', name="input")) #Input layer
    model.add(tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=5, activation='relu', name='conv_2'))
    model.add(tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=3, activation='relu', name='conv_3'))
    model.add(tf.keras.layers.Flatten(name='flat'))
    model.add(tf.keras.layers.Dense(10, activation='sigmoid', name='dense1'))
    model.add(tf.keras.layers.Dense(4, activation='softmax', name="output")) #Output layer
    return model


def cleanStates(states):
    bigState = np.dstack((states[0], states[1], states[2], states[3]))
    cleaned = []
    for i in range(4):
        state = states[i]
        state = Image.fromarray(state)
        state = state.convert('L')
        state = np.asarray(state)
        state = np.reshape(state, [1, 210, 160])
        cleaned.append(state)
    bigState = np.stack((cleaned[0], cleaned[1], cleaned[2], cleaned[3]), axis=-1)
    return bigState


def generate_inital_pop():
    num = params['individuals']
    population = []
    total = num
    while num:
        population.append(build_model("cave_man_{}".format(total - num)))
        num -= 1
    return population

def get_action(action):
    #{'n':0, 'u':1, 'd': 4, 'l':3, 'r':2, 'ul':6, 'ur':5, 'dl': 8, 'dr':7}
    try:
        action = action[0]
        action = np.where(action == np.amax(action))
        return max(action, key = lambda x: x[0])[0]
    except:
        print("Error getting next action")
        print(action)
        return null

def runEpisode(env, person, render):
    state = env.reset()
    states = []
    states.append(state)
    emphasis = params["emphasis"]
    for i in range(2):
        state, reward, done, info = env.step(0)
        states.append(state)
    state, reward, done, info = env.step(1)
    states.append(state)
    if render:
        env.render()
    totalReward = 0
    numSteps = 0
    score = 0
    done = False
    prev_lives = info['ale.lives']
    actions = []
    while numSteps < 5000 and not done:
        action = person.predict(cleanStates(states))
        action = get_action(action)
        states = []
        actions.append(action)
        #death reset
        if info['ale.lives'] != prev_lives or numSteps % 100 == 0:
            prev_lives = info['ale.lives']
            state, reward, done, info = env.step(1)
        else:
            prev_lives = info['ale.lives']
        for i in range(4):
            state, reward, done, info = env.step(action)
            states.append(state)
            if render:
                env.render()
            score += reward
            totalReward += reward*emphasis**score
        numSteps += 1
    tally = Counter(actions)
    zero, two, three = tally.get(0, 0), tally.get(2, 0), tally.get(3, 0)
    variety = -1*max(abs(zero-two), abs(two-three), abs(three-zero))
    netReward = -1000 if score < 2 else totalReward + variety*params["alpha"]
    print("Score: {}, Reward {}".format(score, netReward))
    return netReward

def run_generation(population, gen, render):
    performance = {}
    attempts = params["attempts"]
    count = 0
    avgReward = []
    for person in population:
        count += 1
        print("\nIndividual {} out of {} ({}%)\n{}".format(count, len(population), int(count/len(population)*100), person.name))
        rewards = []
        for i in range(attempts):
            rewards.append(runEpisode(env, person, render))
        reward = sum(rewards)/attempts
        if attempts > 1:
            print("Average reward of {}".format(reward))
        performance[person] = reward
        avgReward.append(reward)
    avgReward = sum(avgReward)/len(avgReward)
    print("\nAverage reward for generation {} was {}".format(gen, avgReward))
    return performance

def the_fittest(population):
    still_alive = []
    survivors = params['survivors']
    while survivors:
        best = max(population, key = lambda x: population[x])
        still_alive.append(best)
        del population[best]
        survivors -= 1
    return still_alive

def survival_of_the_fittest(population, gen):
    parents = the_fittest(population)
    children = mating_season(parents, gen)
    pop = parents + children
    return pop

def mating_season(parents, gen):
    count = 0
    children = []
    individuals = params['individuals']
    survivors = params['survivors']
    for parent in parents:
        for i in range(int((individuals-survivors)/survivors)):
            junior = build_model("gen_{}_ind_{}".format(gen, count))
            copy_model(parent, junior)
            mutate(junior)
            children.append(junior)
            count += 1
    return children

def mutate(junior):
    #mutate conv layers
    odds = params['odds']
    severity = params["severity"]
    extreme_odds = params["extreme_odds"]
    if event(extreme_odds):
        severity = params["extreme_severity"]
    for layer in params["conv_layers"]:
        weights = junior.get_layer(layer).get_weights()
        bias = weights[1]
        weights = weights[0]
        rng1, rng2, rng3, rng4 = weights.shape
        for i in range(rng1):
            for j in range(rng2):
                for k in range(rng3):
                    for l in range(rng4):
                        if event(odds):
                            weights[i][j][k][l] += (-1)**coin_flip() * np.random.uniform() * severity
        for i in range(len(bias)):
            if event(odds):
                bias[i] += (-1)**coin_flip() * np.random.uniform() * severity
        weights = [weights, bias]
        junior.get_layer(layer).set_weights(weights)
    for layer in params["flat_layers"]:
        weights = junior.get_layer(layer).get_weights()
        bias = weights[1]
        weights = weights[0]
        rng1, rng2 = weights.shape
        for i in range(rng1):
            for j in range(rng2):
                if event(odds):
                    weights[i][j] += (-1)**coin_flip() * np.random.uniform() * severity
        for i in range(len(bias)):
            if event(odds):
                bias[i] += (-1)**coin_flip() * np.random.uniform() * severity
        weights = [weights, bias]
        junior.get_layer(layer).set_weights(weights)











params = {
    #hyperparameters
    #need individuals-survivors)/survivors to be an int
    "conv_layers" : ["input", "conv_2", "conv_3"],
    "flat_layers" : ["dense1", "output"],
    "generations" : 3000,
    "individuals" : 12,
    "attempts" : 5,
    "survivors" : 3,
    "alpha" : .5, #modifier on variety added to reward after run; 0 to use default reward
    "emphasis" : 3.5, #modifier on actual score
    "emphasis_threshold" : 3,#when to start applying emphasis
    "odds" : 50, #percent chance of a mutation
    "severity" : 1e-3, #scales mutations; if set to 1 mutations are uniform from [-1,1]
    "extreme_odds" : 10, #percent chance of an extreme mutation
    "extreme_severity" : 1e-1, #severity if an extreme mutation occurs
    "switch" : None, #when to switch to new params
    "switched" : ["individuals", "attempts", "survivors", "alpha", "emphasis", "emphasis_threshold", "odds", "severity", "extreme_odds", "extreme_severity"] #params to switch
}

#params for later generations
switch = {
    "individuals" : 12,
    "attempts" : 3,
    "survivors" : 3,
    "alpha" : 1, #modifier on variety added to reward after run; 0 to use default reward
    "emphasis" : 5, #modifier on score
    "emphasis_threshold" : 3, #when to start applying emphasis
    "odds" : 60, #percent chance of a mutation
    "severity" : 1e-3, #scales mutations; if set to 1 mutations are uniform from [-1,1]
    "extreme_odds" : 7, #percent chance of an extreme mutation
    "extreme_severity" : 1e-1 #severity if an extreme mutation occurs
}



#evolution algs
def evolve(render=False):
    "Starts a new family tree <3"
    try:
        population = generate_inital_pop()
        for gen in range(0, params["generations"]+1):
            print("\n•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••\nRunning Generation {}, {}% done\n•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••\n".format(gen, int(gen/params['generations']*100)))
            population = run_generation(population, gen, render)
            print("\n•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••\nCreating Generation {}\n•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••".format(gen+1))
            population = survival_of_the_fittest(population, gen+1)
            if (gen+1) % 10 == 0:
                save(population, "checkpoint_{}".format(gen+1))
            if gen == params["switch"]:
                for param in params["switched"]:
                    params[param] = switch[param]
        population = run_generation(population, gen, render)
        return the_fittest(population)[0], population
    except KeyboardInterrupt:
        return None, population

def cont(population, curr_generations, render=False):
    "Continues a family tree <3"
    try:
        for gen in range(curr_generations, params["generations"]+1):
            print("\n•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••\nRunning Generation {}, {}% done\n•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••\n".format(gen, int(gen/params['generations']*100)))
            population = run_generation(population, gen, render)
            print("\n•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••\nCreating Generation {}\n•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••".format(gen+1))
            population = survival_of_the_fittest(population, gen+1)
            if (gen+1) % 10 == 0:
                save(population, "checkpoint_{}".format(gen+1))
        population = run_generation(population, gen, render)
        return the_fittest(population)[0], population
    except KeyboardInterrupt:
        return None, population







#utility functions
def save(models, file_name):
    #Dont use a try block like that fix
    fp = "/content/gdrive/My Drive/gamerAI/breakout_models/"
    event = file_name+"/"
    fp += event
    try:
        i = 0
        for model in models:
            fp_sp = fp+"model_"+str(i)+".tf"
            tf.keras.models.save_model(model, fp_sp)
            i+=1
    except:
        fp += "model.tf"
        tf.keras.models.save_model(models, fp)

def load(file_name, models=1):
    model_list = []
    if models == 1:
        model_list.append(tf.keras.models.load_model("/content/gdrive/My Drive/gamerAI/breakout_models/"+file_name+"/model.tf"))
    else:
        for i in range(models):
            model_list.append(tf.keras.models.load_model("/content/gdrive/My Drive/gamerAI/breakout_models/"+file_name+"/model_{}.tf".format(i)))
    return model_list

def coin_flip():
    return np.random.randint(2)

def event(thresh):
    return np.random.randint(1,101) < thresh

def copy_model(model_source, model_target, certain_layer=""):
    for target_layers, source_layers in zip(model_target.layers, model_source.layers):
        weights = source_layers.get_weights()
        target_layers.set_weights(weights)
        if target_layers.name == certain_layer:
            break





# main
env = gym.make('Breakout-v0')
wkday = datetime.datetime.today().strftime('%A')
purpose = input("\nHello Harrison. Hope you're having a nice {}.\nAre you starting a new instance (n), continuing an old one (c), or debugging (d)?\n".format(wkday))
render = False
if purpose == "n":
    print("\nLet there be life!\n")
    champion, population = evolve(render=render)
    if champion:
        print("Your champion is:")
        print(champion.name)
        save(population, "donezo")
    else:
        print("\nEarly termination")
        saving=input("Save models? (y or n) ")
        if saving == "y" or saving=="yes":
            fp = input("Name of Directory: ")
            save(population, fp)
elif purpose == "c":
    fp = input("\nName of directory of models: ")
    num = input("\nHow many models are being loaded? ")
    gen = int(input("\nWhat generation is this? "))
    num = int(num)
    print("\nLet there be even more life!\n")
    population = load(fp, num)
    champion, population = cont(population, gen, render=render)
    if champion:
        print("Your champion is:")
        print(champion.name)
        save(population, "woah")
    else:
        print("\nEarly termination")
        saving=input("\nSave models? (y or n) ")
        if saving == "y" or saving=="yes":
            fp = input("\nName of Directory: ")
            save(population, fp)
else:
    bob = build_model('bob')
