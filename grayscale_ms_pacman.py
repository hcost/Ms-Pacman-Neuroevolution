import gym
import numpy as np
import tensorflow as tf
from statistics import median as med
from statistics import variance as var
from statistics import mean
import datetime
from time import process_time
from PIL import Image

"""2D CNN neuroevolution for Ms Pacman, now with less colors!"""

def build_model(name):
    model = tf.keras.models.Sequential(name=name)
    model.add(tf.keras.layers.Conv2D(filters=7, kernel_size=3, strides=1, input_shape=(210, 160, 1), activation='relu', name="input")) #Input layer
    model.add(tf.keras.layers.BatchNormalization(name="norm"))
    model.add(tf.keras.layers.Conv2D(filters=9, kernel_size=5, strides=2, activation='relu', name='conv_2'))
    model.add(tf.keras.layers.Conv2D(filters=13, kernel_size=7, strides=3, activation='relu', name='conv_3'))
    model.add(tf.keras.layers.Flatten(name='flat'))
    model.add(tf.keras.layers.Dense(25, activation='sigmoid', name='dense1'))
    model.add(tf.keras.layers.Dense(5, activation='softmax', name="output")) #Output layer
    return model

def cleanState(state):
    state = Image.fromarray(state)
    state = state.convert('L')
    state = np.asarray(state)
    return np.reshape(state, [1, 210, 160])

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
    if render:
        env.render()
    totalReward = 0
    numSteps = 0
    done = False
    while numSteps < 5000 and not done:
        action = person.predict(cleanState(state))
        action = get_action(action)
        state, reward, done, info = env.step(action)
        reward = 0.25 * reward if reward > 10 else reward
        if render:
            env.render()
        totalReward += reward
        numSteps += 1
    print(totalReward)
    return totalReward

def run_generation(population, gen, render):
    performance = {}
    attempts = params["attempts"]
    count = 0
    for person in population:
        count += 1
        print("\nIndividual {} out of {} ({}%)\n{}\nScores:".format(count, len(population), int(count/len(population)*100), person.name))
        rewards = []
        for i in range(attempts):
            rewards.append(runEpisode(env, person, render))
        performance[person] = rewards
    return performance

def the_fittest(population):
    still_alive = []
    survivors = params['survivors']
    while survivors:
        best = max(population, key = lambda x: max(population[x]))
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
    return children

def mutate(junior):
    #mutate conv layers
    odds = params['odds']
    severity = params["severity"]
    extreme_odds = params["extreme_odds"]
    for layer in params["conv_layers"]:
        weights = junior.get_layer('layer').get_weights()
        bias = weights[1]
        weights = weights[0]
        rng1, rng2, rng3, rng4 = weights.shape
        for i in range(rng1):
            for j in range(rng2):
                for k in range(rng3):
                    for l in range(0, rng4, np.randint(3)):
                        if event(extreme_odds):
                            weights[i][j][k][l] += (-1)**coin_flip * 1/np.random.randint(11)
                        elif event(odds):
                            weights[i][j][k][l] += (-1)**coin_flip * np.random.uniform() * severity
        for i in range(len(bias)):
            if event(odds):
                bias[i] += (-1)**coin_flip * np.random.uniform() * severity
        weights = [weights, bias]
        junior.get_layer('layer').set_weights(weights)
    for layer in params["flat_layers"]:
        weights = junior.get_layer('layer').get_weights()
        bias = weights[1]
        weights = weights[0]
        rng1, rng2 = weights.shape
        for i in range(0, rng1, np.randint(rng1//5)):
            for j in range(rng2):
                if event(extreme_odds):
                    weights[i][j] += (-1)**coin_flip * 1/np.random.randint(11)
                elif event(odds):
                    weights[i][j] += (-1)**coin_flip * np.random.uniform() * severity
        for i in range(len(bias)):
            if event(odds):
                bias[i] += (-1)**coin_flip * np.random.uniform() * severity
        weights = [weights, bias]
        junior.get_layer('layer').set_weights(weights)











params = {
    #hyperparameters
    #need individuals-survivors)/survivors to be an int
    "conv_layers" : ["input", "conv_2", "conv_3"],
    "flat_layers" : ["dense1", "output"],
    "generations" : 200,
    "individuals" : 12,
    "attempts" : 1,
    "survivors" : 2,
    "odds" : 65, #percent chance of a mutation
    "severity" : 1e-1, #scales mutations; if set to 1 mutations are uniform from [0,1]
    "extreme_odds" : 5 #percent chance of an extreme mutation
}

#evolution algs
def evolve(render=False):
    "Starts a new family tree <3"
    try:
        population = generate_inital_pop()
        for gen in range(0, params["generations"]+1):
            print("\n•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••\nRunning Generation {}, {}% done\n•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••\n".format(gen, int(gen/generations*100)))
            population = run_generation(population, gen, render)
            print("\n•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••\nCreating Generation {}\n•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••".format(gen+1))
            population = survival_of_the_fittest(population, gen+1)
            if (gen+1) % 10 == 0:
                save(population, "checkpoint_{}".format(gen+1))
        population = run_generation(population, gen, render)
        return the_fittest(population, 1)[0], population
    except KeyboardInterrupt:
        return None, population

def cont(population, curr_generations, render=False):
    "Continues a family tree <3"
    try:
        for gen in range(curr_generations, params["generations"]+1):
            print("\n•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••\nRunning Generation {}, {}% done\n•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••\n".format(gen, int(gen/generations*100)))
            population = run_generation(population, gen, render)
            print("\n•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••\nCreating Generation {}\n•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••".format(gen+1))
            population = survival_of_the_fittest(population, gen+1)
            if (gen+1) % 10 == 0:
                save(population, "checkpoint_{}".format(gen+1))
        population = run_generation(population, gen, render)
        return the_fittest(population, 1)[0], population
    except KeyboardInterrupt:
        return None, population







#utility functions
def save(models, file_name):
    #Dont use a try block like that fix
    fp = "/Users/harrison/gamerAI/models"
    event = "/"+file_name+"/"
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
        model_list.append(tf.keras.models.load_model("/Users/harrison/gamerAI/models/"+file_name+"/model.tf"))
    else:
        for i in range(models):
            model_list.append(tf.keras.models.load_model("/Users/harrison/gamerAI/models/"+file_name+"/model_{}.tf".format(i)))
    return model_list

def coin_flip():
    return np.random.randint(2)

def event(thresh):
    return np.random.randint(101) < thresh

def copy_model(model_source, model_target, certain_layer=""):
    for target_layers, source_layers in zip(model_target.layers, model_source.layers):
        weights = source_layers.get_weights()
        target_layers.set_weights(weights)
        if target_layers.name == certain_layer:
            break





# main
env = gym.make('MsPacman-v0')
wkday = datetime.datetime.today().strftime('%A')
purpose = input("\nHello Harrison. Hope you're having a nice {}.\nAre you starting a new instance (n), continuing an old one (c), or debugging (d)?\n".format(wkday))
render = input("\nWould you like to see the action? (y or n)\n")
render = True if render=="y" else False
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
