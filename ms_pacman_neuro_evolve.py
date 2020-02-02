import gym
import numpy as np
import tensorflow as tf
from statistics import median as med
from statistics import variance as var
from statistics import mean
import datetime
from time import process_time
wkday = datetime.datetime.today().strftime('%A')

"""2D CNN neuroevolution for Ms Pacman (hopefully)"""

env = gym.make('MsPacman-v0')

def build_model(me_llamo_es, bias=False):
    if bias:
        sign1 = 1 if coin_flip() else -1
        sign2 = 1 if coin_flip() else -1
        bias1 = tf.constant_initializer(sign1*np.random.rand()/100)
        bias2 = tf.constant_initializer(sign2*np.random.rand()/100)
        bias3 = tf.constant_initializer(sign1*np.random.rand()/100)
        bias4 = tf.constant_initializer(sign2*np.random.rand()/100)
        bias5 = tf.constant_initializer(sign1*sign2*np.random.rand()/100)
    else:
        bias1 = 'zeros'
        bias2 = 'zeros'
        bias3 = 'zeros'
        bias4 = 'zeros'
        bias5 = 'zeros'
    model = tf.keras.models.Sequential(name=me_llamo_es)
    model.add(tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=2, input_shape=(210, 160, 3), activation='relu', name="input", bias_initializer=bias1)) #Input layer
    model.add(tf.keras.layers.MaxPooling2D((2,2), name="pooling"))
    model.add(tf.keras.layers.BatchNormalization(name="normal_1"))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu', name='conv_2', bias_initializer=bias2))
    model.add(tf.keras.layers.BatchNormalization(name='normal_2'))
    model.add(tf.keras.layers.Flatten(name='flat'))
    model.add(tf.keras.layers.Dense(243, activation='relu', name='dense1', bias_initializer=bias3))
    model.add(tf.keras.layers.Dense(27, activation='relu', name='dense2', bias_initializer=bias4))
    model.add(tf.keras.layers.Dense(5, activation='sigmoid', name="output", bias_initializer=bias5)) #Output layer
    return model

def cleanState(state):
    return np.reshape(state, [1, 210, 160, 3])

def homo_erectus(num):
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
        print("Could not get laid")
        print(action)
        return null



def runEpisode(env, person, render):
    state = env.reset()
    if render:
        env.render()
    totalReward = 0
    numSteps = 0
    done = False
    #3000?
    last_actions = []
    while numSteps < 5000 and not done:
        action = person.predict(cleanState(state))
        action = get_action(action)
        last_actions.append(action)
        if len(last_actions) > 16:
            test_bool = True
            for act in last_actions[len(last_actions)-16:]:
                if act != action:
                    test_bool = False
            if test_bool:
                totalReward -= 10
        state, reward, done, info = env.step(action)
        if render:
            env.render()
        totalReward += reward
        numSteps += 1
    print(totalReward)
    return totalReward



def run_generation(population, attempts, gen, render):
    performance = {}
    count = 0
    for person in population:
        count += 1
        print("\nIndividual {} out of {} ({}%)\n{}\nScores:".format(count, len(population), int(count/len(population)*100), person.name))
        rewards = []
        # time_start = process_time()
        for i in range(attempts):
            rewards.append(runEpisode(env, person, render))
        # time_end = process_time()
        # time_elapsed = time_end - time_start
        performance[person] = rewards
    return performance


def the_fittest(population, survivors):
    still_alive = []
    switch = 0
    while survivors:
        if switch:
            best = max(population, key = lambda x: max(population[x]))
        else:
            best = max(population, key = lambda x: med(population[x]))
        still_alive.append(best)
        del population[best]
        switch = (switch + 1) % 2
        survivors -= 1
    return still_alive


def survival_of_the_fittest(population, individuals, survivors, gen, mutation_odds, alpha):
    parents = the_fittest(population, survivors)
    children = mating_season(parents, individuals, survivors, gen, mutation_odds, alpha)
    pop = parents + children
    return pop

def mating_season(parents, individuals, survivors, gen, mutation_odds, alpha):
    children = []
    num = 0
    num_kids = int((individuals-(2*survivors)-3)/survivors)
    #TODO remove assumption there are three survivors
    kid0 = mate(parents[0], parents[1], gen, num, mutable=False)
    children.append(kid0)
    num += 1
    kid01 = mate(parents[0], parents[1], gen, num, mult=1)
    children.append(kid01)
    num += 1
    kid1 = mate(parents[2], parents[0], gen, num, mult=0)
    children.append(kid1)
    num += 1
    kid2 = mate(parents[1], parents[2], gen, num, mult=1)
    children.append(kid2)
    num += 1
    kid3 = mate(parents[1], parents[0], gen, num, mult=2)
    children.append(kid3)
    num += 1
    kid4 = mate(parents[0], parents[2], gen, num, mult=1)
    children.append(kid4)
    num += 1
    kid5 = mate(parents[2], parents[1], gen, num, mult=0)
    children.append(kid5)
    num += 1
    kid6a = mate(parents[2], parents[1], gen, num, mult=2)
    children.append(kid6a)
    num += 1
    kid6b = mate(parents[0], parents[0], gen, num, mult=3.5)
    children.append(kid6b)
    num += 1
    # kid6c = mate(parents[0], build_model("rand"), gen, num, True)
    # children.append(kid6c)
    # num += 1
    # kid7 = mate(parents[1], build_model("rand"), gen, num, True)
    # children.append(kid7)
    # num += 1
    # kid8 = mate(parents[2], build_model("rand"), gen, num, True)
    # children.append(kid8)
    # num += 1
    # for parent in parents:
    #     kid9 = make_kid(parent, gen, num, mutation_odds, alpha, "same")
    #     children.append(kid9)
    #     num += 1
        # kid10 = make_kid(parent, gen, num, mutation_odds, alpha, "none")
        # children.append(kid10)
        # num += 1
        # for i in range(num_kids):
        #     children.append(make_kid(parent, gen, num, mutation_odds, alpha, "rand"))
        #     num += 1
    return children

def mate(dad, mom, gen, num, small_odds=False, mutable=True, mult=0):
    layers = ["input", "conv_2", "dense1", "dense2", "output"]
    junior = build_model("gen_{}_num_{}".format(gen, num))
    for layer in layers:
        if layer == "dense1":
            stride = np.random.randint(250, 650)
        elif layer == "conv_2":
            stride = np.random.randint(4, 16)
        else:
            stride = 1
        if layer == "input" or layer == "conv_2":
            conv = True
        else:
            conv = False
        dad_genes = dad.get_layer(layer).get_weights()
        dad_params = dad_genes[0]
        dad_bias = dad_genes[1]
        mom_genes = mom.get_layer(layer).get_weights()
        mom_params = mom_genes[0]
        mom_bias = mom_genes[1]
        new_weights = adult_wrestling(dad_params, mom_params, stride, conv, small_odds, mutable=mutable, mult=mult)
        theta = dad_bias if coin_flip() else mom_bias
        mutation = (new_weights, theta)
        junior.get_layer(layer).set_weights(mutation)
    return junior

def adult_wrestling(dad_genes, mom_genes, stride=1, conv=False, small_odds=False, mutable=True, mult=0):
    if conv:
        rng1 = len(dad_genes[0])
        rng2 = len(dad_genes[0][0])
        rng3 = len(dad_genes[0][0][0])
        for i in range(len(dad_genes)):
            for j in range(rng1):
                for k in range(rng2):
                    for l in range(0, rng3, stride):
                        if coin_flip():
                            dad_genes[i][j][k][l] = mom_genes[i][j][k][l]
                        if coin_flip() and mutable:
                            dad_genes[i][j][k][l] += (2*np.random.rand()-1)/7.5*.0075*mult
    else:
        rng = len(dad_genes[0])
        for i in range(0, len(dad_genes), stride):
            for j in range(rng):
                if coin_flip():
                    dad_genes[i][j] = mom_genes[i][j]
                if coin_flip() and mutable:
                    dad_genes[i][j] += (2*np.random.rand()-1)/7.5*.0075*mult
    return dad_genes


def make_kid(parent, gen, num, mutation_odds, alpha, bias="none"):
    layers = ["input", "conv_2", "dense1", "dense2", "output"]
    bool = True if bias == "rand" else False
    junior = build_model("gen_{}_num_{}".format(gen, num), bool)
    multiplier = 1/10*alpha
    for layer in layers:
        if layer == "dense1":
            stride = np.random.randint(500, 1500)
        elif layer == "conv_2":
            stride = np.random.randint(8, 20)
        else:
            stride = 1
        if layer == "input" or layer == "conv_2":
            conv = True
        else:
            conv = False
        base = parent.get_layer(layer).get_weights()
        params = base[0]
        theta = base[1]
        new_weights = mutate_layer(params, multiplier, mutation_odds, stride, conv)
        mutation = (new_weights, theta)
        junior.get_layer(layer).set_weights(mutation)
    return junior

def mutate_layer(dna, mult, odds, stride=1, conv=False):
    if coin_flip()*coin_flip()*coin_flip()*coin_flip():
        odds = 3
    if conv:
        rng1 = len(dna[0])
        rng2 = len(dna[0][0])
        rng3 = len(dna[0][0][0])
        for i in range(len(dna)):
            for j in range(rng1):
                for k in range(rng2):
                    for l in range(0, rng3, stride):
                        if not np.random.randint(odds):
                            sgn = 1 if coin_flip() else -1
                            dna[i][j][k][l] += mult*sgn*np.random.rand()
    else:
        rng = len(dna[0])
        for i in range(0, len(dna), stride):
            for j in range(rng):
                if not np.random.randint(odds):
                    sgn = 1 if coin_flip() else -1
                    dna[i][j] += mult*sgn*np.random.rand()
    return dna


def coin_flip():
    return np.random.randint(2)


#TODO make mutation severity a function of the current generation
#TODO encode lineage
#TODO discourage "ghost hunting" methods by DQing large spread models

#INDIVIDUALS MUST BE MULTIPLE OF 3

"""
IDEAS:
Remove random in choosing action, see if can then run single attempt
Fuse EVERYTHING, with combos and fusing with random/new model--with smaller stride lengths
Reevalutate scoring metric--somehow implement time survived into criteria
"""

def evolve(generations=75, individuals=12, attempts=1, survivors=3, mutation_odds=510, alpha=1.5, render=False):
    "Starts a new family tree <3"
# try:
    population = homo_erectus(individuals)
    for gen in range(0, generations+1):
        if gen == 12:
            mutation_odds = 10
            alpha = 1
        if gen == 25:
            alpha = 0.75
            mutation_odds = 17
        print("\n•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••\nRunning Generation {}, {}% done\n•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••\n".format(gen, int(gen/generations*100)))
        population = run_generation(population, attempts, gen, render)
        print("\n•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••\nCreating Generation {}\n•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••".format(gen+1))
        population = survival_of_the_fittest(population, individuals, survivors, gen+1, mutation_odds, alpha)
        if (gen+1) % 10 == 0:
            save(population, "checkpoint_{}".format(gen+1))
    population = run_generation(population, attempts)
    return the_fittest(population, 1)[0], population
# except:
#     return None, population



def cont(population, curr_generations=50, generations=300, attempts=1, survivors=3, mutation_odds=10, alpha=2, render=False, individuals=12):
    "Continues a family tree <3"
    try:
        for gen in range(curr_generations, generations+1):
            # if gen >= 12 and gen < 25:
            #     mutation_odds = 15
            #     alpha = 1
            # elif gen >= 25:
            #     alpha = 0.75
            #     mutation_odds = 17
            print("\n•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••\nRunning Generation {}, {}% done\n•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••\n".format(gen, int(gen/generations*100)))
            population = run_generation(population, attempts, gen, render)
            print("\n•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••\nCreating Generation {}\n•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••".format(gen+1))
            population = survival_of_the_fittest(population, individuals, survivors, gen+1, mutation_odds, alpha)
            if (gen+1) % 10 == 0:
                save(population, "checkpoint_{}".format(gen+1))
        population = run_generation(population, attempts)
        return the_fittest(population, 1)[0], population
    except:
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






purpose = input("\nHello Harrison. Hope you're having a nice {}.\nAre you starting a new instance (n), continuing an old one (c), or debugging (d)?\n".format(wkday))
render = input("\nWould you like to see the action? (y or n)\n")
render = True if render=="y" else False
if purpose == "n":
    print("\nLet there be life!\n")
    champion, population = evolve(render=render)
    if champion:
        print("Your champion is:")
        print(champion.name)
        print(champion.summary())
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
        print(champion.summary())
        save(population, "woah")
    else:
        print("\nEarly termination")
        saving=input("\nSave models? (y or n) ")
        if saving == "y" or saving=="yes":
            fp = input("\nName of Directory: ")
            save(population, fp)
else:
    bob = build_model('bob')
