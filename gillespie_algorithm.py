import numpy as np
import matplotlib.pyplot as plt
from random import random
from tqdm import tqdm

b_r = 0.8
d_r = 0.1
b_f = 0.1
d_f = 0.6
a = 0.04
beta = 0.01


'''

X^ = x(b - d) - a(x*y) - x*f
Y^ = y*(b^ - d^) + y*Beta*x - y*(f/2)

rabbit population growth = (b - d) -> (x = x + 1)
fox population decline = (b^ - d^) -> (y = y - 1)
Rabbit is eaten =  a(x*y) -> (x = x - 1)
Fox eats food =  B*x*y -> (y = y + 1; x = x - 1)
Rabbit is hunted = x*f -> (x = x - 1)
Fox is hunted = y*(f/2) -> (y = y - 1)
'''

def probabilities(x, y, f):
    rabbit_population_growth = (b_r - d_r)*x
    fox_population_decline = abs(b_f - d_f)*y
    rabbit_eaten = a*x*y
    fox_eats = beta*x*y
    rabbit_hunted = x*f
    fox_hunted = y*(f/2)

    total = rabbit_population_growth + fox_population_decline + rabbit_eaten + fox_eats + rabbit_hunted + fox_hunted
    probabilities = [rabbit_population_growth, fox_population_decline, rabbit_eaten, fox_eats, rabbit_hunted, fox_hunted]
    probabilities = [p/total for p in probabilities]
    return probabilities, total


def sample(lamda):
    return -1*np.log(random())/lamda


def gillespie(f):
    rabbits = 25
    foxes = 15
    r = []
    t = 0
    x = [25]
    y = [15]
    time = [0]
    action_species = ['r', 'f', 'r', 'f', 'r', 'f']
    while t < 10:
        actions = [rabbits + 1, foxes - 1, rabbits - 1, foxes + 1, rabbits - 1, foxes - 1]
        actions = [(actions[i], action_species[i]) for i in range(len(actions))]

        probability, q = probabilities(rabbits, foxes, f)
        time_ocour = sample(q)
        result = np.random.choice(len(actions), p=probability)
        r.append(result)
        if actions[result][1] == 'r':
            rabbits = actions[result][0]
        else:
            foxes = actions[result][0]

        t+= time_ocour
        time.append(t)
        x.append(rabbits)
        y.append(foxes)
    return x, y, time



def change_in_rabbit(x, y, f):
    return x*(b_r - d_r) - a*(x*y) - x*f


def change_in_fox(x, y, f):
    return y*(b_f - d_f) + y*beta*x - y*(f/2)


def rk2(f, f2, tspan, f0, f2_0, h, b, hunt_rate):

    x = np.arange(tspan[0], tspan[1], h).tolist()
    n = len(x)
    rabbits = [f0 for _ in range(n)]
    foxes = [f2_0 for _ in range(n)]

    a = 1-b
    alpha = 1/(2*b)

    for i in range(n-1):
        k1 = f(rabbits[i], foxes[i], hunt_rate)
        k2 = f(rabbits[i]+(alpha*h), foxes[i]+(alpha*k1*h), hunt_rate)
        rabbits[i+1] = rabbits[i] + h*(a*k1 + b*k2)

        k1 = f2(rabbits[i], foxes[i], hunt_rate)
        k2 = f2(rabbits[i] + (alpha * h), foxes[i] + (alpha * k1 * h), hunt_rate)
        foxes[i + 1] = foxes[i] + h * (a * k1 + b * k2)

    return rabbits, foxes, x

def call_algo(hunt_rate):
    r, f, x = rk2(change_in_rabbit, change_in_fox, [0, 100], 25, 15, 0.001, 0.4, hunt_rate)
    return r, f, x

def call_gillespie():
    hunting_rates = np.linspace(0, 1, num=40)
    probability_extinct = []
    for rate in tqdm(hunting_rates):
        total = 0
        for _ in range(100):
            try:
                x, y, t = gillespie(rate)
            except:
                total += 1
                continue

            if min(x) == 0 or min(y) == 0:
                total += 1

        probability_extinct.append(total/100)


    plt.plot(hunting_rates, probability_extinct)

    plt.title('hunting rates vs probability of extinction')
    plt.axhline(y=0.2, color = 'orange')


    plt.show()


#call_gillespie()

def call_rk2_different_rates():

    hunting_rates = [0.001, 0.01, 0.1, 0.4, 0.6]

    for rate in hunting_rates:
        x, y, t = call_algo(rate)
        rabbits, = plt.plot(t, x, label='rabbit population')
        foxes, = plt.plot(t, y, label='fox population')
        plt.title('Gillepsie simulation hunting rate = {0}'.format(rate))
        plt.xlabel('Rabbits')
        plt.ylabel('Foxes')
        plt.legend(handles=[rabbits, foxes], loc='best')
        plt.show()



def call_rk2_different_rates_phase():

    hunting_rates = [0.001, 0.01, 0.1, 0.4, 0.6]

    for rate in hunting_rates:
        x, y, t = call_algo(rate)
        plt.plot(x, y)
        plt.title('Gillepsie simulation hunting rate = {0} phase plot'.format(rate))
        plt.xlabel('Rabbits')
        plt.ylabel('Foxes')
        plt.show()

call_rk2_different_rates_phase()

def call_gillepsie_1B():

    hunting_rates = [0.001, 0.01, 0.1, 0.4, 0.6]

    for rate in hunting_rates:
        x, y, t = gillespie(rate)
        rabbits, = plt.plot(t, x, label='rabbit population')
        foxes, = plt.plot(t, y, label='fox population')
        plt.title('Gillepsie simulation hunting rate = {0}'.format(rate))
        plt.xlabel('time (years)')
        plt.ylabel('population')
        plt.legend(handles=[rabbits, foxes], loc='best')
        plt.show()



