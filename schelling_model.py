import numpy as np
from copy import deepcopy
import random


def all_possible_swaps(array):
    permutations = []
    for i in range(len(array)):
        for j in range(len(array)):
            if i == j:
                continue
            else:
                current = deepcopy(array)
                current[i], current[j] = current[j], current[i]
                permutations.append(current)

    permutations = [array] + permutations
    return_list = []
    seen = set()
    for item in permutations:
        t = tuple(item)

        if t not in seen:
            return_list.append(item)
            seen.add(t)
    return return_list



def is_valid_swap(current, proposed, states):

    current = [current[-1]] + current + [current[0]]
    current_happiness = 0

    for i in range(1, len(current)-1):
        if states[current[i-1]-1] == states[current[i]-1] or states[current[i+1]-1] == states[current[i]-1]:
            current_happiness += 1

    proposed = [proposed[-1]] + proposed + [proposed[0]]
    proposed_happiness = 0

    for i in range(1, len(current)-1):
        if states[proposed[i-1]-1] == states[proposed[i]-1] or states[proposed[i+1]-1] == states[proposed[i]-1]:
            proposed_happiness += 1

    return proposed_happiness > current_happiness


def produce_markov_row(current_state, permutations, states, epsilon):
    indexes = []
    current_index = 0
    for s in range(len(permutations)):
        state = permutations[s]
        if state == current_state:
            current_index = s
        elif is_valid_swap(current_state, state, states):
            indexes.append(s)

    valid_swap_prob = 1/len(permutations) - epsilon
    current_state_prob = (len(permutations) - len(indexes))/len(permutations) + len(indexes)*epsilon - (len(permutations)-1 - len(indexes))*epsilon
    row = [epsilon for _ in range(len(permutations))]
    row[current_index] = current_state_prob
    for i in indexes:
        row[i] = valid_swap_prob

    return row


def create_markov_chain(n, states, e):
    permutations = all_possible_swaps(list(range(1,n+1)))
    markov_chain = []
    for row in range(len(permutations)):
        state = permutations[row]
        r = produce_markov_row(state, permutations, states, e)
        markov_chain.append(r)

    mc = np.array(markov_chain)

    return mc

def canonical_form(matrix):
    Q = deepcopy(matrix)
    R = deepcopy(matrix)
    indexes_q = []
    indexes_r = []

    for i in range(matrix.shape[1]):
        if 1 in matrix[:, i]:
            indexes_q.append(i)
        else:
            indexes_r.append(i)

    Q = np.delete(Q, indexes_q, axis=0)
    Q = np.delete(Q, indexes_q, axis=1)

    R = np.delete(R, indexes_q, axis=0)
    R = np.delete(R, indexes_r, axis=1)

    O = np.zeros((matrix.shape[0] - Q.shape[0], Q.shape[1]))
    I = np.eye(matrix.shape[0] - R.shape[0])

    # building matrix
    bottom = np.concatenate((O, I), axis=1)
    top = np.concatenate((Q, R), axis=1)
    canonical = np.concatenate((top, bottom), axis=0)
    return canonical, Q


def call_montecarlo():
    for n in range(4, 11):
        print('n = ' + str(n))
        montecarlo_simulation(n)


def numerically_approximate_absorbtion(n):
    states = np.empty((n,))
    states[::2] = 1
    states[1::2] = 0
    mc = create_markov_chain(n, states)
    canonical_f, Q = canonical_form(mc)
    size = Q.shape[0]
    identity = np.eye(size)
    N = np.linalg.inv(identity - Q)
    return N[0].sum(axis=0)


perm = all_possible_swaps([1, 2, 3, 4])
current = [1, 2, 3, 4]
e = 0.001
states = [1, 0, 1, 0]

print(sum(produce_markov_row(current, perm, states, e)))

mc = create_markov_chain(4, [1, 0, 1, 0], 0.001)

def stationary_dist(Q):
    evals, evecs = np.linalg.eig(Q.T)
    evec1 = evecs[:, np.isclose(evals, 1)]
    evec1 = evec1[:, 0]
    stationary = evec1 / evec1.sum()
    stationary = stationary.real
    return stationary

print(list(stationary_dist(mc)))

def montecarlo_simulation(n):

    states = np.empty((n,))
    states[::2] = 1
    states[1::2] = 0
    #random.shuffle(states)
    markov_chain = create_markov_chain(n, states, 0.001)
    indexes = []
    state_index = 0
    for i in range(1000000):
        choice = np.random.choice([i for i in range(0, markov_chain.shape[0])], 1, p=markov_chain[state_index])
        state_index = choice[0]
        indexes.append(state_index)

    n_list = list(range(0, markov_chain.shape[0]))
    stationary_dist = []
    for n in n_list:
        stationary_dist.append(indexes.count(n)/len(indexes))
    return stationary_dist

#print(montecarlo_simulation(4))