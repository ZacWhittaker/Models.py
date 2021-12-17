import random
from math import sqrt, exp, floor
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def create_puzzle(filename):
    file = open(filename, "r")
    lines = file.readlines()
    file.close()
    puzzle = []
    for line in lines:
        puzzle.append([int(val) for val in line.split()])

    return np.array(puzzle)


def get_subgrids(puzzle):
    subgrids = []
    size = len(puzzle)
    root = int(sqrt(size))
    for box_i in range(root):
        for box_j in range(root):
            subgrid = []
            for i in range(root):
                for j in range(root):
                    subgrid.append(puzzle[root*box_i + i][root*box_j + j])
            subgrids.append(subgrid)
    return np.array(subgrids)


def create_solution(puzzle):
    subgrid = get_subgrids(puzzle)
    size = len(puzzle)
    for grid in range(size):
        for z_index in np.where(subgrid[grid] == 0)[0]:

            while subgrid[grid][z_index] == 0:
                r_num = random.randint(1, size)
                if r_num not in subgrid[grid]:
                    subgrid[grid][z_index] = r_num

    return get_subgrids(subgrid)


def cost(puzzle):
    size = len(puzzle)
    columns_sum = 0
    rows_sum = 0
    for i in range(size):
        columns_sum += size - len(set(puzzle[:, i]))

    for i in range(size):
        rows_sum += size - len(set(puzzle[i]))

    return columns_sum+rows_sum


def find_grid_two_zeros(puzzle):
    sub = get_subgrids(puzzle)
    zero_index = []
    for row in range(len(sub)):
        if len(np.where(sub[row] == 0)[0]) >= 2:
            zero_index.append(row)

    grid = random.randint(0, len(zero_index)-1)
    zeros = np.where(sub[zero_index[grid]] == 0)[0]

    n = len(zeros)

    while True:
        index_1 = floor(random.random() * n)
        index_2 = floor(random.random() * n)
        if index_1 != index_2:
            break

    return zero_index[grid], (zeros[index_1], zeros[index_2])


def simulated_annealing(puzzle, t):
    n = len(puzzle)
    solution_found = False
    solution = create_solution(puzzle)
    cost_solution = cost(solution)
    subgrids_original = get_subgrids(puzzle)
    sch = 0.995

    while t > 0:
        t *= sch
        successor = deepcopy(solution)
        # find random subgrid with 2 zero's
        grid, indexes = find_grid_two_zeros(puzzle)
        index_1, index_2 = indexes

        successor = get_subgrids(successor)
        successor[grid][index_1], successor[grid][index_2] = successor[grid][index_2], successor[grid][index_1]
        successor = get_subgrids(successor)
        delta = cost(successor) - cost_solution

        if delta < 0 or random.uniform(0, 1) < exp(-delta / t):
            solution = deepcopy(successor)
            if cost(solution) != cost_solution:
                print(cost(solution))
            cost_solution = cost(solution)
        if cost_solution == 0:
            return solution


def merge_solutions(s1, s2):
    n = len(s1)
    c = random.randint(0, n - 1)
    s1_grid = get_subgrids(s1)[0:c]
    s2_grid = get_subgrids(s2)[c:n]

    merge = np.concatenate((s1_grid, s2_grid), axis=0)
    merge = get_subgrids(merge)

    return merge


def mutate(puzzle, original):
    mutated = deepcopy(puzzle)
    grid, indexes = find_grid_two_zeros(original)
    index_1, index_2 = indexes
    mutated = get_subgrids(mutated)
    mutated[grid][index_1], mutated[grid][index_2] = mutated[grid][index_2], mutated[grid][index_1]
    mutated = get_subgrids(mutated)

    return mutated


def genetic_sudoku(population, original):
    mutation_prob = 0.1
    new_pop = []
    cost_pop = [(b, cost(b)) for b in population]
    sorted_pop = sorted(cost_pop, key=lambda tup: tup[1])
    size = len(population)
    min = 0
    max = size-1

    for i in range(size):
        p1 = floor(abs(random.random() - random.random()) * (1 + max - min) + min)

        p2 = floor(abs(random.random() - random.random()) * (1 + max - min) + min)

        child = merge_solutions(sorted_pop[p1][0], sorted_pop[p2][0])
        if random.random() < mutation_prob:
            child = mutate(child, original)

        new_pop.append(child)
        if cost(child) == 0:
            return child

    return new_pop


p = create_puzzle("Sudoku Problem files-20210520/easy1.sku")


def call_sudoku_genetic(p):

    population = [create_solution(p) for _ in range(5000)]
    for x in tqdm(range(100)):
        population = genetic_sudoku(population, p)

        if np.array(population).shape == (len(p), len(p)):
            return population

    costs = [(b, cost(b)) for b in population]
    sorted_pop = sorted(costs, key=lambda tup: tup[1])
    return sorted_pop

print(call_sudoku_genetic(p))

