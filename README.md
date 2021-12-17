# `Models.py`

This repo contains examples of computational models implemented in python 🐍


## **`Gillespie_algorithm.py`**  🐰🦊

The Gillespie algorithm generates a statistically correct trajectory of a Stochastic equation system for which the Reaction rates are known. Here I have used it to model the trajectory of the population of 🐰  where a predator is present.

## **`Schelling_Model.py`**  🕺↕️

The Schelling model of segregation is an agent-based model that illustrates how individual tendencies regarding neighbors can lead to segregation. In this model, each agent is separated into one of two groups and places on a ring. Agents then have a probability to switch with each other at each time step. Each encounter may result in the agents trading places or retaining their position. Agents will agree to trade places if and only if at least one of the two agents benefits§ , and none of the two is worse off after the swap.

##  **`SEIAR_Model.py`** 🦠

This file contains a SEIAR Model for modelling the spread of an infectious disease. 

## **`SA_sudoku_solver.py`** 🧩

Simulated annealing (SA) is a method for solving unconstrained and bound-constrained optimization problems. The method models the physical process of heating a material and then slowly lowering the temperature to decrease defects, thus minimizing the system energy. In this file I have used this technique paired with a board represented as a markov chain to solve a sudoku puzzle
