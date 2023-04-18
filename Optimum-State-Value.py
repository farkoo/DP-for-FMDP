# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 17:02:50 2023

@author: ACER
"""

import numpy as np


world_size = 5
A_pos = [0, 1]
A_prime_pos = [4, 1]
B_pos = [0, 3]
B_prime_pos = [2, 3]
discount = 0.9

# left, up, right, down
actions = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]

action_prob = 0.25


def step(state, action):
    if state == A_pos:
        return A_prime_pos, 10
    if state == B_pos:
        return B_prime_pos, 5

    next_state = (np.array(state) + action).tolist()
    x, y = next_state
    if x < 0 or x >= world_size or y < 0 or y >= world_size:
        reward = -1.0
        next_state = state
    else:
        reward = 0
    return next_state, reward

def calculate_value_function():
    value = np.zeros((world_size, world_size))
    while True:
        # keep iteration until convergence
        new_value = np.zeros_like(value)
        for i in range(world_size):
            for j in range(world_size):
                for action in actions:
                    (next_i, next_j), reward = step([i, j], action)
                    # bellman equation
                    new_value[i, j] += action_prob * (reward + discount * value[next_i, next_j])
        if np.sum(np.abs(value - new_value)) < 1e-4:
            break
        value = new_value
    print(np.round(value, decimals=1))


if __name__ == '__main__':
    calculate_value_function()