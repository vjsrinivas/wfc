import os
import sys
import time
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np


from qiskit import BasicAer, IBMQ
from utils.rand_gen import random_gen

########### Calling Quantum Computer######

api_file = open("API_token.txt", "r")
api_key = ""
if api_file.mode == 'r':
    api_key = api_file.read()
    #print("API_key: ", api_key)

IBMQ.save_account(api_key)
provider = IBMQ.load_account()
backend = provider.get_backend('ibmq_essex')  #select which machine

##########################################

#backend = BasicAer.get_backend("qasm_simulator")  #simulator

q_gen = random_gen(1, backend)


class Tile():
    def __init__(self, img, idx):
        self.img = img
        self.idx = idx
        self.neighbors = {
            'top':    set(),
            'bottom': set(),
            'left':   set(),
            'right':  set()
        }
        self.exclusions = {
            'top':    set(),
            'bottom': set(),
            'left':   set(),
            'right':  set()
        }
        
    def __str__(self):
        plt.figure(figsize = (2, 2))
        plt.imshow(self.img)
        plt.show()
        return "Tile object"
    
    def __getitem__(self, idx):
        return self.img[idx]
        
    def add_neighbor(self, direction, tile):
        self.neighbors[direction].add(tile.idx)


def check_side(side1, side2):
    ratio = 1.0
    num_pixels = np.prod(side1.shape)
    threshold = ratio * num_pixels
    if np.sum(side1 == side2) >= threshold:
        return True
    elif np.sum(side1[:-1] == side2[1:]) >= threshold:
        return True
    elif np.sum(side1[1:] == side2[:-1]) >= threshold:
        return True


def check_and_add(tile1, tile2):
    if check_side(tile1[0, :], tile2[-1, :]):
        tile1.add_neighbor('top', tile2)
        tile2.add_neighbor('bottom', tile1)
    if check_side(tile1[-1, :], tile2[0, :]):
        tile1.add_neighbor('bottom', tile2)
        tile2.add_neighbor('top', tile1)
    if check_side(tile1[:, 0], tile2[:, -1]):
        tile1.add_neighbor('left', tile2)
        tile2.add_neighbor('right', tile1)
    if check_side(tile1[:, -1], tile2[:, 0]):
        tile1.add_neighbor('right', tile2)
        tile2.add_neighbor('left', tile1)


def reduce_prob(choices, tiles, row, col, rows, cols, TILE_IDX_LIST):
    neighbor_choices = []
    valid_choices = deepcopy(TILE_IDX_LIST)
    for i, j, direction in [[row-1, col, 'bottom'], [row+1, col, 'top'], [row, col-1, 'right'], [row, col+1, 'left']]:
        exclusion_idx_list = []
        if 0 <= i < rows and 0 <= j < cols:
            for tile_idx in choices[(i, j)]:
                tile = tiles[tile_idx]
                exclusion_idx_list.append(tile.exclusions[direction])
        total_num = len(exclusion_idx_list)
        if len(exclusion_idx_list) > 0:
            for idx in TILE_IDX_LIST:
                vote = 0
                for exclusion in exclusion_idx_list:
                    if idx in exclusion:
                        vote += 1
                if (vote == total_num) and (idx in valid_choices):
                    valid_choices.remove(idx)
    if len(valid_choices) == 0:
        return None
    else:
        choices[(row, col)] = valid_choices
        return choices


def get_min_entropy_coord(entropy_board, observed):
    rows, cols = entropy_board.shape
    min_row, min_col = -1, -1
    min_entropy = 1000
    coord_list = []
    for row in range(rows):
        for col in range(cols):
            if not observed[row, col]:
                if 1 <= entropy_board[row, col] < min_entropy:
                    min_entropy = entropy_board[row, col]
                    coord_list = []
                    coord_list.append((row, col))
                elif 1 <= entropy_board[row, col] == min_entropy:
                    coord_list.append((row, col))
    if len(coord_list) > 0:
        coord_idx = q_gen.choice(np.arange(len(coord_list)))
        # coord_idx = np.random.choice(np.arange(len(coord_list)))
        min_row, min_col = coord_list[coord_idx]
        return min_row, min_col
    else:
        return -1, -1


def update_entropy(choices, rows, cols):
    entropy_board = np.zeros(shape = (rows, cols))
    for row in range(rows):
        for col in range(cols):
            entropy_board[row, col] = len(choices[(row, col)])
    return entropy_board


def step(info, row_col = None):
    entropy_board   = info['entropy_board']
    tile_idx_list   = info['tile_idx_list']
    observed        = info['observed']
    choices         = info['choices']
    history         = info['history']
    canvas          = info['canvas']
    tiles           = info['tiles']
    rows            = info['rows']
    cols            = info['cols']
    
    if row_col:
        row, col = row_col
    else:
        row, col = get_min_entropy_coord(entropy_board, observed)

    state = q_gen.choice(choices[(row,  col)])
    # state = np.random.choice(choices[(row,  col)])

    history.append((row, col, state, choices[(row,  col)]))
    choices_temp = deepcopy(choices)
    choices_temp[(row, col)] = [state]
    retract = False
    
    # compute new probability for 4 immediate neighbors
    for i, j in [[row-1, col], [row+1, col], [row, col-1], [row, col+1]]:
        if 0 <= i < rows and 0 <= j < cols:
            if not observed[i, j]:
                attempt = reduce_prob(choices_temp, tiles, i, j, rows, cols, tile_idx_list)
                if attempt:
                    choices_temp = attempt
                else:
                    retract = True
                    break
    
    canvas[row,  col] = state
    observed[row, col] = True
    
    info['entropy_board']   = entropy_board
    info['observed']        = observed
    info['choices']         = choices_temp
    info['history']         = history
    info['canvas']          = canvas
    info['tiles']           = tiles
    
    return info, retract
