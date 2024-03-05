# -*- coding: utf-8 -*-
import sys

import torch


from common import from_array_to_input_tensor, ROOT_PATH

sys.path.append('../src')
sys.path.append(str(ROOT_PATH))
sys.path.append(str(ROOT_PATH / 'build'))

# import torch
# from library import Gomoku
import neural_network
import numpy as np
from library import MCTS, WMChess, NeuralNetwork
from config import config

def tuple_2d_to_numpy_2d(tuple_2d):
    # help function
    # convert type
    res = [None] * len(tuple_2d)
    for i, tuple_1d in enumerate(tuple_2d):
        res[i] = list(tuple_1d)
    return np.array(res)


if __name__ == "__main__":
    # policy_value_net = neural_network.NeuralNetWorkWrapper(config['lr'], config['l2'], config['num_layers'],
    #                                                        config['num_channels'], config['n'], 72, False, False, None)
    # policy_value_net.load_model(str(ROOT_PATH / "models"), "checkpoint")
    # board = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # temp = from_array_to_input_tensor(board)
    # board_batch, last_action_batch, cur_player_batch = torch.tensor(temp
    #                                                                 ).unsqueeze(0), torch.tensor(
    #     [(-1, -1)]), torch.tensor(
    #     [[1]])
    # p, v = policy_value_net._infer(policy_value_net._data_convert(board_batch, last_action_batch, cur_player_batch))
    # print(v)

    libtorch = NeuralNetwork(str(ROOT_PATH / 'models/checkpoint.pt'), config['libtorch_use_gpu'], 12)
    wm = WMChess(7, 1)
    v = libtorch.get_value(wm)
    print(v)
