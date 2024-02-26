import json
import os
import random
from pathlib import Path

import numpy as np
import torch

ROOT_PATH = Path(__file__).parent.parent


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_directory(path):
    if not os.path.exists(str(path)):
        os.mkdir(str(path))


DISTANCEPATH = str(ROOT_PATH / 'assets/distance.txt')
MAPPATH = str(ROOT_PATH / 'assets/pointPos.txt')


def get_distance():
    with open(DISTANCEPATH, 'rb') as f:
        distance = json.loads(f.read())
        return distance


def get_map():
    with open(MAPPATH, 'rb') as f:
        point_pos = json.loads(f.read())
        return point_pos


def getNeighboors(chessman, distance):
    neighboorChessmen = []
    for eachChessman, eachDistance in enumerate(distance[chessman]):
        if eachDistance == 1:
            neighboorChessmen.append(eachChessman)
    return neighboorChessmen


DISTANCE = get_distance()
GAME_MAP = get_map()
MOVE_TO_INDEX_DICT = {}
INDEX_TO_TUPLE_DICT = {}
MOVE_LIST = []
# MOVE_LIST从小到大排列
for from_point in range(21):
    to_point_list = getNeighboors(from_point, DISTANCE)
    to_point_list = sorted(to_point_list)
    for to_point in to_point_list:
        MOVE_LIST.append((from_point, to_point))
for idx, move_tuple in enumerate(MOVE_LIST):
    MOVE_TO_INDEX_DICT[move_tuple] = idx
    INDEX_TO_TUPLE_DICT[idx] = move_tuple

# TODO://对于7x7的矩阵映射关系
ARRAY_TO_IMAGE = {
    0: (0, 3), 15: (6, 3), 6: (3, 0), 10: (3, 6),
    1: (0, 2), 3: (0, 4), 2: (1, 3),
    4: (2, 0), 7: (4, 0), 5: (3, 1),
    8: (2, 6), 9: (3, 5), 11: (4, 6),
    12: (5, 3), 13: (6, 2), 14: (6, 4),
    20: (3, 3),
    16: (2, 3), 17: (3, 2), 18: (3, 4), 19: (4, 3)
}
