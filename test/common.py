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
