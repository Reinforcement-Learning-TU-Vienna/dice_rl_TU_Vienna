# ---------------------------------------------------------------- #

import random
import numpy as np
import torch
import tensorflow as tf
import os

# ---------------------------------------------------------------- #

def set_all_seeds(seed=None):
    if seed is None: return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ---------------------------------------------------------------- #
