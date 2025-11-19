# Utility scripts
"""
"""

# %%
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import einops as ein
import fast_pytorch_kmeans as fpk
# import faiss
# import faiss.contrib.torch_utils
import random
import os
from PIL import Image
# from sklearn.decomposition import PCA
from typing import Union, List, Tuple, Literal

import matplotlib.pyplot as plt

# %% ------------------- Utility functions -------------------
# Set a seed value
def seed_everything(seed=42):
    """
        Set the `seed` value for torch and numpy seeds. Also turns on
        deterministic execution for cudnn.
        
        Parameters:
        - seed:     A hashable seed value
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to: {seed} (type: {type(seed)})")

# %%
seed_everything()