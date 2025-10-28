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
import faiss
import faiss.contrib.torch_utils
import random
import os
from PIL import Image
from sklearn.decomposition import PCA
from typing import Union, List, Tuple, Literal

import matplotlib.pyplot as plt

# %% -------------------- Converter functions --------------------
# Convert to numpy
def to_np(x, ret_type=float) -> np.ndarray:
    """
        Converts 'x' to numpy object of `dtype` as 'ret_type'
        
        Parameters:
        - x:    An object
        
        Returns:
        - x_np:     A numpy array of dtype `ret_type`
    """
    x_np: np.ndarray = None
    if type(x) == torch.Tensor:
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.array(x)
    x_np = x_np.astype(ret_type)
    return x_np

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