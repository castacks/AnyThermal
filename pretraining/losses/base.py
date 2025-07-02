import torch
import torch.nn as nn
import torch.nn.functional as F

def off_diagonal(x):
    """
    Return the off-diagonal elements of a square matrix.
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()