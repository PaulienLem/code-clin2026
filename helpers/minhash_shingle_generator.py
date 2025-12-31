#%% md
# 1. DBBE
#%%
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, v_measure_score

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU detected - using CuPy acceleration")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("No GPU - using NumPy (CPU mode)")
class ShingleGenerator:
    def __init__(self, shingle_size: int = 4, use_gpu: bool = GPU_AVAILABLE):
        self.shingle_size = shingle_size
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np

    def generate_shingles(self, text: str) -> np.ndarray:
        if len(text) < self.shingle_size:
            return np.array([hash(text) % (2**31)], dtype=np.int32)

        chars = self.xp.array([ord(c) for c in text], dtype=np.int32)
        n_shingles = len(text) - self.shingle_size + 1

        shingles = self.xp.zeros(n_shingles, dtype=np.int32)
        for i in range(self.shingle_size):
            shingles += chars[i:i+n_shingles] * (31 ** i)

        unique_shingles = self.xp.unique(shingles)

        if self.use_gpu:
            unique_shingles = cp.asnumpy(unique_shingles)

        return unique_shingles

    def generate_batch(self, texts: List[str]) -> List[np.ndarray]:
        return [self.generate_shingles(t) for t in texts]