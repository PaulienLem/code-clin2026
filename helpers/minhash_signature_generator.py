
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

class MinHashSignatureGenerator:
    def __init__(self, num_perm: int = 128, use_gpu: bool = GPU_AVAILABLE):
        self.num_perm = num_perm
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np

        rng = self.xp.random.RandomState(42)
        self.hash_a = rng.randint(1, 2**31-1, num_perm, dtype=np.int64)
        self.hash_b = rng.randint(0, 2**31-1, num_perm, dtype=np.int64)
        self.prime = np.int64(2**31-1)

        if self.use_gpu:
            print(f"Using GPU for MinHash ({num_perm} permutations)")

    def compute_signature(self, shingles: np.ndarray) -> np.ndarray:
        if len(shingles) == 0:
            return np.full(self.num_perm, self.prime, dtype=np.int64)

        if self.use_gpu:
            shingles_gpu = self.xp.array(shingles, dtype=np.int64)
        else:
            shingles_gpu = shingles.astype(np.int64)

        shingles_expanded = shingles_gpu[:, self.xp.newaxis]
        hashes = (self.hash_a * shingles_expanded + self.hash_b) % self.prime
        signature = self.xp.min(hashes, axis=0)

        if self.use_gpu:
            signature = cp.asnumpy(signature)

        return signature

    def compute_batch(self, shingles_batch: List[np.ndarray]) -> np.ndarray:
        signatures = np.zeros((len(shingles_batch), self.num_perm), dtype=np.int64)
        for i, shingles in enumerate(shingles_batch):
            signatures[i] = self.compute_signature(shingles)
        return signatures
