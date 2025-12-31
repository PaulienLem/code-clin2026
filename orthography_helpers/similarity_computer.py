
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
class SimilarityComputer:
    def __init__(self, threshold: float = 0.3, use_gpu: bool = GPU_AVAILABLE):
        self.threshold = threshold
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np

    def compute_batch_similarities(self, query_sig: np.ndarray,
                                   candidate_sigs: np.ndarray) -> np.ndarray:
        if self.use_gpu:
            query_gpu = self.xp.array(query_sig)
            cands_gpu = self.xp.array(candidate_sigs)
            query_expanded = self.xp.tile(query_gpu, (len(candidate_sigs), 1))
            matches = self.xp.sum(query_expanded == cands_gpu, axis=1)
            sims = matches.astype(np.float32) / query_sig.shape[0]
            return cp.asnumpy(sims)
        else:
            query_expanded = np.tile(query_sig, (len(candidate_sigs), 1))
            matches = np.sum(query_expanded == candidate_sigs, axis=1)
            return matches.astype(np.float32) / query_sig.shape[0]