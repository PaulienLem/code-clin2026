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

class LSHIndex:
    def __init__(self, threshold: float = 0.3, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm = num_perm
        self.bands = 16
        self.rows = num_perm // self.bands
        self.signatures = []
        self.num_docs = 0
        self.hash_tables = [defaultdict(list) for _ in range(self.bands)]

    def _hash_band(self, band: np.ndarray) -> int:
        return int(hash(tuple(band)) % (2**31))

    def insert_batch(self, signatures: np.ndarray, start_idx: int):
        batch_size = signatures.shape[0]
        self.signatures.append(signatures)

        for band_idx in range(self.bands):
            start_row = band_idx * self.rows
            end_row = start_row + self.rows

            for doc_idx in range(batch_size):
                band = signatures[doc_idx, start_row:end_row]
                band_hash = self._hash_band(band)
                global_doc_id = start_idx + doc_idx
                self.hash_tables[band_idx][band_hash].append(global_doc_id)

        self.num_docs += batch_size

    def query_batch(self, signatures: np.ndarray, start_idx: int) -> List[set]:
        batch_size = signatures.shape[0]
        candidates = [set() for _ in range(batch_size)]

        for band_idx in range(self.bands):
            start_row = band_idx * self.rows
            end_row = start_row + self.rows

            for doc_idx in range(batch_size):
                query_doc_id = start_idx + doc_idx
                band = signatures[doc_idx, start_row:end_row]
                band_hash = self._hash_band(band)
                bucket = self.hash_tables[band_idx].get(band_hash, [])
                candidates[doc_idx].update(c for c in bucket if c < query_doc_id)

        return candidates
