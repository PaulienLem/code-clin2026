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
from .text_preprocessor import TextPreprocessor
from .shingle_generator import ShingleGenerator
from .lsh_index import LSHIndex
from .similarity_computer import SimilarityComputer
from .minhash_processor import MinHashProcessor
from  .union_find import UnionFind
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU detected - using CuPy acceleration")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("No GPU - using NumPy (CPU mode)")
class FastMinHashClustering:
    def __init__(self, threshold: float = 0.3, shingle_size: int = 4,
                 num_perm: int = 128, chunk_size: int = 50000,
                 use_gpu: Optional[bool] = None):

        if use_gpu is None:
            use_gpu = GPU_AVAILABLE

        self.threshold = threshold
        self.chunk_size = chunk_size
        self.use_gpu = use_gpu and GPU_AVAILABLE

        self.preprocessor = TextPreprocessor(
            lowercase=True,
            remove_punctuation=True,
            remove_diacritics=True
        )
        self.shingler = ShingleGenerator(shingle_size, use_gpu)
        self.minhash = MinHashProcessor(num_perm, use_gpu)
        self.lsh_index = LSHIndex(threshold, num_perm)
        self.similarity_computer = SimilarityComputer(threshold, use_gpu)
        self.all_similarities = []

        mode = "GPU (CuPy)" if self.use_gpu else "CPU (NumPy)"
        print(f"Initialized in {mode} mode")

    def cluster(self, texts: List[str]) -> Tuple[Dict[int, int], List[Tuple[int, int, float]]]:
        n_docs = len(texts)
        n_chunks = (n_docs + self.chunk_size - 1) // self.chunk_size

        print(f"\nClustering {n_docs:,} documents in {n_chunks} chunks")
        print(f"threshold={self.threshold}, chunk_size={self.chunk_size:,}")

        start_time = time.time()

        for chunk_idx in tqdm(range(n_chunks), desc="Processing"):
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size, n_docs)
            chunk_texts = texts[chunk_start:chunk_end]

            processed = self.preprocessor.preprocess_batch(chunk_texts)
            shingles = self.shingler.generate_batch(processed)
            signatures = self.minhash.compute_batch(shingles)
            self.lsh_index.insert_batch(signatures, chunk_start)

            if chunk_start > 0:
                candidates = self.lsh_index.query_batch(signatures, chunk_start)

                for doc_idx, cand_set in enumerate(candidates):
                    if not cand_set:
                        continue

                    query_doc_id = chunk_start + doc_idx
                    query_sig = signatures[doc_idx]

                    cand_list = sorted(cand_set)
                    cand_sigs = []
                    for cand_id in cand_list:
                        batch_idx = cand_id // self.chunk_size
                        local_idx = cand_id % self.chunk_size
                        if batch_idx < len(self.lsh_index.signatures):
                            cand_sigs.append(self.lsh_index.signatures[batch_idx][local_idx])

                    if cand_sigs:
                        cand_sigs = np.array(cand_sigs)
                        sims = self.similarity_computer.compute_batch_similarities(
                            query_sig, cand_sigs
                        )

                        for cand_id, sim in zip(cand_list[:len(sims)], sims):
                            if sim >= self.threshold:
                                self.all_similarities.append((cand_id, query_doc_id, float(sim)))

        elapsed = time.time() - start_time
        print(f"\nFound {len(self.all_similarities):,} similarities in {elapsed:.2f}s")
        print(f"Throughput: {n_docs/elapsed:,.0f} docs/sec")

        print("Building clusters...")
        uf = UnionFind(n_docs)
        for doc1, doc2, _ in tqdm(self.all_similarities, desc="Clustering"):
            uf.union(doc1, doc2)

        clusters = uf.get_clusters()
        n_clusters = len(set(clusters.values()))

        total_time = time.time() - start_time
        print(f"\nCreated {n_clusters:,} clusters in {total_time:.2f}s total")

        return clusters, self.all_similarities
