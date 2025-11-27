import numpy as np
import random
from collections import defaultdict
from numba import jit, prange
import logging

logger = logging.getLogger(__name__)


@jit(nopython=True, parallel=True, cache=True)
def union_find_batch(edges, n_items):
    parent = np.arange(n_items, dtype=np.int32)
    rank = np.zeros(n_items, dtype=np.int32)
    
    for edge_idx in range(len(edges)):
        x = edges[edge_idx, 0]
        y = edges[edge_idx, 1]
        
        root_x = x
        while parent[root_x] != root_x:
            parent[root_x] = parent[parent[root_x]]
            root_x = parent[root_x]
        
        root_y = y
        while parent[root_y] != root_y:
            parent[root_y] = parent[parent[root_y]]
            root_y = parent[root_y]
        
        if root_x != root_y:
            if rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            elif rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            else:
                parent[root_y] = root_x
                rank[root_x] += 1
    
    clusters = np.empty(n_items, dtype=np.int32)
    for i in range(n_items):
        root = i
        while parent[root] != root:
            root = parent[root]
        clusters[i] = root
    return clusters


class LSHClusterer:
    def __init__(self, max_edges=30000000):
        self.max_edges = max_edges
    
    def cluster(self, signatures, ids, num_perm, threshold):
        n_bands, rows_per_band = self._calculate_optimal_bands(threshold, num_perm)
        
        n_items = len(ids)
        buckets = [defaultdict(list) for _ in range(n_bands)]
        
        max_bucket_size = 2000
        
        for idx in range(n_items):
            sig = signatures[idx]
            
            for band_idx in range(n_bands):
                start_row = band_idx * rows_per_band
                end_row = start_row + rows_per_band
                band_hash = hash(tuple(sig[start_row:end_row].tolist()))
                
                if len(buckets[band_idx][band_hash]) < max_bucket_size:
                    buckets[band_idx][band_hash].append(idx)
        
        edges = set()
        
        for bucket in buckets:
            for bucket_items in bucket.values():
                n_items_bucket = len(bucket_items)
                
                if n_items_bucket < 2:
                    continue
                
                if n_items_bucket <= 50:
                    for i in range(n_items_bucket):
                        for j in range(i + 1, n_items_bucket):
                            edges.add(tuple(sorted([bucket_items[i], bucket_items[j]])))
                elif n_items_bucket <= 200:
                    for i in range(min(50, n_items_bucket)):
                        for j in range(i + 1, min(i + 50, n_items_bucket)):
                            edges.add(tuple(sorted([bucket_items[i], bucket_items[j]])))
                else:
                    sample_size = min(100, n_items_bucket)
                    sampled = random.sample(bucket_items, sample_size)
                    for i in range(len(sampled)):
                        for j in range(i + 1, min(i + 20, len(sampled))):
                            edges.add(tuple(sorted([sampled[i], sampled[j]])))
                
                if len(edges) >= self.max_edges:
                    break
            
            if len(edges) >= self.max_edges:
                break
        
        if len(edges) > 0:
            edges_array = np.array(list(edges), dtype=np.int32)
            clusters_array = union_find_batch(edges_array, n_items)
            cluster_map = {ids[idx]: int(clusters_array[idx]) for idx in range(n_items)}
        else:
            cluster_map = {ids[idx]: idx for idx in range(n_items)}
        
        return cluster_map
    
    def _calculate_optimal_bands(self, threshold, num_perm):
        candidates = []
        
        for b in range(1, num_perm + 1):
            if num_perm % b == 0:
                r = num_perm // b
                if r >= 2:
                    prob_threshold = (1.0 / b) ** (1.0 / r)
                    error = abs(prob_threshold - threshold)
                    candidates.append((error, b, r))
        
        if not candidates:
            return 1, num_perm
        
        candidates.sort()
        _, best_bands, best_rows = candidates[0]
        
        return best_bands, best_rows