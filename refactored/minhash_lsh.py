"""
MinHash LSH implementation for verse similarity detection.
Uses datasketch library for efficient near-duplicate detection.
"""
import numpy as np
from datasketch import MinHash, MinHashLSH
from typing import List, Dict, Tuple, Set
import logging
from collections import defaultdict
import multiprocessing as mp
from functools import partial
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MinHashVectorizer:
    """
    Creates MinHash signatures for text documents.
    Supports various shingle sizes and parallel processing.
    """
    
    def __init__(self, shingle_size: int = 3, num_perm: int = 128, seed: int = 42):
        """
        Initialize the MinHash vectorizer.
        
        Args:
            shingle_size: Size of character n-grams (shingles)
            num_perm: Number of permutations for MinHash (more = more accurate)
            seed: Random seed for reproducibility
        """
        self.shingle_size = shingle_size
        self.num_perm = num_perm
        self.seed = seed
        
    def create_shingles(self, text: str) -> Set[str]:
        """
        Create character-level shingles from text.
        
        Args:
            text: Input text
            
        Returns:
            Set of shingles
        """
        # Normalize text
        if not isinstance(text, str) or not text:
            return set()
        
        text = text.lower().strip()
        
        if len(text) < self.shingle_size:
            # For very short texts, just return the whole text as a single shingle
            return {text} if text else set()
        
        # Create character-level shingles
        shingles = set()
        for i in range(len(text) - self.shingle_size + 1):
            shingle = text[i:i + self.shingle_size]
            shingles.add(shingle)
        
        return shingles
    
    def create_minhash(self, text: str) -> MinHash:
        """
        Create MinHash signature for a single text.
        
        Args:
            text: Input text
            
        Returns:
            MinHash object
        """
        minhash = MinHash(num_perm=self.num_perm, seed=self.seed)
        shingles = self.create_shingles(text)
        
        # Handle empty shingle sets
        if not shingles:
            # Create a minimal signature for empty/very short texts
            # This ensures they still get a valid MinHash object
            minhash.update(b'__empty__')
        else:
            for shingle in shingles:
                minhash.update(shingle.encode('utf8'))
        
        return minhash
    
    def create_minhash_batch(self, texts: List[str], 
                            text_ids: List[str],
                            n_jobs: int = -1) -> Dict[str, MinHash]:
        """
        Create MinHash signatures for multiple texts in parallel.
        
        Args:
            texts: List of input texts
            text_ids: List of text identifiers
            n_jobs: Number of parallel jobs (-1 for all cores)
            
        Returns:
            Dictionary mapping text_id to MinHash
        """
        start_time = time.time()
        
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        
        logger.info(f"Creating MinHash signatures for {len(texts)} texts "
                   f"(shingle_size={self.shingle_size}, num_perm={self.num_perm}, "
                   f"n_jobs={n_jobs})")
        
        # Create partial function with fixed parameters
        create_minhash_func = partial(self._create_minhash_worker)
        
        # Process in parallel
        with mp.Pool(processes=n_jobs) as pool:
            minhashes = pool.map(create_minhash_func, texts)
        
        # Create dictionary
        minhash_dict = dict(zip(text_ids, minhashes))
        
        elapsed = time.time() - start_time
        logger.info(f"Created {len(minhash_dict)} MinHash signatures in {elapsed:.2f}s "
                   f"({len(minhash_dict)/elapsed:.0f} signatures/s)")
        
        return minhash_dict
    
    def _create_minhash_worker(self, text: str) -> MinHash:
        """Worker function for parallel MinHash creation."""
        return self.create_minhash(text)


class LSHClusterer:
    """
    Performs LSH-based clustering using MinHash signatures.
    Implements union-find for efficient cluster merging.
    """
    
    def __init__(self, threshold: float = 0.8, num_perm: int = 128):
        """
        Initialize LSH clusterer.
        
        Args:
            threshold: Jaccard similarity threshold for matching
            num_perm: Number of permutations (must match MinHash)
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = None
        
    def build_lsh_index(self, minhash_dict: Dict[str, MinHash]):
        """
        Build LSH index from MinHash signatures.
        
        Args:
            minhash_dict: Dictionary mapping IDs to MinHash objects
        """
        start_time = time.time()
        
        logger.info(f"Building LSH index with {len(minhash_dict)} items "
                   f"(threshold={self.threshold})")
        
        # Create LSH index
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        
        # Insert all MinHash signatures
        for text_id, minhash in minhash_dict.items():
            self.lsh.insert(text_id, minhash)
        
        elapsed = time.time() - start_time
        logger.info(f"Built LSH index in {elapsed:.2f}s")
        
    def find_similar_pairs(self, minhash_dict: Dict[str, MinHash]) -> Set[Tuple[str, str]]:
        """
        Find all similar pairs using LSH.
        
        Args:
            minhash_dict: Dictionary mapping IDs to MinHash objects
            
        Returns:
            Set of similar pairs (as tuples)
        """
        if self.lsh is None:
            raise ValueError("LSH index not built. Call build_lsh_index() first.")
        
        start_time = time.time()
        logger.info("Finding similar pairs using LSH")
        
        similar_pairs = set()
        
        for text_id, minhash in minhash_dict.items():
            # Query LSH for similar items
            candidates = self.lsh.query(minhash)
            
            # Add pairs (avoid duplicates by sorting)
            for candidate_id in candidates:
                if candidate_id != text_id:
                    pair = tuple(sorted([text_id, candidate_id]))
                    similar_pairs.add(pair)
        
        elapsed = time.time() - start_time
        logger.info(f"Found {len(similar_pairs)} similar pairs in {elapsed:.2f}s")
        
        return similar_pairs
    
    def cluster_with_union_find(self, similar_pairs: Set[Tuple[str, str]], 
                                all_ids: List[str]) -> Dict[str, int]:
        """
        Cluster items using union-find algorithm.
        
        Args:
            similar_pairs: Set of similar pairs
            all_ids: List of all item IDs
            
        Returns:
            Dictionary mapping item_id to cluster_id
        """
        start_time = time.time()
        logger.info(f"Clustering {len(all_ids)} items using union-find")
        
        # Initialize union-find structure
        parent = {item_id: item_id for item_id in all_ids}
        rank = {item_id: 0 for item_id in all_ids}
        
        def find(x):
            """Find root with path compression."""
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            """Union by rank."""
            root_x = find(x)
            root_y = find(y)
            
            if root_x == root_y:
                return
            
            if rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            elif rank[root_x] > rank[root_y]:
                parent[root_y] = root_x
            else:
                parent[root_y] = root_x
                rank[root_x] += 1
        
        # Merge similar pairs
        for item1, item2 in similar_pairs:
            union(item1, item2)
        
        # Assign cluster IDs
        cluster_map = {}
        cluster_id = 0
        root_to_cluster = {}
        
        for item_id in all_ids:
            root = find(item_id)
            if root not in root_to_cluster:
                root_to_cluster[root] = cluster_id
                cluster_id += 1
            cluster_map[item_id] = root_to_cluster[root]
        
        elapsed = time.time() - start_time
        n_clusters = len(set(cluster_map.values()))
        logger.info(f"Created {n_clusters} clusters in {elapsed:.2f}s")
        
        return cluster_map
    
    def fit_predict(self, minhash_dict: Dict[str, MinHash]) -> Dict[str, int]:
        """
        Complete clustering pipeline: build index, find pairs, cluster.
        
        Args:
            minhash_dict: Dictionary mapping IDs to MinHash objects
            
        Returns:
            Dictionary mapping item_id to cluster_id
        """
        self.build_lsh_index(minhash_dict)
        similar_pairs = self.find_similar_pairs(minhash_dict)
        all_ids = list(minhash_dict.keys())
        cluster_map = self.cluster_with_union_find(similar_pairs, all_ids)
        
        return cluster_map