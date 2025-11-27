import numpy as np
import pandas as pd
from numba import jit, prange
import multiprocessing as mp
from pathlib import Path
import logging
import time
from typing import List, Dict, Optional, Tuple
import random
from collections import defaultdict
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from visualizer import Visualizer
from report_generator import ReportGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def create_signatures_numba(texts_encoded, shingle_size, num_perm, prime, a_params, b_params):
    n_texts = len(texts_encoded)
    signatures = np.full((n_texts, num_perm), prime, dtype=np.int32)
    
    for text_idx in prange(n_texts):
        text = texts_encoded[text_idx]
        text_len = len(text)
        
        if text_len < shingle_size:
            continue
        
        sig = np.full(num_perm, prime, dtype=np.int32)
        
        for i in range(text_len - shingle_size + 1):
            shingle_hash = 0
            for j in range(shingle_size):
                shingle_hash = shingle_hash * 31 + text[i + j]
            
            shingle_hash = shingle_hash % prime
            
            for perm_idx in range(num_perm):
                h = (a_params[perm_idx] * shingle_hash + b_params[perm_idx]) % prime
                if h < sig[perm_idx]:
                    sig[perm_idx] = h
        
        signatures[text_idx] = sig
    
    return signatures


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

class CompletePipeline:
    def __init__(self,
                 csv_path: str,
                 scratch_dir: str = '/scratch/gent/vo/000/gvo00042/vsc48660',
                 output_dir: str = './results',
                 num_perms: Optional[List[int]] = None,
                 shingle_sizes: Optional[List[int]] = None,
                 thresholds: Optional[List[float]] = None,
                 poem_thresholds: Optional[List[float]] = None,
                 n_jobs: int = -1,
                 max_edges: int = 30000000,
                 auto_optimize: bool = True):
        
        self.csv_path = csv_path
        self.scratch_dir = Path(scratch_dir) / 'complete_clustering'
        self.output_dir = Path(output_dir)
        self.max_edges = max_edges
        self.auto_optimize = auto_optimize
        
        self.cpu_count = psutil.cpu_count(logical=False)
        self.cpu_count_logical = psutil.cpu_count(logical=True)
        self.total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if auto_optimize:
            self._optimize_for_system()
        
        self.n_jobs = n_jobs if n_jobs != -1 else self.optimal_n_jobs
        
        self.num_perms = num_perms if num_perms else [16, 24]
        self.shingle_sizes = shingle_sizes if shingle_sizes else [3, 4]
        self.thresholds = thresholds if thresholds else [0.75, 0.8, 0.85]
        self.poem_thresholds = poem_thresholds if poem_thresholds else [0.2, 0.3, 0.4, 0.5]
        
        self.prime = 2147483647
        
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.has_verse_gt = False
        self.has_poem_gt = False
        self.best_verse_params = None
        self.verse_clusters = None
        self.poem_clusters = None
        
        self.timing = {}
        self.resources = {
            'cpu_percent': [],
            'memory_percent': [],
            'memory_used_gb': []
        }
        self.process = psutil.Process()

    def _optimize_for_system(self):
        logger.info(f"\nSystem detected: {self.cpu_count} cores, {self.total_memory_gb:.0f} GB RAM")
        
        if self.cpu_count >= 64:
            self.optimal_n_jobs = int(self.cpu_count * 0.8)  # Use 80% of cores
            self.optimal_chunk_size = 50000
            self.batch_grid_search = True
            logger.info(f"→ HPC mode: Using {self.optimal_n_jobs} cores")
        elif self.cpu_count >= 16:
            self.optimal_n_jobs = int(self.cpu_count * 0.9)
            self.optimal_chunk_size = 25000
            self.batch_grid_search = False
            logger.info(f"→ Workstation mode: Using {self.optimal_n_jobs} cores")
        else:
            self.optimal_n_jobs = max(1, self.cpu_count - 1)
            self.optimal_chunk_size = 10000
            self.batch_grid_search = False
            logger.info(f"→ Desktop mode: Using {self.optimal_n_jobs} cores")
        
        if self.total_memory_gb >= 256:
            self.optimal_chunk_size = 100000
            self.use_memory_aggressive = True
            logger.info(f"→ Large memory mode: {self.optimal_chunk_size} chunk size")
        elif self.total_memory_gb >= 64:
            self.use_memory_aggressive = False
            logger.info(f"→ Standard memory mode: {self.optimal_chunk_size} chunk size")
        else:
            self.optimal_chunk_size = min(self.optimal_chunk_size, 10000)
            self.use_memory_aggressive = False
            logger.info(f"→ Conservative memory mode: {self.optimal_chunk_size} chunk size")
        
    def _track_resources(self):
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        mem_process = self.process.memory_info().rss / (1024**3)  # GB
        
        self.resources['cpu_percent'].append(cpu)
        self.resources['memory_percent'].append(mem.percent)
        self.resources['memory_used_gb'].append(mem_process)
        
    def run_full_pipeline(self):
        pipeline_start = time.time()
        logger.info("="*70)
        logger.info("COMPLETE PRODUCTION PIPELINE")
        logger.info("="*70)
        logger.info(f"Scratch: {self.scratch_dir}")
        logger.info(f"Verse grid search: {len(self.num_perms)}×{len(self.shingle_sizes)}×{len(self.thresholds)} = {len(self.num_perms)*len(self.shingle_sizes)*len(self.thresholds)} configs")
        logger.info(f"Poem grid search: {len(self.poem_thresholds)} thresholds")
        
        self._load_and_validate_data()
        self._run_verse_clustering()
        self._run_poem_clustering()
        self._save_results()
        
        total_time = time.time() - pipeline_start
        self.timing['total'] = total_time
        self._generate_report()
        
        logger.info(f"\nPipeline completed in {total_time/60:.1f} minutes")
    
    def _load_and_validate_data(self):
        start = time.time()
        logger.info("\nStep 1: Loading and validating data")
        self._track_resources()
        
        self.df = pd.read_csv(self.csv_path)
        
        required_cols = ['verse', 'idoriginal_poem', 'order']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        if 'id' not in self.df.columns:
            self.df['id'] = self.df.index.astype(str)
        
        self.df['verse'] = self.df['verse'].fillna('').astype(str)
        
        empty_mask = self.df['verse'].str.strip() == ''
        if empty_mask.any():
            logger.warning(f"Removing {empty_mask.sum():,} empty verses")
            self.df = self.df[~empty_mask].reset_index(drop=True)
        
        self.has_verse_gt = 'idgroup' in self.df.columns
        self.has_poem_gt = 'type_id' in self.df.columns
        
        if self.has_verse_gt:
            nan_mask = self.df['idgroup'].isna()
            if nan_mask.any():
                logger.warning(f"Removing {nan_mask.sum():,} verses with NaN ground truth")
                self.df = self.df[~nan_mask].reset_index(drop=True)
        
        logger.info(f"Loaded {len(self.df):,} verses from {self.df['idoriginal_poem'].nunique():,} poems")
        logger.info(f"Has verse ground truth: {self.has_verse_gt}")
        logger.info(f"Has poem ground truth: {self.has_poem_gt}")
        
        self.timing['load'] = time.time() - start
        self._track_resources()
        
    def _run_verse_clustering(self):
        start = time.time()
        logger.info("\nStep 2: Verse-level clustering with grid search")
        self._track_resources()
        
        if self.has_verse_gt:
            logger.info("Running supervised grid search on full data")
            self._grid_search_supervised()
        else:
            logger.info("Running unsupervised grid search on sample")
            self._grid_search_unsupervised()
        
        logger.info(f"\nBest verse parameters: {self.best_verse_params}")
        
        logger.info("\nClustering full dataset with best parameters...")
        self._cluster_full_dataset()
        
        self.timing['verse_clustering'] = time.time() - start
        self._track_resources()
        logger.info(f"Verse clustering completed in {self.timing['verse_clustering']/60:.1f} minutes")
    
    def _grid_search_unsupervised(self):
        sample_size = max(15000, int(len(self.df) * 0.01))
        sample_indices = np.random.choice(len(self.df), size=sample_size, replace=False)
        sample_df = self.df.iloc[sample_indices].copy()
        
        logger.info(f"Grid search on {len(sample_df):,} verses ({len(sample_df)/len(self.df)*100:.2f}%)")
        
        if hasattr(self, 'batch_grid_search') and self.batch_grid_search:
            logger.info(f"HPC mode: Each config uses {self.n_jobs} cores (Numba parallel)")
        
        results = self._sequential_grid_search(sample_df)
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / 'verse_grid_search_results.csv', index=False)
    
    def _sequential_grid_search(self, sample_df):
        results = []
        best_score = -np.inf
        
        for num_perm in self.num_perms:
            for shingle_size in self.shingle_sizes:
                for threshold in self.thresholds:
                    config_start = time.time()
                    
                    logger.info(f"  Testing: perm={num_perm}, shingle={shingle_size}, thresh={threshold:.2f}")
                    
                    cluster_map = self._cluster_config(
                        sample_df,
                        num_perm,
                        shingle_size,
                        threshold
                    )
                    
                    score, metrics = self._calculate_quality_metrics(cluster_map, len(sample_df))
                    
                    result = {
                        'num_perm': num_perm,
                        'shingle_size': shingle_size,
                        'threshold': threshold,
                        'score': score,
                        'time': time.time() - config_start
                    }
                    result.update(metrics)
                    results.append(result)
                    
                    logger.info(f"    Score: {score:.4f}, clusters: {metrics['n_clusters']}, "
                              f"avg_size: {metrics['avg_cluster_size']:.1f}")
                    
                    if score > best_score:
                        best_score = score
                        self.best_verse_params = {
                            'num_perm': num_perm,
                            'shingle_size': shingle_size,
                            'threshold': threshold
                        }
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / 'verse_grid_search_results.csv', index=False)
        return results
        
    def _grid_search_supervised(self):
        logger.info("Running supervised grid search with ARI")
        
        from sklearn.metrics import adjusted_rand_score
        
        results = []
        best_score = -np.inf
        
        for num_perm in self.num_perms:
            for shingle_size in self.shingle_sizes:
                for threshold in self.thresholds:
                    config_start = time.time()
                    
                    logger.info(f"  Testing: perm={num_perm}, shingle={shingle_size}, thresh={threshold:.2f}")
                    
                    cluster_map = self._cluster_config(
                        self.df,
                        num_perm,
                        shingle_size,
                        threshold
                    )
                    
                    true_labels = self.df['idgroup'].tolist()
                    pred_labels = [cluster_map.get(str(vid), -1) for vid in self.df['id']]
                    
                    ari = adjusted_rand_score(true_labels, pred_labels)
                    n_clusters = len(set(cluster_map.values()))
                    
                    result = {
                        'num_perm': num_perm,
                        'shingle_size': shingle_size,
                        'threshold': threshold,
                        'ari': ari,
                        'n_clusters': n_clusters,
                        'time': time.time() - config_start
                    }
                    results.append(result)
                    
                    logger.info(f"    ARI: {ari:.4f}, clusters: {n_clusters}")
                    
                    if ari > best_score:
                        best_score = ari
                        self.best_verse_params = {
                            'num_perm': num_perm,
                            'shingle_size': shingle_size,
                            'threshold': threshold
                        }
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / 'verse_grid_search_results.csv', index=False)
    
    def _cluster_config(self, df, num_perm, shingle_size, threshold):
        texts = df['verse'].str.lower().tolist()
        ids = df['id'].astype(str).tolist()
        
        signatures = self._create_signatures(texts, num_perm, shingle_size)
        
        cluster_map = self._lsh_cluster(signatures, ids, num_perm, threshold)
        
        return cluster_map
    
    def _create_signatures(self, texts, num_perm, shingle_size):
        texts_encoded = []
        for text in texts:
            encoded = np.array([ord(c) for c in text[:500]], dtype=np.int32)
            texts_encoded.append(encoded)
        
        max_len = max(len(t) for t in texts_encoded)
        texts_padded = np.zeros((len(texts_encoded), max_len), dtype=np.int32)
        for i, t in enumerate(texts_encoded):
            texts_padded[i, :len(t)] = t
        
        np.random.seed(42)
        a_params = np.random.randint(1, self.prime, size=num_perm, dtype=np.int64)
        b_params = np.random.randint(0, self.prime, size=num_perm, dtype=np.int64)
        
        signatures = create_signatures_numba(
            texts_padded,
            shingle_size,
            num_perm,
            self.prime,
            a_params,
            b_params
        )
        
        return signatures
    
    def _lsh_cluster(self, signatures, ids, num_perm, threshold):
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
    
    def _calculate_quality_metrics(self, cluster_map, n_items):
        cluster_sizes = defaultdict(int)
        for c in cluster_map.values():
            cluster_sizes[c] += 1
        
        n_clusters = len(cluster_sizes)
        
        if n_clusters < 2 or n_clusters >= n_items:
            return 0.0, {
                'n_clusters': n_clusters,
                'avg_cluster_size': 1.0,
                'singleton_ratio': 1.0
            }
        
        sizes = list(cluster_sizes.values())
        avg_size = np.mean(sizes)
        std_size = np.std(sizes)
        
        singleton_count = sum(1 for s in sizes if s == 1)
        singleton_ratio = singleton_count / n_clusters
        
        cluster_ratio = n_clusters / n_items
        
        size_score = 1.0 / (1.0 + std_size / avg_size) if avg_size > 0 else 0.0
        ratio_score = 1.0 - cluster_ratio
        singleton_penalty = 1.0 - singleton_ratio
        
        combined_score = (0.4 * size_score + 0.3 * ratio_score + 0.3 * singleton_penalty)
        
        return combined_score, {
            'n_clusters': n_clusters,
            'avg_cluster_size': avg_size,
            'singleton_ratio': singleton_ratio,
            'cluster_ratio': cluster_ratio
        }
    
    def _cluster_full_dataset(self):
        logger.info("Clustering full dataset with best parameters...")
        
        self.verse_clusters = self._cluster_config(
            self.df,
            self.best_verse_params['num_perm'],
            self.best_verse_params['shingle_size'],
            self.best_verse_params['threshold']
        )
        
        n_clusters = len(set(self.verse_clusters.values()))
        logger.info(f"Created {n_clusters:,} verse clusters")
    
    def _run_poem_clustering(self):
        start = time.time()
        logger.info("\nStep 3: Poem-level clustering with grid search")
        self._track_resources()
        
        poems = self._reconstruct_poems()
        
        if self.has_poem_gt:
            logger.info("Running supervised grid search on all poems")
            best_threshold = self._poem_grid_search_supervised(poems)
        else:
            logger.info("Running unsupervised grid search on 1% poem sample")
            best_threshold = self._poem_grid_search_unsupervised(poems)
        
        logger.info(f"\nBest poem threshold: {best_threshold:.2f}")
        logger.info("Clustering all poems with best threshold...")
        
        self.poem_clusters = self._cluster_poems(poems, best_threshold)
        
        n_poem_clusters = len(set(self.poem_clusters.values()))
        logger.info(f"Created {n_poem_clusters:,} poem clusters")
        
        self.timing['poem_clustering'] = time.time() - start
        self._track_resources()
        logger.info(f"Poem clustering completed in {self.timing['poem_clustering']:.2f}s")
    
    def _poem_grid_search_unsupervised(self, poems):
        all_poem_ids = list(poems.keys())
        sample_size = max(1000, int(len(all_poem_ids) * 0.01))
        sample_poem_ids = random.sample(all_poem_ids, sample_size)
        sample_poems = {pid: poems[pid] for pid in sample_poem_ids}
        
        logger.info(f"Testing {len(self.poem_thresholds)} thresholds on {len(sample_poems):,} poems ({len(sample_poems)/len(poems)*100:.2f}%)")
        
        results = []
        best_score = -np.inf
        best_threshold = self.poem_thresholds[0]
        
        for threshold in self.poem_thresholds:
            poem_clusters = self._cluster_poems(sample_poems, threshold)
            
            n_clusters = len(set(poem_clusters.values()))
            cluster_ratio = n_clusters / len(sample_poems)
            
            cluster_sizes = defaultdict(int)
            for c in poem_clusters.values():
                cluster_sizes[c] += 1
            
            sizes = list(cluster_sizes.values())
            avg_size = np.mean(sizes) if sizes else 1.0
            singleton_count = sum(1 for s in sizes if s == 1)
            singleton_ratio = singleton_count / n_clusters if n_clusters > 0 else 1.0
            
            ratio_score = 1.0 - cluster_ratio
            singleton_penalty = 1.0 - singleton_ratio
            score = 0.6 * ratio_score + 0.4 * singleton_penalty
            
            results.append({
                'threshold': threshold,
                'score': score,
                'n_clusters': n_clusters,
                'cluster_ratio': cluster_ratio,
                'avg_cluster_size': avg_size,
                'singleton_ratio': singleton_ratio
            })
            
            logger.info(f"  Threshold {threshold:.2f}: score={score:.4f}, clusters={n_clusters}, avg_size={avg_size:.1f}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / 'poem_grid_search_results.csv', index=False)
        
        return best_threshold
    
    def _poem_grid_search_supervised(self, poems):
        from sklearn.metrics import adjusted_rand_score
        
        logger.info(f"Testing {len(self.poem_thresholds)} thresholds on all {len(poems):,} poems")
        
        results = []
        best_ari = -np.inf
        best_threshold = self.poem_thresholds[0]
        
        for threshold in self.poem_thresholds:
            poem_clusters = self._cluster_poems(poems, threshold)
            
            poem_ids = [pid for pid in poems.keys() if poems[pid]['type_id'] is not None]
            true_labels = [poems[pid]['type_id'] for pid in poem_ids]
            pred_labels = [poem_clusters[pid] for pid in poem_ids]
            
            ari = adjusted_rand_score(true_labels, pred_labels)
            n_clusters = len(set(poem_clusters.values()))
            
            results.append({
                'threshold': threshold,
                'ari': ari,
                'n_clusters': n_clusters
            })
            
            logger.info(f"  Threshold {threshold:.2f}: ARI={ari:.4f}, clusters={n_clusters}")
            
            if ari > best_ari:
                best_ari = ari
                best_threshold = threshold
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / 'poem_grid_search_results.csv', index=False)
        
        return best_threshold
    
    def _reconstruct_poems(self):
        poems = {}
        for _, group in self.df.groupby('idoriginal_poem'):
            poem_id = group['idoriginal_poem'].iloc[0]  # Keep original type
            sorted_group = group.sort_values('order')
            
            verse_clusters = []
            for vid in sorted_group['id']:
                vid_str = str(vid)
                if vid_str in self.verse_clusters:
                    verse_clusters.append(self.verse_clusters[vid_str])
            
            poems[poem_id] = {
                'verse_clusters': set(verse_clusters),
                'type_id': group['type_id'].iloc[0] if 'type_id' in group.columns else None
            }
        
        logger.info(f"Reconstructed {len(poems):,} poems")
        return poems
    
    def _cluster_poems(self, poems, jaccard_threshold):
        poem_ids = [str(pid) for pid in poems.keys()]  
        n_poems = len(poem_ids)
        
        logger.info(f"Fast clustering {n_poems:,} poems (threshold={jaccard_threshold:.2f})")
        
        verse_to_poems = defaultdict(set)
        for poem_id, poem_data in poems.items():
            poem_id_str = str(poem_id)
            for vc in poem_data['verse_clusters']:
                verse_to_poems[vc].add(poem_id_str)
        
        logger.info("Finding candidate poem pairs...")
        candidate_pairs = set()
        
        processed_vcs = 0
        for vc, poem_set in verse_to_poems.items():
            if len(poem_set) > 1:
                poem_list = list(poem_set)
                if len(poem_list) <= 100:
                    for i in range(len(poem_list)):
                        for j in range(i + 1, len(poem_list)):
                            pair = (poem_list[i], poem_list[j]) if poem_list[i] < poem_list[j] else (poem_list[j], poem_list[i])
                            candidate_pairs.add(pair)
                else:
                    import random
                    for i in range(min(100, len(poem_list))):
                        for j in range(i + 1, min(i + 50, len(poem_list))):
                            pair = (poem_list[i], poem_list[j]) if poem_list[i] < poem_list[j] else (poem_list[j], poem_list[i])
                            candidate_pairs.add(pair)
            
            processed_vcs += 1
            if processed_vcs % 50000 == 0:
                logger.info(f"  Processed {processed_vcs:,} verse clusters, {len(candidate_pairs):,} candidate pairs so far")
        
        logger.info(f"Found {len(candidate_pairs):,} candidate poem pairs")
        
        logger.info("Computing Jaccard similarities...")
        edges = []
        
        poem_id_to_idx = {pid: idx for idx, pid in enumerate(poem_ids)}
        poems_str = {str(k): v for k, v in poems.items()}
        
        processed_pairs = 0
        for pid1, pid2 in candidate_pairs:
            if pid1 not in poems_str or pid2 not in poems_str:
                continue
                
            set1 = poems_str[pid1]['verse_clusters']
            set2 = poems_str[pid2]['verse_clusters']
            
            if len(set1) == 0 and len(set2) == 0:
                continue
            
            intersection = len(set1 & set2)
            union_size = len(set1 | set2)
            
            if union_size > 0:
                jaccard = intersection / union_size
                if jaccard >= jaccard_threshold:
                    idx1 = poem_id_to_idx[pid1]
                    idx2 = poem_id_to_idx[pid2]
                    edges.append((idx1, idx2))
            
            processed_pairs += 1
            if processed_pairs % 100000 == 0:
                logger.info(f"  Processed {processed_pairs:,}/{len(candidate_pairs):,} pairs, found {len(edges):,} similar pairs")
        
        logger.info(f"Found {len(edges):,} similar poem pairs")
        
        logger.info("Running Union-Find on poems...")
        if len(edges) > 0:
            edges_array = np.array(edges, dtype=np.int32)
            clusters_array = union_find_batch(edges_array, n_poems)
            cluster_map = {}
            for orig_pid in poems.keys():
                str_pid = str(orig_pid)
                if str_pid in poem_id_to_idx:
                    idx = poem_id_to_idx[str_pid]
                    cluster_map[orig_pid] = int(clusters_array[idx])
        else:
            cluster_map = {orig_pid: idx for idx, orig_pid in enumerate(poems.keys())}
        
        return cluster_map
    
    def _save_results(self):
        logger.info("\nStep 4: Saving results")
        self._track_resources()
        
        self.df['predicted_verse_cluster'] = self.df['id'].astype(str).map(self.verse_clusters)
        
        verse_output = self.output_dir / 'verse_clusters.csv'
        self.df.to_csv(verse_output, index=False)
        logger.info(f"Saved verse clusters to {verse_output}")
        
        poem_df = pd.DataFrame([
            {
                'idoriginal_poem': poem_id,
                'predicted_poem_cluster': cluster_id
            }
            for poem_id, cluster_id in self.poem_clusters.items()
        ])
        
        poem_output = self.output_dir / 'poem_clusters.csv'
        poem_df.to_csv(poem_output, index=False)
        logger.info(f"Saved poem clusters to {poem_output}")
        
        logger.info("\nCreating visualizations...")
        visualizer = Visualizer(
            output_dir=self.output_dir,
            has_verse_gt=self.has_verse_gt,
            has_poem_gt=self.has_poem_gt
        )
    
        visualizer.create_visualizations()

    def _generate_report(self):
        generator = ReportGenerator(
            df=self.df,
            verse_clusters=self.verse_clusters,
            poem_clusters=self.poem_clusters,
            best_verse_params=self.best_verse_params,
            timing=self.timing,
            resources=self.resources,
            output_dir=self.output_dir,
            format_stage_resources_fn=self._format_stage_resources
        )
        generator.generate()

    
    def _format_stage_resources(self, start_idx, end_idx):
        if not self.resources['cpu_percent']:
            return "N/A"
        
        cpu_slice = self.resources['cpu_percent'][start_idx:end_idx]
        mem_slice = self.resources['memory_used_gb'][start_idx:end_idx]
        
        if not cpu_slice:
            return "N/A"
        
        return f"CPU avg={np.mean(cpu_slice):.1f}%, Memory avg={np.mean(mem_slice):.2f} GB"