import numpy as np
import pandas as pd
import logging
import time
import random
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class GridSearchOptimizer:
    def __init__(self, output_dir, num_perms, shingle_sizes, thresholds):
        self.output_dir = Path(output_dir)
        self.num_perms = num_perms
        self.shingle_sizes = shingle_sizes
        self.thresholds = thresholds
        self.best_params = None
    
    def search_unsupervised(self, df, cluster_fn):
        sample_size = max(15000, int(len(df) * 0.01))
        sample_indices = np.random.choice(len(df), size=sample_size, replace=False)
        sample_df = df.iloc[sample_indices].copy()
        
        logger.info(f"Grid search on {len(sample_df):,} verses ({len(sample_df)/len(df)*100:.2f}%)")
        
        results = []
        best_score = -np.inf
        
        for num_perm in self.num_perms:
            for shingle_size in self.shingle_sizes:
                for threshold in self.thresholds:
                    config_start = time.time()
                    
                    logger.info(f"  Testing: perm={num_perm}, shingle={shingle_size}, thresh={threshold:.2f}")
                    
                    cluster_map = cluster_fn(sample_df, num_perm, shingle_size, threshold)
                    
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
                        self.best_params = {
                            'num_perm': num_perm,
                            'shingle_size': shingle_size,
                            'threshold': threshold
                        }
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / 'verse_grid_search_results.csv', index=False)
        return self.best_params
    
    def search_supervised(self, df, cluster_fn):
        from sklearn.metrics import adjusted_rand_score
        
        logger.info("Running supervised grid search with ARI")
        
        results = []
        best_score = -np.inf
        
        for num_perm in self.num_perms:
            for shingle_size in self.shingle_sizes:
                for threshold in self.thresholds:
                    config_start = time.time()
                    
                    logger.info(f"  Testing: perm={num_perm}, shingle={shingle_size}, thresh={threshold:.2f}")
                    
                    cluster_map = cluster_fn(df, num_perm, shingle_size, threshold)
                    
                    true_labels = df['idgroup'].tolist()
                    pred_labels = [cluster_map.get(str(vid), -1) for vid in df['id']]
                    
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
                        self.best_params = {
                            'num_perm': num_perm,
                            'shingle_size': shingle_size,
                            'threshold': threshold
                        }
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / 'verse_grid_search_results.csv', index=False)
        return self.best_params
    
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