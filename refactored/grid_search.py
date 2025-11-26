"""
Grid search for hyperparameter optimization.
Supports both supervised and unsupervised optimization.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from itertools import product
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from minhash_lsh import MinHashVectorizer, LSHClusterer
from evaluation import ClusteringEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GridSearchCV:
    """
    Grid search for MinHash LSH clustering hyperparameters.
    Supports both supervised (ARI) and unsupervised (cohesion+separation) optimization.
    """
    
    def __init__(self, 
                 shingle_sizes: List[int] = [2, 3, 4, 5],
                 thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9],
                 num_perm: int = 128,
                 n_jobs: int = -1):
        """
        Initialize grid search.
        
        Args:
            shingle_sizes: List of shingle sizes to try
            thresholds: List of Jaccard thresholds to try
            num_perm: Number of MinHash permutations
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.shingle_sizes = shingle_sizes
        self.thresholds = thresholds
        self.num_perm = num_perm
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        
        self.results_: Optional[pd.DataFrame] = None
        self.best_params_: Optional[Dict] = None
        self.best_score_: Optional[float] = None
        
    def _evaluate_params_supervised(self, 
                                    params: Tuple[int, float],
                                    texts: List[str],
                                    text_ids: List[str],
                                    true_labels: List[int]) -> Dict:
        """
        Evaluate a single parameter combination (supervised).
        
        Args:
            params: Tuple of (shingle_size, threshold)
            texts: List of texts
            text_ids: List of text IDs
            true_labels: Ground truth labels
            
        Returns:
            Dictionary with results
        """
        shingle_size, threshold = params
        
        try:
            # Create MinHash signatures
            vectorizer = MinHashVectorizer(shingle_size=shingle_size, 
                                          num_perm=self.num_perm)
            minhash_dict = vectorizer.create_minhash_batch(texts, text_ids, n_jobs=1)
            
            # Cluster
            clusterer = LSHClusterer(threshold=threshold, num_perm=self.num_perm)
            pred_clusters = clusterer.fit_predict(minhash_dict)
            
            # Evaluate - align predictions with ground truth
            # Only evaluate on items that have both predictions and ground truth
            aligned_pred = []
            aligned_true = []
            
            for i, tid in enumerate(text_ids):
                if tid in pred_clusters and i < len(true_labels):
                    # Check if true label is valid (not NaN)
                    if pd.notna(true_labels[i]):
                        aligned_pred.append(pred_clusters[tid])
                        aligned_true.append(true_labels[i])
            
            if len(aligned_pred) == 0:
                logger.warning(f"No valid predictions for params {params}")
                return {
                    'shingle_size': shingle_size,
                    'threshold': threshold,
                    'ari': 0.0,
                    'n_clusters': 0,
                    'success': False
                }
            
            ari = ClusteringEvaluator.compute_ari(aligned_true, aligned_pred)
            n_clusters = len(set(aligned_pred))
            
            return {
                'shingle_size': shingle_size,
                'threshold': threshold,
                'ari': ari,
                'n_clusters': n_clusters,
                'success': True
            }
        
        except Exception as e:
            logger.warning(f"Error with params {params}: {e}")
            return {
                'shingle_size': shingle_size,
                'threshold': threshold,
                'ari': -1.0,
                'n_clusters': 0,
                'success': False
            }
    
    def _evaluate_params_unsupervised(self,
                                     params: Tuple[int, float],
                                     texts: List[str],
                                     text_ids: List[str]) -> Dict:
        """
        Evaluate a single parameter combination (unsupervised).
        
        Args:
            params: Tuple of (shingle_size, threshold)
            texts: List of texts
            text_ids: List of text IDs
            
        Returns:
            Dictionary with results
        """
        shingle_size, threshold = params
        
        try:
            # Create MinHash signatures
            vectorizer = MinHashVectorizer(shingle_size=shingle_size, 
                                          num_perm=self.num_perm)
            minhash_dict = vectorizer.create_minhash_batch(texts, text_ids, n_jobs=1)
            
            # Cluster
            clusterer = LSHClusterer(threshold=threshold, num_perm=self.num_perm)
            pred_clusters = clusterer.fit_predict(minhash_dict)
            
            # Evaluate
            pred_labels = [pred_clusters[tid] for tid in text_ids]
            metrics = ClusteringEvaluator.evaluate_unsupervised(texts, pred_labels)
            
            return {
                'shingle_size': shingle_size,
                'threshold': threshold,
                'cohesion': metrics['cohesion'],
                'separation': metrics['separation'],
                'combined_score': metrics['combined_score'],
                'n_clusters': metrics['n_clusters'],
                'success': True
            }
        
        except Exception as e:
            logger.warning(f"Error with params {params}: {e}")
            return {
                'shingle_size': shingle_size,
                'threshold': threshold,
                'cohesion': 0.0,
                'separation': 0.0,
                'combined_score': 0.0,
                'n_clusters': 0,
                'success': False
            }
    
    def fit_supervised(self, 
                      texts: List[str],
                      text_ids: List[str],
                      true_labels: List[int]) -> 'GridSearchCV':
        """
        Run grid search with supervised evaluation (ARI).
        
        Args:
            texts: List of texts
            text_ids: List of text IDs
            true_labels: Ground truth cluster labels
            
        Returns:
            Self
        """
        start_time = time.time()
        param_grid = list(product(self.shingle_sizes, self.thresholds))
        
        logger.info(f"Starting supervised grid search with {len(param_grid)} combinations")
        logger.info(f"Shingle sizes: {self.shingle_sizes}")
        logger.info(f"Thresholds: {self.thresholds}")
        logger.info(f"Using {self.n_jobs} parallel jobs")
        
        results = []
        
        # Sequential processing (MinHash creation is already parallel)
        for params in param_grid:
            result = self._evaluate_params_supervised(params, texts, text_ids, true_labels)
            results.append(result)
            logger.info(f"  Params {params}: ARI={result['ari']:.4f}, "
                       f"n_clusters={result['n_clusters']}")
        
        # Store results
        self.results_ = pd.DataFrame(results)
        self.results_ = self.results_[self.results_['success']]
        
        # Find best parameters
        best_idx = self.results_['ari'].idxmax()
        self.best_params_ = {
            'shingle_size': int(self.results_.loc[best_idx, 'shingle_size']),
            'threshold': float(self.results_.loc[best_idx, 'threshold'])
        }
        self.best_score_ = float(self.results_.loc[best_idx, 'ari'])
        
        elapsed = time.time() - start_time
        logger.info(f"\nGrid search completed in {elapsed:.2f}s")
        logger.info(f"Best parameters: {self.best_params_}")
        logger.info(f"Best ARI: {self.best_score_:.4f}")
        
        return self
    
    def fit_unsupervised(self,
                        texts: List[str],
                        text_ids: List[str]) -> 'GridSearchCV':
        """
        Run grid search with unsupervised evaluation (cohesion+separation).
        
        Args:
            texts: List of texts
            text_ids: List of text IDs
            
        Returns:
            Self
        """
        start_time = time.time()
        param_grid = list(product(self.shingle_sizes, self.thresholds))
        
        logger.info(f"Starting unsupervised grid search with {len(param_grid)} combinations")
        logger.info(f"Shingle sizes: {self.shingle_sizes}")
        logger.info(f"Thresholds: {self.thresholds}")
        logger.info(f"Using {self.n_jobs} parallel jobs")
        
        results = []
        
        # Sequential processing (MinHash creation is already parallel)
        for params in param_grid:
            result = self._evaluate_params_unsupervised(params, texts, text_ids)
            results.append(result)
            logger.info(f"  Params {params}: combined_score={result['combined_score']:.4f}, "
                       f"n_clusters={result['n_clusters']}")
        
        # Store results
        self.results_ = pd.DataFrame(results)
        self.results_ = self.results_[self.results_['success']]
        
        # Find best parameters
        best_idx = self.results_['combined_score'].idxmax()
        self.best_params_ = {
            'shingle_size': int(self.results_.loc[best_idx, 'shingle_size']),
            'threshold': float(self.results_.loc[best_idx, 'threshold'])
        }
        self.best_score_ = float(self.results_.loc[best_idx, 'combined_score'])
        
        elapsed = time.time() - start_time
        logger.info(f"\nGrid search completed in {elapsed:.2f}s")
        logger.info(f"Best parameters: {self.best_params_}")
        logger.info(f"Best combined score: {self.best_score_:.4f}")
        
        return self
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get grid search results as DataFrame.
        
        Returns:
            DataFrame with all parameter combinations and scores
        """
        if self.results_ is None:
            raise ValueError("Grid search not run yet. Call fit_supervised() or fit_unsupervised() first.")
        
        return self.results_.copy()