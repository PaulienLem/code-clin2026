import numpy as np
import pandas as pd
from itertools import product
import time

class GridSearch:
    def __init__(self, config):
        self.config = config
    
    def verse_gridsearch(self, similarity_search, leiden_clustering, embeddings, id_mapping, df=None):
        start = time.time()
        
        results = []
        
        param_grid = list(product(
            self.config.verse_similarity_thresholds,
            self.config.leiden_resolutions
        ))
        
        total_iterations = min(self.config.n_gridsearch_iterations, len(param_grid))
        
        has_gt = df is not None and self.config.verse_gt_col in df.columns
        
        for i, (threshold, resolution) in enumerate(param_grid[:total_iterations]):
            iter_start = time.time()
            
            pairs_result = similarity_search.find_similar_pairs(threshold)
            pairs = pairs_result['pairs']
            
            clustering_result = leiden_clustering.cluster_leiden(pairs, resolution)
            
            stability_result = leiden_clustering.bootstrap_stability(
                pairs,
                resolution,
                self.config.n_bootstraps
            )
            
            result_dict = {
                'threshold': threshold,
                'resolution': resolution,
                'n_pairs': pairs_result['n_pairs'],
                'n_clusters': clustering_result['n_clusters'],
                'modularity': clustering_result.get('modularity', 0),
                'stability': stability_result['stability'],
                'iteration_time': time.time() - iter_start
            }
            
            if has_gt:
                from evaluation import Evaluator
                evaluator = Evaluator(self.config)
                eval_result = evaluator.evaluate_clustering(
                    df,
                    clustering_result['node_to_cluster'],
                    'verse_cluster',
                    self.config.verse_gt_col
                )
                result_dict['ari'] = eval_result.get('ari', 0)
                result_dict['nmi'] = eval_result.get('nmi', 0)
                result_dict['fmi'] = eval_result.get('fmi', 0)
            
            results.append(result_dict)
        
        total_time = time.time() - start
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            if has_gt and 'ari' in results_df.columns:
                best_idx = results_df['ari'].idxmax()
                best_params = results_df.iloc[best_idx]
            else:
                results_df['composite_score'] = (
                    0.5 * results_df['stability'] + 
                    0.3 * (results_df['n_clusters'] / results_df['n_clusters'].max()) +
                    0.2 * results_df['modularity']
                )
                best_idx = results_df['composite_score'].idxmax()
                best_params = results_df.iloc[best_idx]
        else:
            best_params = None
        
        return {
            'results': results_df,
            'best_params': best_params,
            'total_time': total_time,
            'has_ground_truth': has_gt
        }
    
    def poem_gridsearch(self, poem_to_clusters, df=None):
        start = time.time()
        
        results = []
        
        has_gt = df is not None and self.config.poem_gt_col in df.columns
        
        from poem_clustering import PoemClustering
        pc = PoemClustering(self.config)
        
        for threshold in self.config.poem_similarity_thresholds[:self.config.n_gridsearch_iterations]:
            iter_start = time.time()
            
            cluster_result = pc.cluster_poems_jaccard(poem_to_clusters, threshold)
            
            n_clusters = cluster_result['n_clusters']
            
            poem_ids = list(poem_to_clusters.keys())
            cluster_sizes = {}
            for pid in poem_ids:
                cluster_id = cluster_result['poem_to_cluster'].get(pid, -1)
                cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
            
            avg_cluster_size = np.mean(list(cluster_sizes.values())) if cluster_sizes else 0
            
            silhouette = 0.0
            
            result_dict = {
                'threshold': threshold,
                'n_pairs': cluster_result['n_pairs'],
                'n_clusters': n_clusters,
                'avg_cluster_size': avg_cluster_size,
                'silhouette': silhouette,
                'iteration_time': time.time() - iter_start
            }
            
            if has_gt:
                from evaluation import Evaluator
                evaluator = Evaluator(self.config)
                df_poems = df.drop_duplicates(subset=['idoriginal_poem']).copy()
                df_poems['id'] = df_poems['idoriginal_poem']
                eval_result = evaluator.evaluate_clustering(
                    df_poems,
                    cluster_result['poem_to_cluster'],
                    'poem_cluster',
                    self.config.poem_gt_col
                )
                result_dict['ari'] = eval_result.get('ari', 0)
                result_dict['nmi'] = eval_result.get('nmi', 0)
                result_dict['fmi'] = eval_result.get('fmi', 0)
            
            results.append(result_dict)
        
        total_time = time.time() - start
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            if has_gt and 'ari' in results_df.columns:
                best_idx = results_df['ari'].idxmax()
                best_params = results_df.iloc[best_idx]
            else:
                results_df['composite_score'] = (
                    0.3 * (results_df['n_clusters'] / results_df['n_clusters'].max() if results_df['n_clusters'].max() > 0 else 0) +
                    0.7 * (1.0 / (1.0 + np.abs(results_df['avg_cluster_size'] - 5)))
                )
                best_idx = results_df['composite_score'].idxmax()
                best_params = results_df.iloc[best_idx]
        else:
            best_params = None
        
        return {
            'results': results_df,
            'best_params': best_params,
            'total_time': total_time,
            'has_ground_truth': has_gt
        }