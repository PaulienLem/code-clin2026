import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
import time

class Evaluator:
    def __init__(self, config):
        self.config = config
    
    def evaluate_clustering(self, df, predicted_clusters, cluster_col_name, gt_col):
        start = time.time()
        
        if gt_col not in df.columns:
            return {
                'has_ground_truth': False,
                'eval_time': 0.0
            }
        
        df_eval = df.copy()
        df_eval['predicted'] = df_eval['id'].map(predicted_clusters)
        df_eval = df_eval.dropna(subset=['predicted', gt_col])
        
        if len(df_eval) == 0:
            return {
                'has_ground_truth': True,
                'n_evaluated': 0,
                'eval_time': time.time() - start
            }
        
        y_true = df_eval[gt_col].values
        y_pred = df_eval['predicted'].values
        
        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        fmi = fowlkes_mallows_score(y_true, y_pred)
        
        eval_time = time.time() - start
        
        return {
            'has_ground_truth': True,
            'n_evaluated': len(df_eval),
            'ari': ari,
            'nmi': nmi,
            'fmi': fmi,
            'n_true_clusters': len(np.unique(y_true)),
            'n_pred_clusters': len(np.unique(y_pred)),
            'eval_time': eval_time
        }
