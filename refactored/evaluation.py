"""
Evaluation metrics for clustering quality.
Supports both supervised (with ground truth) and unsupervised metrics.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusteringEvaluator:
    """
    Evaluates clustering quality using various metrics.
    """
    
    @staticmethod
    def compute_ari(true_labels: List[int], pred_labels: List[int]) -> float:
        """
        Compute Adjusted Rand Index.
        
        Args:
            true_labels: Ground truth cluster labels
            pred_labels: Predicted cluster labels
            
        Returns:
            ARI score (higher is better, max 1.0)
        """
        return adjusted_rand_score(true_labels, pred_labels)
    
    @staticmethod
    def compute_pairwise_similarity_features(texts: List[str], 
                                            clusters: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute pairwise similarity features for cohesion/separation metrics.
        This is a simplified version using character overlap.
        
        Args:
            texts: List of text strings
            clusters: List of cluster assignments
            
        Returns:
            Tuple of (similarities array, same_cluster boolean array)
        """
        n = len(texts)
        similarities = []
        same_cluster = []
        
        for i in range(n):
            for j in range(i + 1, n):
                # Simple character overlap similarity
                text1_chars = set(texts[i].lower())
                text2_chars = set(texts[j].lower())
                
                if len(text1_chars | text2_chars) > 0:
                    sim = len(text1_chars & text2_chars) / len(text1_chars | text2_chars)
                else:
                    sim = 0.0
                
                similarities.append(sim)
                same_cluster.append(clusters[i] == clusters[j])
        
        return np.array(similarities), np.array(same_cluster)
    
    @staticmethod
    def compute_cohesion(similarities: np.ndarray, same_cluster: np.ndarray) -> float:
        """
        Compute cluster cohesion (average similarity within clusters).
        
        Args:
            similarities: Array of pairwise similarities
            same_cluster: Boolean array indicating same cluster membership
            
        Returns:
            Cohesion score (higher is better)
        """
        within_cluster_sims = similarities[same_cluster]
        
        if len(within_cluster_sims) == 0:
            return 0.0
        
        return float(np.mean(within_cluster_sims))
    
    @staticmethod
    def compute_separation(similarities: np.ndarray, same_cluster: np.ndarray) -> float:
        """
        Compute cluster separation (average dissimilarity between clusters).
        
        Args:
            similarities: Array of pairwise similarities
            same_cluster: Boolean array indicating same cluster membership
            
        Returns:
            Separation score (higher is better)
        """
        between_cluster_sims = similarities[~same_cluster]
        
        if len(between_cluster_sims) == 0:
            return 1.0
        
        # Return 1 - avg_similarity for separation (higher is better)
        return float(1.0 - np.mean(between_cluster_sims))
    
    @staticmethod
    def compute_combined_unsupervised_score(cohesion: float, 
                                           separation: float,
                                           cohesion_weight: float = 0.6,
                                           separation_weight: float = 0.4) -> float:
        """
        Compute combined unsupervised quality score.
        
        Args:
            cohesion: Cohesion score
            separation: Separation score
            cohesion_weight: Weight for cohesion
            separation_weight: Weight for separation
            
        Returns:
            Combined score (higher is better)
        """
        return cohesion_weight * cohesion + separation_weight * separation
    
    @staticmethod
    def evaluate_supervised(true_labels: pd.Series, 
                           pred_labels: pd.Series) -> Dict[str, float]:
        """
        Evaluate clustering with ground truth.
        
        Args:
            true_labels: Ground truth cluster labels
            pred_labels: Predicted cluster labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Align labels
        common_idx = true_labels.index.intersection(pred_labels.index)
        
        logger.info(f"Evaluation alignment - True labels: {len(true_labels)}, "
                   f"Pred labels: {len(pred_labels)}, Common: {len(common_idx)}")
        
        if len(common_idx) == 0:
            logger.warning("No common indices between true and predicted labels!")
            logger.info(f"Sample true label indices: {true_labels.index[:5].tolist()}")
            logger.info(f"Sample pred label indices: {pred_labels.index[:5].tolist()}")
            return {
                'ari': 0.0,
                'n_true_clusters': 0,
                'n_pred_clusters': 0,
                'n_items': 0
            }
        
        true_aligned = true_labels.loc[common_idx].values
        pred_aligned = pred_labels.loc[common_idx].values
        
        # Remove any NaN values
        valid_mask = ~(pd.isna(true_aligned) | pd.isna(pred_aligned))
        true_aligned = true_aligned[valid_mask]
        pred_aligned = pred_aligned[valid_mask]
        
        if len(true_aligned) == 0:
            logger.warning("No valid (non-NaN) labels after alignment!")
            return {
                'ari': 0.0,
                'n_true_clusters': 0,
                'n_pred_clusters': 0,
                'n_items': 0
            }
        
        ari = adjusted_rand_score(true_aligned, pred_aligned)
        
        # Cluster statistics
        n_true_clusters = len(set(true_aligned))
        n_pred_clusters = len(set(pred_aligned))
        
        return {
            'ari': ari,
            'n_true_clusters': n_true_clusters,
            'n_pred_clusters': n_pred_clusters,
            'n_items': len(true_aligned)
        }
    
    @staticmethod
    def evaluate_unsupervised(texts: List[str], 
                             clusters: List[int],
                             sample_size: int = 5000) -> Dict[str, float]:
        """
        Evaluate clustering without ground truth.
        Uses cohesion and separation metrics on a sample for efficiency.
        
        Args:
            texts: List of text strings
            clusters: List of cluster assignments
            sample_size: Maximum number of items to sample for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Sample if dataset is too large
        if len(texts) > sample_size:
            indices = np.random.choice(len(texts), size=sample_size, replace=False)
            texts = [texts[i] for i in indices]
            clusters = [clusters[i] for i in indices]
        
        # Compute pairwise features
        similarities, same_cluster = ClusteringEvaluator.compute_pairwise_similarity_features(
            texts, clusters
        )
        
        # Compute metrics
        cohesion = ClusteringEvaluator.compute_cohesion(similarities, same_cluster)
        separation = ClusteringEvaluator.compute_separation(similarities, same_cluster)
        combined = ClusteringEvaluator.compute_combined_unsupervised_score(cohesion, separation)
        
        # Cluster statistics
        n_clusters = len(set(clusters))
        cluster_sizes = Counter(clusters)
        avg_cluster_size = np.mean(list(cluster_sizes.values()))
        
        return {
            'cohesion': cohesion,
            'separation': separation,
            'combined_score': combined,
            'n_clusters': n_clusters,
            'avg_cluster_size': avg_cluster_size,
            'n_items': len(texts)
        }
    
    @staticmethod
    def create_evaluation_report(metrics: Dict[str, float], 
                                metric_type: str = 'supervised') -> str:
        """
        Create a formatted evaluation report.
        
        Args:
            metrics: Dictionary with evaluation metrics
            metric_type: Type of evaluation ('supervised' or 'unsupervised')
            
        Returns:
            Formatted report string
        """
        report = f"\n{'='*60}\n"
        report += f"Clustering Evaluation Report ({metric_type})\n"
        report += f"{'='*60}\n"
        
        if metric_type == 'supervised':
            report += f"Adjusted Rand Index (ARI): {metrics['ari']:.4f}\n"
            report += f"Number of true clusters: {metrics['n_true_clusters']}\n"
            report += f"Number of predicted clusters: {metrics['n_pred_clusters']}\n"
            report += f"Number of items evaluated: {metrics['n_items']}\n"
        else:
            report += f"Cohesion: {metrics['cohesion']:.4f}\n"
            report += f"Separation: {metrics['separation']:.4f}\n"
            report += f"Combined Score: {metrics['combined_score']:.4f}\n"
            report += f"Number of clusters: {metrics['n_clusters']}\n"
            report += f"Average cluster size: {metrics['avg_cluster_size']:.2f}\n"
            report += f"Number of items evaluated: {metrics['n_items']}\n"
        
        report += f"{'='*60}\n"
        
        return report