import numpy as np
import pandas as pd
import logging
import random
from collections import defaultdict
from pathlib import Path
from lsh_clusterer import union_find_batch

logger = logging.getLogger(__name__)


class PoemClusterer:
    def __init__(self, output_dir, poem_thresholds):
        self.output_dir = Path(output_dir)
        self.poem_thresholds = poem_thresholds
    
    def reconstruct_poems(self, df, verse_clusters):
        poems = {}
        for _, group in df.groupby('idoriginal_poem'):
            poem_id = group['idoriginal_poem'].iloc[0] 
            sorted_group = group.sort_values('order')
            
            verse_clusters_list = []
            for vid in sorted_group['id']:
                vid_str = str(vid)
                if vid_str in verse_clusters:
                    verse_clusters_list.append(verse_clusters[vid_str])
            
            poems[poem_id] = {
                'verse_clusters': set(verse_clusters_list),
                'type_id': group['type_id'].iloc[0] if 'type_id' in group.columns else None
            }
        
        logger.info(f"Reconstructed {len(poems):,} poems")
        return poems
    
    def grid_search_unsupervised(self, poems):
        all_poem_ids = list(poems.keys())
        sample_size = max(1000, int(len(all_poem_ids) * 0.01))
        sample_poem_ids = random.sample(all_poem_ids, sample_size)
        sample_poems = {pid: poems[pid] for pid in sample_poem_ids}
        
        logger.info(f"Testing {len(self.poem_thresholds)} thresholds on {len(sample_poems):,} poems ({len(sample_poems)/len(poems)*100:.2f}%)")
        
        results = []
        best_score = -np.inf
        best_threshold = self.poem_thresholds[0]
        
        for threshold in self.poem_thresholds:
            poem_clusters = self.cluster_poems(sample_poems, threshold)
            
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
    
    def grid_search_supervised(self, poems):
        from sklearn.metrics import adjusted_rand_score
        
        logger.info(f"Testing {len(self.poem_thresholds)} thresholds on all {len(poems):,} poems")
        
        results = []
        best_ari = -np.inf
        best_threshold = self.poem_thresholds[0]
        
        for threshold in self.poem_thresholds:
            poem_clusters = self.cluster_poems(poems, threshold)
            
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
    
    def cluster_poems(self, poems, jaccard_threshold):
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