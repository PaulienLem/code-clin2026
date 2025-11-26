"""
Poem-level clustering based on verse-level cluster assignments.
Clusters poems that share similar verse cluster patterns.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PoemClusterer:
    """
    Clusters poems based on their verse-level cluster memberships.
    Uses Jaccard similarity between sets of verse clusters.
    """
    
    def __init__(self, similarity_threshold: float = 0.5):
        """
        Initialize poem clusterer.
        
        Args:
            similarity_threshold: Minimum Jaccard similarity for poem matching
        """
        self.similarity_threshold = similarity_threshold
        
    def compute_poem_cluster_sets(self, df: pd.DataFrame, 
                                  verse_clusters: Dict[str, int]) -> Dict[str, Set[int]]:
        """
        For each poem, get the set of verse clusters it contains.
        
        Args:
            df: DataFrame with verse data
            verse_clusters: Mapping from verse_id to cluster_id
            
        Returns:
            Dictionary mapping poem_id to set of verse cluster IDs
        """
        logger.info("Computing verse cluster sets for each poem")
        
        # Add cluster assignments to dataframe
        df_copy = df.copy()
        df_copy['predicted_cluster'] = df_copy['id'].map(verse_clusters)
        
        # Group by poem and collect unique verse clusters
        poem_cluster_sets = {}
        for poem_id, group in df_copy.groupby('idoriginal_poem'):
            cluster_set = set(group['predicted_cluster'].dropna().astype(int).values)
            poem_cluster_sets[poem_id] = cluster_set
        
        logger.info(f"Created cluster sets for {len(poem_cluster_sets)} poems")
        
        return poem_cluster_sets
    
    def compute_jaccard_similarity(self, set1: Set[int], set2: Set[int]) -> float:
        """
        Compute Jaccard similarity between two sets.
        
        Args:
            set1: First set
            set2: Second set
            
        Returns:
            Jaccard similarity (intersection / union)
        """
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def find_similar_poems(self, poem_cluster_sets: Dict[str, Set[int]]) -> Set[Tuple[str, str]]:
        """
        Find all pairs of poems with Jaccard similarity above threshold.
        
        Args:
            poem_cluster_sets: Dictionary mapping poem_id to verse cluster set
            
        Returns:
            Set of similar poem pairs
        """
        start_time = time.time()
        logger.info(f"Finding similar poems (threshold={self.similarity_threshold})")
        
        similar_pairs = set()
        poem_ids = list(poem_cluster_sets.keys())
        
        # Brute force comparison (optimized with early stopping)
        total_comparisons = len(poem_ids) * (len(poem_ids) - 1) // 2
        comparisons_done = 0
        
        for i, poem1 in enumerate(poem_ids):
            for poem2 in poem_ids[i+1:]:
                similarity = self.compute_jaccard_similarity(
                    poem_cluster_sets[poem1], 
                    poem_cluster_sets[poem2]
                )
                
                if similarity >= self.similarity_threshold:
                    pair = tuple(sorted([poem1, poem2]))
                    similar_pairs.add(pair)
                
                comparisons_done += 1
        
        elapsed = time.time() - start_time
        logger.info(f"Found {len(similar_pairs)} similar poem pairs in {elapsed:.2f}s "
                   f"({total_comparisons} comparisons, {total_comparisons/elapsed:.0f} comp/s)")
        
        return similar_pairs
    
    def cluster_poems_union_find(self, similar_pairs: Set[Tuple[str, str]], 
                                 all_poem_ids: List[str]) -> Dict[str, int]:
        """
        Cluster poems using union-find algorithm.
        
        Args:
            similar_pairs: Set of similar poem pairs
            all_poem_ids: List of all poem IDs
            
        Returns:
            Dictionary mapping poem_id to cluster_id
        """
        start_time = time.time()
        logger.info(f"Clustering {len(all_poem_ids)} poems using union-find")
        
        # Initialize union-find structure
        parent = {poem_id: poem_id for poem_id in all_poem_ids}
        rank = {poem_id: 0 for poem_id in all_poem_ids}
        
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
        for poem1, poem2 in similar_pairs:
            union(poem1, poem2)
        
        # Assign cluster IDs
        cluster_map = {}
        cluster_id = 0
        root_to_cluster = {}
        
        for poem_id in all_poem_ids:
            root = find(poem_id)
            if root not in root_to_cluster:
                root_to_cluster[root] = cluster_id
                cluster_id += 1
            cluster_map[poem_id] = root_to_cluster[root]
        
        elapsed = time.time() - start_time
        n_clusters = len(set(cluster_map.values()))
        logger.info(f"Created {n_clusters} poem clusters in {elapsed:.2f}s")
        
        return cluster_map
    
    def fit_predict(self, df: pd.DataFrame, 
                   verse_clusters: Dict[str, int]) -> Dict[str, int]:
        """
        Complete poem clustering pipeline.
        
        Args:
            df: DataFrame with verse data
            verse_clusters: Verse-level cluster assignments
            
        Returns:
            Dictionary mapping poem_id to cluster_id
        """
        # Get verse cluster sets for each poem
        poem_cluster_sets = self.compute_poem_cluster_sets(df, verse_clusters)
        
        # Find similar poems
        similar_pairs = self.find_similar_poems(poem_cluster_sets)
        
        # Cluster using union-find
        all_poem_ids = list(poem_cluster_sets.keys())
        poem_clusters = self.cluster_poems_union_find(similar_pairs, all_poem_ids)
        
        return poem_clusters
    
    def get_cluster_statistics(self, poem_clusters: Dict[str, int]) -> pd.DataFrame:
        """
        Get statistics about poem clusters.
        
        Args:
            poem_clusters: Poem cluster assignments
            
        Returns:
            DataFrame with cluster statistics
        """
        cluster_sizes = Counter(poem_clusters.values())
        
        stats = []
        for cluster_id, size in sorted(cluster_sizes.items()):
            stats.append({
                'cluster_id': cluster_id,
                'size': size
            })
        
        return pd.DataFrame(stats)