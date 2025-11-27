import numpy as np
import time

class PoemClustering:
    def __init__(self, config):
        self.config = config
    
    def create_poem_representations(self, df, verse_clusters):
        start = time.time()
        
        df_with_clusters = df.copy()
        df_with_clusters['verse_cluster'] = df_with_clusters['id'].apply(
            lambda x: verse_clusters.get(x, -1)
        )
        df_with_clusters = df_with_clusters[df_with_clusters['verse_cluster'] != -1]
        
        poem_to_clusters = {}
        for poem_id, group in df_with_clusters.groupby('idoriginal_poem'):
            cluster_set = set(group['verse_cluster'].unique())
            poem_to_clusters[poem_id] = cluster_set
        
        create_time = time.time() - start
        
        return {
            'poem_to_clusters': poem_to_clusters,
            'n_poems': len(poem_to_clusters),
            'create_time': create_time
        }
    
    def calculate_jaccard_similarity(self, set_a, set_b):
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0
    
    def cluster_poems_jaccard(self, poem_to_clusters, threshold):
        start = time.time()
        
        poem_ids = list(poem_to_clusters.keys())
        n_poems = len(poem_ids)
        
        pairs = []
        for i in range(n_poems):
            for j in range(i + 1, n_poems):
                pid_a, pid_b = poem_ids[i], poem_ids[j]
                sim = self.calculate_jaccard_similarity(
                    poem_to_clusters[pid_a], 
                    poem_to_clusters[pid_b]
                )
                if sim >= threshold:
                    pairs.append((pid_a, pid_b, sim))
        
        from graph_builder import GraphBuilder
        gb = GraphBuilder(self.config)
        graph_result = gb.build_graph(pairs, poem_ids)
        
        cluster_time = time.time() - start
        
        return {
            'poem_to_cluster': graph_result['node_to_component'],
            'n_clusters': graph_result['n_components'],
            'n_pairs': len(pairs),
            'cluster_time': cluster_time
        }
    
    def create_poem_vectors(self, df, verse_clusters):
        start = time.time()
        
        df_with_clusters = df.copy()
        df_with_clusters['verse_cluster'] = df_with_clusters['id'].apply(
            lambda x: verse_clusters.get(x, -1)
        )
        df_with_clusters = df_with_clusters[df_with_clusters['verse_cluster'] != -1]
        
        all_verse_clusters = sorted(df_with_clusters['verse_cluster'].unique())
        cluster_to_idx = {cluster: idx for idx, cluster in enumerate(all_verse_clusters)}
        
        poem_vectors = []
        poem_ids = []
        
        for poem_id, group in df_with_clusters.groupby('idoriginal_poem'):
            verse_cluster_counts = group['verse_cluster'].value_counts()
            
            vector = np.zeros(len(all_verse_clusters))
            for cluster, count in verse_cluster_counts.items():
                vector[cluster_to_idx[cluster]] = count
            
            if vector.sum() > 0:
                vector = vector / vector.sum()
            
            poem_vectors.append(vector)
            poem_ids.append(poem_id)
        
        poem_vectors = np.array(poem_vectors)
        
        create_time = time.time() - start
        
        return {
            'poem_vectors': poem_vectors,
            'poem_ids': poem_ids,
            'n_poems': len(poem_ids),
            'vector_dim': poem_vectors.shape[1],
            'create_time': create_time
        }
    
    def cluster_poems(self, poem_vectors, poem_ids, threshold):
        start = time.time()
        
        n_poems = len(poem_vectors)
        similarities = np.dot(poem_vectors, poem_vectors.T)
        
        pairs = []
        for i in range(n_poems):
            for j in range(i + 1, n_poems):
                if similarities[i, j] >= threshold:
                    pairs.append((poem_ids[i], poem_ids[j], similarities[i, j]))
        
        from graph_builder import GraphBuilder
        gb = GraphBuilder(self.config)
        graph_result = gb.build_graph(pairs, poem_ids)
        
        cluster_time = time.time() - start
        
        return {
            'poem_to_cluster': graph_result['node_to_component'],
            'n_clusters': graph_result['n_components'],
            'n_pairs': len(pairs),
            'cluster_time': cluster_time
        }
