import igraph as ig
import leidenalg
import numpy as np
from sklearn.metrics import adjusted_rand_score
import time

class LeidenClustering:
    def __init__(self, config):
        self.config = config
    
    def build_igraph(self, pairs):
        edges = [(id1, id2) for id1, id2, sim in pairs]
        weights = [sim for id1, id2, sim in pairs]
        
        g = ig.Graph()
        all_nodes = set()
        for id1, id2, _ in pairs:
            all_nodes.add(id1)
            all_nodes.add(id2)
        
        node_list = sorted(list(all_nodes))
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        
        g.add_vertices(len(node_list))
        g.vs['name'] = node_list
        
        edge_list = [(node_to_idx[id1], node_to_idx[id2]) for id1, id2, _ in pairs]
        g.add_edges(edge_list)
        
        if self.config.use_weight_scaling and len(weights) > 0:
            w = np.array(weights)
            w_scaled = ((w - w.min()) / (w.max() - w.min() + 1e-10)) ** 3
            g.es['weight'] = w_scaled.tolist()
        else:
            g.es['weight'] = weights
        
        if self.config.use_hub_penalty:
            for v in range(g.vcount()):
                if g.degree(v) > self.config.hub_threshold:
                    for e in g.incident(v):
                        g.es[e]['weight'] *= 0.5
        
        return g
    
    def cluster_leiden(self, pairs, resolution):
        start = time.time()
        
        if len(pairs) == 0:
            return {
                'node_to_cluster': {},
                'n_clusters': 0,
                'cluster_time': time.time() - start
            }
        
        g = self.build_igraph(pairs)
        
        if self.config.use_cpm_leiden:
            partition = leidenalg.find_partition(
                g,
                leidenalg.CPMVertexPartition,
                weights=g.es['weight'],
                resolution_parameter=resolution,
                n_iterations=-1
            )
        else:
            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                weights=g.es['weight'],
                resolution_parameter=resolution,
                n_iterations=-1
            )
        
        node_to_cluster = {}
        for cluster_id, nodes_in_cluster in enumerate(partition):
            for node_idx in nodes_in_cluster:
                node_name = g.vs[node_idx]['name']
                node_to_cluster[node_name] = cluster_id
        
        cluster_time = time.time() - start
        
        return {
            'node_to_cluster': node_to_cluster,
            'n_clusters': len(partition),
            'modularity': partition.quality(),
            'cluster_time': cluster_time
        }
    
    def bootstrap_stability(self, pairs, resolution, n_bootstraps):
        start = time.time()
        
        if len(pairs) == 0:
            return {
                'stability': 0.0,
                'bootstrap_time': 0.0
            }
        
        g = self.build_igraph(pairs)
        
        partitions = []
        for _ in range(n_bootstraps):
            if self.config.use_cpm_leiden:
                partition = leidenalg.find_partition(
                    g,
                    leidenalg.CPMVertexPartition,
                    weights=g.es['weight'],
                    resolution_parameter=resolution,
                    n_iterations=-1
                )
            else:
                partition = leidenalg.find_partition(
                    g,
                    leidenalg.RBConfigurationVertexPartition,
                    weights=g.es['weight'],
                    resolution_parameter=resolution,
                    n_iterations=-1
                )
            
            membership = [0] * len(g.vs)
            for cluster_id, nodes_in_cluster in enumerate(partition):
                for node_idx in nodes_in_cluster:
                    membership[node_idx] = cluster_id
            partitions.append(membership)
        
        ari_scores = []
        for i in range(len(partitions)):
            for j in range(i + 1, len(partitions)):
                ari = adjusted_rand_score(partitions[i], partitions[j])
                ari_scores.append(ari)
        
        stability = np.mean(ari_scores) if ari_scores else 0.0
        bootstrap_time = time.time() - start
        
        return {
            'stability': stability,
            'bootstrap_time': bootstrap_time
        }
