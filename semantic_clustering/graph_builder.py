import time

class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}
    
    def make_set(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
    
    def get_components(self):
        components = {}
        for node in self.parent:
            root = self.find(node)
            if root not in components:
                components[root] = []
            components[root].append(node)
        return list(components.values())

class GraphBuilder:
    def __init__(self, config):
        self.config = config
    
    def build_graph(self, pairs, all_ids):
        start = time.time()
        
        uf = UnionFind()
        
        for vertex_id in all_ids:
            uf.make_set(vertex_id)
        
        for id1, id2, sim in pairs:
            uf.union(id1, id2)
        
        components = uf.get_components()
        
        node_to_component = {}
        for comp_id, comp_nodes in enumerate(components):
            for node in comp_nodes:
                node_to_component[node] = comp_id
        
        build_time = time.time() - start
        
        return {
            'components': components,
            'node_to_component': node_to_component,
            'n_components': len(components),
            'build_time': build_time
        }
