import faiss
import numpy as np
import time

class SimilaritySearch:
    def __init__(self, config):
        self.config = config
        self.index = None
        self.embeddings = None
        self.id_mapping = None
        
    def build_index(self, embeddings, id_mapping):
        start = time.time()
        self.embeddings = embeddings.astype('float32')
        self.id_mapping = id_mapping
        
        faiss.normalize_L2(self.embeddings)
        
        d = self.embeddings.shape[1]
        
        if len(self.embeddings) > 1000:
            quantizer = faiss.IndexFlatIP(d)
            nlist = min(self.config.faiss_nlist, len(self.embeddings) // 10)
            self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(self.embeddings)
            self.index.add(self.embeddings)
            self.index.nprobe = min(self.config.faiss_nprobe, nlist)
        else:
            self.index = faiss.IndexFlatIP(d)
            self.index.add(self.embeddings)
        
        build_time = time.time() - start
        
        return {
            'index_type': type(self.index).__name__,
            'n_vectors': len(self.embeddings),
            'dimension': d,
            'build_time': build_time
        }
    
    def find_similar_pairs(self, threshold):
        start = time.time()
        
        k = min(100, len(self.embeddings))
        similarities, indices = self.index.search(self.embeddings, k)
        
        pairs = []
        for i in range(len(self.embeddings)):
            for j, sim in zip(indices[i], similarities[i]):
                if i < j and sim >= threshold:
                    pairs.append((self.id_mapping[i], self.id_mapping[j], sim))
        
        search_time = time.time() - start
        
        return {
            'pairs': pairs,
            'n_pairs': len(pairs),
            'search_time': search_time
        }
