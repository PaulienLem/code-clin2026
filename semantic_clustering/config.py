import os
import multiprocessing as mp

class Config:
    def __init__(self):
        self.model_name = 'kevinkrahn/shlm-grc-en'
        self.csv_path = 'concatenated.csv'
        self.output_dir = '/scratch/gent/vo/000/gvo00042/vsc48660'
        self.use_scratch = True
        
        self.subset_fraction = 0.01
        self.poem_subset_fraction = 0.01
        self.batch_size = 32
        self.max_length = 512
        
        self.verse_gt_col = 'idgroup'
        self.poem_gt_col = 'type_id'
        
        self.n_cores = max(1, mp.cpu_count() - 2)
        
        self.verse_similarity_thresholds = [0.7, 0.75, 0.8, 0.85, 0.9]
        self.leiden_resolutions = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        self.n_bootstraps = 3
        self.n_gridsearch_iterations = 5
        
        self.poem_similarity_thresholds = [0.3, 0.4, 0.5, 0.6, 0.66, 0.7, 0.8]
        
        self.faiss_nlist = 100
        self.faiss_nprobe = 10
        
        self.use_weight_scaling = True
        self.use_hub_penalty = True
        self.hub_threshold = 500
        self.use_cpm_leiden = True
        
        self.random_seed = 42
        
    def ensure_output_dir(self):
        if self.use_scratch and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        return self.output_dir if self.use_scratch else '.'