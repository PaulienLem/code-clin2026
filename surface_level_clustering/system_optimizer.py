import logging
import psutil

logger = logging.getLogger(__name__)


class SystemOptimizer:
    def __init__(self, auto_optimize=True):
        self.cpu_count = psutil.cpu_count(logical=False)
        self.cpu_count_logical = psutil.cpu_count(logical=True)
        self.total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        self.optimal_n_jobs = None
        self.optimal_chunk_size = None
        self.batch_grid_search = None
        self.use_memory_aggressive = None
        
        if auto_optimize:
            self.optimize_for_system()
    
    def optimize_for_system(self):
        logger.info(f"\nSystem detected: {self.cpu_count} cores, {self.total_memory_gb:.0f} GB RAM")
        
        if self.cpu_count >= 64:
            self.optimal_n_jobs = int(self.cpu_count * 0.8) 
            self.optimal_chunk_size = 50000
            self.batch_grid_search = True
            logger.info(f"→ HPC mode: Using {self.optimal_n_jobs} cores")
        elif self.cpu_count >= 16:
            self.optimal_n_jobs = int(self.cpu_count * 0.9)
            self.optimal_chunk_size = 25000
            self.batch_grid_search = False
            logger.info(f"→ Workstation mode: Using {self.optimal_n_jobs} cores")
        else:
            self.optimal_n_jobs = max(1, self.cpu_count - 1)
            self.optimal_chunk_size = 10000
            self.batch_grid_search = False
            logger.info(f"→ Desktop mode: Using {self.optimal_n_jobs} cores")
        
        if self.total_memory_gb >= 256:
            self.optimal_chunk_size = 100000
            self.use_memory_aggressive = True
            logger.info(f"→ Large memory mode: {self.optimal_chunk_size} chunk size")
        elif self.total_memory_gb >= 64:
            self.use_memory_aggressive = False
            logger.info(f"→ Standard memory mode: {self.optimal_chunk_size} chunk size")
        else:
            self.optimal_chunk_size = min(self.optimal_chunk_size, 10000)
            self.use_memory_aggressive = False
            logger.info(f"→ Conservative memory mode: {self.optimal_chunk_size} chunk size")