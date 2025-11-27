import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time
from typing import List, Dict, Optional, Tuple
import psutil
from visualizer import Visualizer
from report_generator import ReportGenerator
from signature_generator import SignatureGenerator
from lsh_clusterer import LSHClusterer
from grid_search_optimizer import GridSearchOptimizer
from poem_clusterer import PoemClusterer
from system_optimizer import SystemOptimizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CompletePipeline:
    def __init__(self,
                 csv_path: str,
                 scratch_dir: str = '/scratch/gent/vo/000/gvo00042/vsc48660',
                 output_dir: str = './results',
                 num_perms: Optional[List[int]] = None,
                 shingle_sizes: Optional[List[int]] = None,
                 thresholds: Optional[List[float]] = None,
                 poem_thresholds: Optional[List[float]] = None,
                 n_jobs: int = -1,
                 max_edges: int = 30000000,
                 auto_optimize: bool = True):
        
        self.csv_path = csv_path
        self.scratch_dir = Path(scratch_dir) / 'complete_clustering'
        self.output_dir = Path(output_dir)
        self.max_edges = max_edges
        self.auto_optimize = auto_optimize
        
        self.system_optimizer = SystemOptimizer(auto_optimize=auto_optimize)
        
        self.n_jobs = n_jobs if n_jobs != -1 else self.system_optimizer.optimal_n_jobs
        
        self.num_perms = num_perms if num_perms else [16, 24]
        self.shingle_sizes = shingle_sizes if shingle_sizes else [3, 4]
        self.thresholds = thresholds if thresholds else [0.75, 0.8, 0.85]
        self.poem_thresholds = poem_thresholds if poem_thresholds else [0.2, 0.3, 0.4, 0.5]
        
        self.prime = 2147483647
        
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.signature_generator = SignatureGenerator(prime=self.prime)
        self.lsh_clusterer = LSHClusterer(max_edges=max_edges)
        self.grid_search_optimizer = GridSearchOptimizer(
            output_dir=self.output_dir,
            num_perms=self.num_perms,
            shingle_sizes=self.shingle_sizes,
            thresholds=self.thresholds
        )
        self.poem_clusterer = PoemClusterer(
            output_dir=self.output_dir,
            poem_thresholds=self.poem_thresholds
        )
        
        self.df = None
        self.has_verse_gt = False
        self.has_poem_gt = False
        self.best_verse_params = None
        self.verse_clusters = None
        self.poem_clusters = None
        
        self.timing = {}
        self.resources = {
            'cpu_percent': [],
            'memory_percent': [],
            'memory_used_gb': []
        }
        self.process = psutil.Process()

    def _track_resources(self):
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        mem_process = self.process.memory_info().rss / (1024**3)  
        
        self.resources['cpu_percent'].append(cpu)
        self.resources['memory_percent'].append(mem.percent)
        self.resources['memory_used_gb'].append(mem_process)
        
    def run_full_pipeline(self):
        pipeline_start = time.time()
        logger.info("="*70)
        logger.info("COMPLETE PRODUCTION PIPELINE")
        logger.info("="*70)
        logger.info(f"Scratch: {self.scratch_dir}")
        logger.info(f"Verse grid search: {len(self.num_perms)}×{len(self.shingle_sizes)}×{len(self.thresholds)} = {len(self.num_perms)*len(self.shingle_sizes)*len(self.thresholds)} configs")
        logger.info(f"Poem grid search: {len(self.poem_thresholds)} thresholds")
        
        self._load_and_validate_data()
        self._run_verse_clustering()
        self._run_poem_clustering()
        self._save_results()
        
        total_time = time.time() - pipeline_start
        self.timing['total'] = total_time
        self._generate_report()
        
        logger.info(f"\nPipeline completed in {total_time/60:.1f} minutes")
    
    def _load_and_validate_data(self):
        start = time.time()
        logger.info("\nStep 1: Loading and validating data")
        self._track_resources()
        
        self.df = pd.read_csv(self.csv_path)
        
        required_cols = ['verse', 'idoriginal_poem', 'order']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        if 'id' not in self.df.columns:
            self.df['id'] = self.df.index.astype(str)
        
        self.df['verse'] = self.df['verse'].fillna('').astype(str)
        
        empty_mask = self.df['verse'].str.strip() == ''
        if empty_mask.any():
            logger.warning(f"Removing {empty_mask.sum():,} empty verses")
            self.df = self.df[~empty_mask].reset_index(drop=True)
        
        self.has_verse_gt = 'idgroup' in self.df.columns
        self.has_poem_gt = 'type_id' in self.df.columns
        
        if self.has_verse_gt:
            nan_mask = self.df['idgroup'].isna()
            if nan_mask.any():
                logger.warning(f"Removing {nan_mask.sum():,} verses with NaN ground truth")
                self.df = self.df[~nan_mask].reset_index(drop=True)
        
        logger.info(f"Loaded {len(self.df):,} verses from {self.df['idoriginal_poem'].nunique():,} poems")
        logger.info(f"Has verse ground truth: {self.has_verse_gt}")
        logger.info(f"Has poem ground truth: {self.has_poem_gt}")
        
        self.timing['load'] = time.time() - start
        self._track_resources()
        
    def _run_verse_clustering(self):
        start = time.time()
        logger.info("\nStep 2: Verse-level clustering with grid search")
        self._track_resources()
        
        if self.has_verse_gt:
            logger.info("Running supervised grid search on full data")
            self.best_verse_params = self.grid_search_optimizer.search_supervised(
                self.df, 
                self._cluster_config
            )
        else:
            logger.info("Running unsupervised grid search on sample")
            self.best_verse_params = self.grid_search_optimizer.search_unsupervised(
                self.df, 
                self._cluster_config
            )
        
        logger.info(f"\nBest verse parameters: {self.best_verse_params}")
        
        logger.info("\nClustering full dataset with best parameters...")
        self._cluster_full_dataset()
        
        self.timing['verse_clustering'] = time.time() - start
        self._track_resources()
        logger.info(f"Verse clustering completed in {self.timing['verse_clustering']/60:.1f} minutes")
    
    def _cluster_config(self, df, num_perm, shingle_size, threshold):
        texts = df['verse'].str.lower().tolist()
        ids = df['id'].astype(str).tolist()
        
        signatures = self.signature_generator.create_signatures(texts, num_perm, shingle_size)
        
        cluster_map = self.lsh_clusterer.cluster(signatures, ids, num_perm, threshold)
        
        return cluster_map
    
    def _cluster_full_dataset(self):
        logger.info("Clustering full dataset with best parameters...")
        
        self.verse_clusters = self._cluster_config(
            self.df,
            self.best_verse_params['num_perm'],
            self.best_verse_params['shingle_size'],
            self.best_verse_params['threshold']
        )
        
        n_clusters = len(set(self.verse_clusters.values()))
        logger.info(f"Created {n_clusters:,} verse clusters")
    
    def _run_poem_clustering(self):
        start = time.time()
        logger.info("\nStep 3: Poem-level clustering with grid search")
        self._track_resources()
        
        poems = self.poem_clusterer.reconstruct_poems(self.df, self.verse_clusters)
        
        if self.has_poem_gt:
            logger.info("Running supervised grid search on all poems")
            best_threshold = self.poem_clusterer.grid_search_supervised(poems)
        else:
            logger.info("Running unsupervised grid search on 1% poem sample")
            best_threshold = self.poem_clusterer.grid_search_unsupervised(poems)
        
        logger.info(f"\nBest poem threshold: {best_threshold:.2f}")
        logger.info("Clustering all poems with best threshold...")
        
        self.poem_clusters = self.poem_clusterer.cluster_poems(poems, best_threshold)
        
        n_poem_clusters = len(set(self.poem_clusters.values()))
        logger.info(f"Created {n_poem_clusters:,} poem clusters")
        
        self.timing['poem_clustering'] = time.time() - start
        self._track_resources()
        logger.info(f"Poem clustering completed in {self.timing['poem_clustering']:.2f}s")
    
    def _save_results(self):
        logger.info("\nStep 4: Saving results")
        self._track_resources()
        
        self.df['predicted_verse_cluster'] = self.df['id'].astype(str).map(self.verse_clusters)
        
        verse_output = self.output_dir / 'verse_clusters.csv'
        self.df.to_csv(verse_output, index=False)
        logger.info(f"Saved verse clusters to {verse_output}")
        
        poem_df = pd.DataFrame([
            {
                'idoriginal_poem': poem_id,
                'predicted_poem_cluster': cluster_id
            }
            for poem_id, cluster_id in self.poem_clusters.items()
        ])
        
        poem_output = self.output_dir / 'poem_clusters.csv'
        poem_df.to_csv(poem_output, index=False)
        logger.info(f"Saved poem clusters to {poem_output}")
        
        logger.info("\nCreating visualizations...")
        visualizer = Visualizer(
            output_dir=self.output_dir,
            has_verse_gt=self.has_verse_gt,
            has_poem_gt=self.has_poem_gt
        )
    
        visualizer.create_visualizations()

    def _generate_report(self):
        generator = ReportGenerator(
            df=self.df,
            verse_clusters=self.verse_clusters,
            poem_clusters=self.poem_clusters,
            best_verse_params=self.best_verse_params,
            timing=self.timing,
            resources=self.resources,
            output_dir=self.output_dir,
            format_stage_resources_fn=self._format_stage_resources
        )
        generator.generate()

    
    def _format_stage_resources(self, start_idx, end_idx):
        if not self.resources['cpu_percent']:
            return "N/A"
        
        cpu_slice = self.resources['cpu_percent'][start_idx:end_idx]
        mem_slice = self.resources['memory_used_gb'][start_idx:end_idx]
        
        if not cpu_slice:
            return "N/A"
        
        return f"CPU avg={np.mean(cpu_slice):.1f}%, Memory avg={np.mean(mem_slice):.2f} GB"