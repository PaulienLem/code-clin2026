import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
import time
from pathlib import Path

from data_loader import VerseDataLoader
from minhash_lsh import MinHashVectorizer, LSHClusterer
from poem_clustering import PoemClusterer
from evaluation import ClusteringEvaluator
from grid_search import GridSearchCV
from visualization import ClusteringVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusteringPipeline:
    def __init__(self, 
                 csv_path: str,
                 output_dir: str = './clustering_results',
                 num_perm: int = 128,
                 n_jobs: int = -1,
                 shingle_sizes: Optional[list] = None,
                 thresholds: Optional[list] = None,
                 poem_thresholds: Optional[list] = None):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.num_perm = num_perm
        self.n_jobs = n_jobs
        self.shingle_sizes = shingle_sizes if shingle_sizes is not None else [2, 3, 4, 5]
        self.thresholds = thresholds if thresholds is not None else [0.5, 0.6, 0.7, 0.8, 0.9]
        self.poem_thresholds = poem_thresholds if poem_thresholds is not None else [0.3, 0.4, 0.5, 0.6, 0.7]
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.data_loader = VerseDataLoader(csv_path)
        self.df: Optional[pd.DataFrame] = None
        
        self.verse_grid_results: Optional[pd.DataFrame] = None
        self.poem_grid_results: Optional[pd.DataFrame] = None
        self.verse_clusters: Optional[Dict[str, int]] = None
        self.poem_clusters: Optional[Dict[str, int]] = None
        self.best_verse_params: Optional[Dict] = None
        self.best_poem_params: Optional[Dict] = None
        self.timing_info: Dict[str, float] = {}

    def run_full_pipeline(self):
        pipeline_start = time.time()
        logger.info("Starting verse and poem clustering pipeline")
        self._load_and_validate_data()
        self._run_verse_clustering()
        self._run_poem_clustering()
        self._create_visualizations()
        self._save_results()
        total_time = time.time() - pipeline_start
        self.timing_info['total_pipeline'] = total_time
        self._generate_report()
        logger.info(f"Pipeline completed in {total_time:.2f}s")
        logger.info(f"Results saved to: {self.output_dir}")

    def _load_and_validate_data(self):
        start_time = time.time()
        logger.info("Step 1: Loading and validating data")
        self.df = self.data_loader.load_data()
        stats = self.data_loader.get_statistics()
        logger.info("Dataset statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        self.timing_info['data_loading'] = time.time() - start_time
        logger.info(f"Data loading completed in {self.timing_info['data_loading']:.2f}s")

    def _run_verse_clustering(self):
        start_time = time.time()
        logger.info("Step 2: Verse-level clustering with grid search")
        valid_mask = self.df['verse'].notna() & (self.df['verse'].str.strip() != '')
        df_valid = self.df[valid_mask].copy()
        texts = df_valid['verse'].tolist()
        text_ids = df_valid['id'].astype(str).tolist()
        logger.info(f"Processing {len(texts)} valid verses")
        logger.info(f"Grid search configuration: shingle_sizes={self.shingle_sizes}, thresholds={self.thresholds}")
        grid_search = GridSearchCV(
            shingle_sizes=self.shingle_sizes,
            thresholds=self.thresholds,
            num_perm=self.num_perm,
            n_jobs=self.n_jobs
        )
        if self.data_loader.has_verse_gt:
            logger.info("Running supervised grid search (optimizing ARI)")
            true_labels = df_valid['idgroup'].tolist()
            grid_search.fit_supervised(texts, text_ids, true_labels)
        else:
            logger.info("Running unsupervised grid search on 1% sample")
            sample_df = self.data_loader.get_sample(fraction=0.01)
            sample_df = sample_df[sample_df['verse'].notna() & (sample_df['verse'].str.strip() != '')].copy()
            sample_texts = sample_df['verse'].tolist()
            sample_ids = sample_df['id'].astype(str).tolist()
            grid_search.fit_unsupervised(sample_texts, sample_ids)
        self.verse_grid_results = grid_search.get_results_dataframe()
        self.best_verse_params = grid_search.best_params_
        
        logger.info(f"Applying best parameters to full dataset: {self.best_verse_params}")
        vectorizer = MinHashVectorizer(
            shingle_size=self.best_verse_params['shingle_size'],
            num_perm=self.num_perm
        )
        minhash_dict = vectorizer.create_minhash_batch(texts, text_ids, n_jobs=1)
        clusterer = LSHClusterer(
            threshold=self.best_verse_params['threshold'],
            num_perm=self.num_perm
        )
        self.verse_clusters = clusterer.fit_predict(minhash_dict)
        df_valid['predicted_verse_cluster'] = df_valid['id'].astype(str).map(self.verse_clusters)
        self.df = df_valid
        if self.data_loader.has_verse_gt:
            pred_labels_list = []
            true_labels_list = []
            for idx, row in self.df.iterrows():
                verse_id = str(row['id'])
                if verse_id in self.verse_clusters and pd.notna(row['idgroup']):
                    pred_labels_list.append(self.verse_clusters[verse_id])
                    true_labels_list.append(row['idgroup'])
            if pred_labels_list:
                from sklearn.metrics import adjusted_rand_score
                ari = adjusted_rand_score(true_labels_list, pred_labels_list)
                logger.info(f"Final verse-level evaluation: ARI={ari:.4f}")
        self.timing_info['verse_clustering'] = time.time() - start_time
        logger.info(f"Verse clustering completed in {self.timing_info['verse_clustering']:.2f}s")

    def _run_poem_clustering(self):
        start_time = time.time()
        logger.info("Step 3: Poem-level clustering with grid search")
        poem_results = []
        for threshold in self.poem_thresholds:
            clusterer = PoemClusterer(similarity_threshold=threshold)
            poem_clusters_temp = clusterer.fit_predict(self.df, self.verse_clusters)
            if self.data_loader.has_poem_gt:
                poems_df = self.data_loader.get_poems_dataframe()
                pred_labels_series = pd.Series(poem_clusters_temp)
                true_labels_series = poems_df.set_index('idoriginal_poem')['type_id']
                metrics = ClusteringEvaluator.evaluate_supervised(true_labels_series, pred_labels_series)
                score = metrics['ari']
                metric_name = 'ari'
            else:
                n_clusters = len(set(poem_clusters_temp.values()))
                score = n_clusters
                metric_name = 'n_clusters'
            poem_results.append({'threshold': threshold, metric_name: score, 'n_clusters': len(set(poem_clusters_temp.values()))})
            logger.info(f"Threshold {threshold}: {metric_name}={score:.4f}, n_clusters={len(set(poem_clusters_temp.values()))}")
        self.poem_grid_results = pd.DataFrame(poem_results)
        best_idx = self.poem_grid_results['ari'].idxmax() if self.data_loader.has_poem_gt else len(poem_results)//2
        best_threshold = self.poem_grid_results.loc[best_idx, 'threshold']
        self.best_poem_params = {'threshold': best_threshold}
        logger.info(f"Best poem similarity threshold: {best_threshold}")
        clusterer = PoemClusterer(similarity_threshold=best_threshold)
        self.poem_clusters = clusterer.fit_predict(self.df, self.verse_clusters)
        self.timing_info['poem_clustering'] = time.time() - start_time
        logger.info(f"Poem clustering completed in {self.timing_info['poem_clustering']:.2f}s")

    def _create_visualizations(self):
        start_time = time.time()
        verse_metric = 'ari' if self.data_loader.has_verse_gt else 'combined_score'
        poem_metric = 'ari' if self.data_loader.has_poem_gt else 'n_clusters'
        ClusteringVisualizer.create_all_visualizations(
            self.verse_grid_results,
            self.poem_grid_results,
            verse_metric,
            poem_metric,
            output_dir=self.output_dir
        )
        self.timing_info['visualization'] = time.time() - start_time
        logger.info(f"Visualizations created in {self.timing_info['visualization']:.2f}s")

    def _save_results(self):
        logger.info("Step 5: Saving results")
        verse_results_df = self.df.copy()
        verse_results_df['predicted_verse_cluster'] = verse_results_df['id'].astype(str).map(self.verse_clusters)
        verse_results_df.to_csv(f"{self.output_dir}/verse_clusters.csv", index=False)
        poems_df = self.data_loader.get_poems_dataframe()
        poems_df['predicted_poem_cluster'] = poems_df['idoriginal_poem'].astype(str).map(self.poem_clusters)
        poems_df.to_csv(f"{self.output_dir}/poem_clusters.csv", index=False)
        self.verse_grid_results.to_csv(f"{self.output_dir}/verse_grid_search_results.csv", index=False)
        self.poem_grid_results.to_csv(f"{self.output_dir}/poem_grid_search_results.csv", index=False)

    def _generate_report(self):
        report = []
        report.append("Clustering pipeline performance report")
        report.append(f"Input: {self.csv_path}")
        report.append(f"Output: {self.output_dir}")
        report.append(f"Total verses: {len(self.df)}")
        report.append(f"Total poems: {self.df['idoriginal_poem'].nunique()}")
        report.append(f"Parallel jobs: {self.n_jobs}")
        report.append(f"MinHash permutations: {self.num_perm}")
        for step, duration in self.timing_info.items():
            report.append(f"{step}: {duration:.2f}s")
        report.append(f"Best verse-level params: {self.best_verse_params}")
        report.append(f"Best poem-level params: {self.best_poem_params}")
        report.append(f"Verse clusters: {len(set(self.verse_clusters.values()))}")
        report.append(f"Poem clusters: {len(set(self.poem_clusters.values()))}")
        report_text = "\n".join(report)
        with open(f"{self.output_dir}/performance_report.txt", 'w') as f:
            f.write(report_text)
        logger.info(report_text)
