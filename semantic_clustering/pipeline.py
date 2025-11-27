# import warnings
# warnings.filterwarnings('ignore')

# from config import Config
# from data_loader import DataLoader
# from embedding import EmbeddingGenerator
# from similarity_search import SimilaritySearch
# from graph_builder import GraphBuilder
# from leiden_clustering import LeidenClustering
# from grid_search import GridSearch
# from poem_clustering import PoemClustering
# from evaluation import Evaluator
# from visualization import Visualizer
# from performance_monitor import PerformanceMonitor

# class Pipeline:
#     def __init__(self, config):
#         self.config = config
#         self.perf_monitor = PerformanceMonitor(config)
        
#         self.data_loader = DataLoader(config)
#         self.embedding_gen = EmbeddingGenerator(config)
#         self.similarity_search = SimilaritySearch(config)
#         self.graph_builder = GraphBuilder(config)
#         self.leiden_clustering = LeidenClustering(config)
#         self.grid_search = GridSearch(config)
#         self.poem_clustering = PoemClustering(config)
#         self.evaluator = Evaluator(config)
#         self.visualizer = Visualizer(config)
        
#         self.verse_clusters = None
#         self.poem_clusters = None
        
#     def run_verse_clustering_on_subset(self):
#         print('\nstep 1: loading data')
#         load_result = self.data_loader.load_data()
#         self.perf_monitor.record('data_loading', load_result)
#         print(f"loaded {load_result['n_verses']} verses from {load_result['n_poems']} poems")
        
#         print('\nstep 2: creating stratified subset')
#         subset_result = self.data_loader.get_stratified_subset()
#         self.perf_monitor.record('stratified_sampling', subset_result)
#         print(f"created subset with {subset_result['subset_size']} verses")
        
#         print('\nstep 3: loading embedding model')
#         model_result = self.embedding_gen.load_model()
#         self.perf_monitor.record('model_loading', model_result)
#         print(f"loaded model on {model_result['device']}")
        
#         print('\nstep 4: embedding verses')
#         subset_df = self.data_loader.get_subset_data()
#         embed_result = self.embedding_gen.embed_texts(
#             subset_df['verse'].tolist(),
#             subset_df['id'].tolist()
#         )
#         self.perf_monitor.record('embedding', embed_result)
#         print(f"generated embeddings with dimension {embed_result['embedding_dim']}")
        
#         print('\nstep 5: building faiss index')
#         index_result = self.similarity_search.build_index(
#             embed_result['embeddings'],
#             embed_result['id_mapping']
#         )
#         self.perf_monitor.record('faiss_index', index_result)
#         print(f"built {index_result['index_type']} index")
        
#         print('\nstep 6: running verse gridsearch')
#         subset_df = self.data_loader.get_subset_data()
#         gridsearch_result = self.grid_search.verse_gridsearch(
#             self.similarity_search,
#             self.leiden_clustering,
#             embed_result['embeddings'],
#             embed_result['id_mapping'],
#             subset_df
#         )
#         self.perf_monitor.record('verse_gridsearch', {
#             'n_configurations': len(gridsearch_result['results']),
#             'total_time': gridsearch_result['total_time'],
#             'has_ground_truth': gridsearch_result.get('has_ground_truth', False)
#         })
        
#         print('\nverse gridsearch results:')
#         print(gridsearch_result['results'])
        
#         if gridsearch_result['best_params'] is not None:
#             print('\nbest parameters:')
#             print(gridsearch_result['best_params'])
        
#         self.visualizer.plot_verse_gridsearch(gridsearch_result['results'])
#         print(f"saved verse gridsearch visualization")
        
#         return gridsearch_result['best_params']
    
#     def run_full_verse_clustering(self, best_threshold, best_resolution):
#         print('\nstep 7: embedding full dataset')
#         full_df = self.data_loader.get_full_data()
#         embed_result = self.embedding_gen.embed_texts(
#             full_df['verse'].tolist(),
#             full_df['id'].tolist()
#         )
        
#         print('\nstep 8: building full faiss index')
#         self.similarity_search.build_index(
#             embed_result['embeddings'],
#             embed_result['id_mapping']
#         )
        
#         print(f'\nstep 9: finding similar pairs with threshold {best_threshold}')
#         pairs_result = self.similarity_search.find_similar_pairs(best_threshold)
#         print(f"found {pairs_result['n_pairs']} similar pairs")
        
#         print(f'\nstep 10: clustering with leiden resolution {best_resolution}')
#         clustering_result = self.leiden_clustering.cluster_leiden(
#             pairs_result['pairs'],
#             best_resolution
#         )
#         print(f"found {clustering_result['n_clusters']} verse clusters")
        
#         self.perf_monitor.record('full_verse_clustering', {
#             'n_pairs': pairs_result['n_pairs'],
#             'n_clusters': clustering_result['n_clusters'],
#             'modularity': clustering_result.get('modularity', 0),
#             'total_time': pairs_result['search_time'] + clustering_result['cluster_time']
#         })
        
#         self.verse_clusters = clustering_result['node_to_cluster']
        
#         print('\nstep 11: evaluating verse clustering')
#         eval_result = self.evaluator.evaluate_clustering(
#             full_df,
#             self.verse_clusters,
#             'verse_cluster',
#             self.config.verse_gt_col
#         )
#         self.perf_monitor.record('verse_evaluation', eval_result)
        
#         if eval_result['has_ground_truth']:
#             print(f"verse clustering evaluation:")
#             print(f"  ari: {eval_result.get('ari', 0):.4f}")
#             print(f"  nmi: {eval_result.get('nmi', 0):.4f}")
#             print(f"  fmi: {eval_result.get('fmi', 0):.4f}")
        
#         return self.verse_clusters
    
#     def run_poem_clustering_on_subset(self):
#         print('\nstep 12: creating poem representations from verse clusters')
        
#         full_df = self.data_loader.get_full_data()
#         subset_df = full_df.sample(
#             frac=self.config.poem_subset_fraction, 
#             random_state=self.config.random_seed
#         )
        
#         poem_rep_result = self.poem_clustering.create_poem_representations(
#             subset_df,
#             self.verse_clusters
#         )
#         self.perf_monitor.record('poem_representation_creation', poem_rep_result)
#         print(f"created representations for {poem_rep_result['n_poems']} poems")
        
#         print('\nstep 13: running poem gridsearch')
#         poem_gridsearch_result = self.grid_search.poem_gridsearch(
#             poem_rep_result['poem_to_clusters'],
#             subset_df
#         )
#         self.perf_monitor.record('poem_gridsearch', {
#             'n_configurations': len(poem_gridsearch_result['results']),
#             'total_time': poem_gridsearch_result['total_time'],
#             'has_ground_truth': poem_gridsearch_result.get('has_ground_truth', False)
#         })
        
#         print('\npoem gridsearch results:')
#         print(poem_gridsearch_result['results'])
        
#         if poem_gridsearch_result['best_params'] is not None:
#             print('\nbest parameters:')
#             print(poem_gridsearch_result['best_params'])
        
#         self.visualizer.plot_poem_gridsearch(poem_gridsearch_result['results'])
#         print(f"saved poem gridsearch visualization")
        
#         return poem_gridsearch_result['best_params']
    
#     def run_full_poem_clustering(self, best_threshold):
#         print(f'\nstep 14: clustering all poems with threshold {best_threshold}')
#         full_df = self.data_loader.get_full_data()
#         poem_rep_result = self.poem_clustering.create_poem_representations(
#             full_df,
#             self.verse_clusters
#         )
        
#         poem_cluster_result = self.poem_clustering.cluster_poems_jaccard(
#             poem_rep_result['poem_to_clusters'],
#             best_threshold
#         )
#         print(f"found {poem_cluster_result['n_clusters']} poem clusters")
        
#         self.perf_monitor.record('full_poem_clustering', poem_cluster_result)
        
#         self.poem_clusters = poem_cluster_result['poem_to_cluster']
        
#         print('\nstep 15: evaluating poem clustering')
#         full_df_poems = full_df.drop_duplicates(subset=['idoriginal_poem']).copy()
#         full_df_poems['id'] = full_df_poems['idoriginal_poem']
#         eval_result = self.evaluator.evaluate_clustering(
#             full_df_poems,
#             self.poem_clusters,
#             'poem_cluster',
#             self.config.poem_gt_col
#         )
#         self.perf_monitor.record('poem_evaluation', eval_result)
        
#         if eval_result['has_ground_truth']:
#             print(f"poem clustering evaluation:")
#             print(f"  ari: {eval_result.get('ari', 0):.4f}")
#             print(f"  nmi: {eval_result.get('nmi', 0):.4f}")
#             print(f"  fmi: {eval_result.get('fmi', 0):.4f}")
        
#         return self.poem_clusters
    
#     def run(self):
#         self.perf_monitor.start()
        
#         print('starting verse and poem clustering pipeline')
#         print('=' * 80)
        
#         best_verse_params = self.run_verse_clustering_on_subset()
        
#         if best_verse_params is not None:
#             best_threshold = best_verse_params['threshold']
#             best_resolution = best_verse_params['resolution']
#         else:
#             best_threshold = self.config.verse_similarity_thresholds[0]
#             best_resolution = self.config.leiden_resolutions[0]
        
#         self.run_full_verse_clustering(best_threshold, best_resolution)
        
#         best_poem_params = self.run_poem_clustering_on_subset()
        
#         if best_poem_params is not None:
#             best_poem_threshold = best_poem_params['threshold']
#         else:
#             best_poem_threshold = self.config.poem_similarity_thresholds[0]
        
#         self.run_full_poem_clustering(best_poem_threshold)
        
#         self.perf_monitor.end()
#         report_path = self.perf_monitor.generate_report()
        
#         print('\n' + '=' * 80)
#         print(f'pipeline completed')
#         print(f'performance report saved to: {report_path}')
        
#         return {
#             'verse_clusters': self.verse_clusters,
#             'poem_clusters': self.poem_clusters
#         }

# def main():
#     config = Config()
#     pipeline = Pipeline(config)
#     results = pipeline.run()
#     return results

# if __name__ == '__main__':
#     main()

import warnings
warnings.filterwarnings('ignore')

from config import Config
from data_loader import DataLoader
from embedding import EmbeddingGenerator
from similarity_search import SimilaritySearch
from graph_builder import GraphBuilder
from leiden_clustering import LeidenClustering
from grid_search import GridSearch
from poem_clustering import PoemClustering
from evaluation import Evaluator
from visualization import Visualizer
from performance_monitor import PerformanceMonitor
from gpu_monitor import GPUMonitor
import pandas as pd
import os

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.perf_monitor = PerformanceMonitor(config)
        self.gpu_monitor = GPUMonitor()
        
        self.data_loader = DataLoader(config)
        self.embedding_gen = EmbeddingGenerator(config)
        self.similarity_search = SimilaritySearch(config)
        self.graph_builder = GraphBuilder(config)
        self.leiden_clustering = LeidenClustering(config)
        self.grid_search = GridSearch(config)
        self.poem_clustering = PoemClustering(config)
        self.evaluator = Evaluator(config)
        self.visualizer = Visualizer(config)
        
        self.verse_clusters = None
        self.poem_clusters = None
        
    def run_verse_clustering_on_subset(self):
        load_result = self.data_loader.load_data()
        self.perf_monitor.record('data_loading', load_result)
        
        subset_result = self.data_loader.get_stratified_subset()
        self.perf_monitor.record('stratified_sampling', subset_result)
        
        model_result = self.embedding_gen.load_model()
        self.perf_monitor.record('model_loading', model_result)
        
        subset_df = self.data_loader.get_subset_data()
        embed_result = self.embedding_gen.embed_texts(
            subset_df['verse'].tolist(),
            subset_df['id'].tolist()
        )
        self.perf_monitor.record('embedding', embed_result)
        
        index_result = self.similarity_search.build_index(
            embed_result['embeddings'],
            embed_result['id_mapping']
        )
        self.perf_monitor.record('faiss_index', index_result)
        
        subset_df = self.data_loader.get_subset_data()
        gridsearch_result = self.grid_search.verse_gridsearch(
            self.similarity_search,
            self.leiden_clustering,
            embed_result['embeddings'],
            embed_result['id_mapping'],
            subset_df
        )
        self.perf_monitor.record('verse_gridsearch', {
            'n_configurations': len(gridsearch_result['results']),
            'total_time': gridsearch_result['total_time'],
            'has_ground_truth': gridsearch_result.get('has_ground_truth', False)
        })
        
        self.visualizer.plot_verse_gridsearch(gridsearch_result['results'])
        
        return gridsearch_result['best_params']
    
    def run_full_verse_clustering(self, best_threshold, best_resolution):
        full_df = self.data_loader.get_full_data()
        embed_result = self.embedding_gen.embed_texts(
            full_df['verse'].tolist(),
            full_df['id'].tolist()
        )
        
        self.similarity_search.build_index(
            embed_result['embeddings'],
            embed_result['id_mapping']
        )
        
        pairs_result = self.similarity_search.find_similar_pairs(best_threshold)
        
        clustering_result = self.leiden_clustering.cluster_leiden(
            pairs_result['pairs'],
            best_resolution
        )
        
        self.perf_monitor.record('full_verse_clustering', {
            'n_pairs': pairs_result['n_pairs'],
            'n_clusters': clustering_result['n_clusters'],
            'modularity': clustering_result.get('modularity', 0),
            'total_time': pairs_result['search_time'] + clustering_result['cluster_time']
        })
        
        self.verse_clusters = clustering_result['node_to_cluster']
        
        eval_result = self.evaluator.evaluate_clustering(
            full_df,
            self.verse_clusters,
            'verse_cluster',
            self.config.verse_gt_col
        )
        self.perf_monitor.record('verse_evaluation', eval_result)
        
        return self.verse_clusters
    
    def run_poem_clustering_on_subset(self):
        full_df = self.data_loader.get_full_data()
        subset_df = full_df.sample(
            frac=self.config.poem_subset_fraction, 
            random_state=self.config.random_seed
        )
        
        poem_rep_result = self.poem_clustering.create_poem_representations(
            subset_df,
            self.verse_clusters
        )
        self.perf_monitor.record('poem_representation_creation', poem_rep_result)
        
        poem_gridsearch_result = self.grid_search.poem_gridsearch(
            poem_rep_result['poem_to_clusters'],
            subset_df
        )
        self.perf_monitor.record('poem_gridsearch', {
            'n_configurations': len(poem_gridsearch_result['results']),
            'total_time': poem_gridsearch_result['total_time'],
            'has_ground_truth': poem_gridsearch_result.get('has_ground_truth', False)
        })
        
        self.visualizer.plot_poem_gridsearch(poem_gridsearch_result['results'])
        
        return poem_gridsearch_result['best_params']
    
    def run_full_poem_clustering(self, best_threshold):
        full_df = self.data_loader.get_full_data()
        poem_rep_result = self.poem_clustering.create_poem_representations(
            full_df,
            self.verse_clusters
        )
        
        poem_cluster_result = self.poem_clustering.cluster_poems_jaccard(
            poem_rep_result['poem_to_clusters'],
            best_threshold
        )
        
        self.perf_monitor.record('full_poem_clustering', poem_cluster_result)
        
        self.poem_clusters = poem_cluster_result['poem_to_cluster']
        
        full_df_poems = full_df.drop_duplicates(subset=['idoriginal_poem']).copy()
        full_df_poems['id'] = full_df_poems['idoriginal_poem']
        eval_result = self.evaluator.evaluate_clustering(
            full_df_poems,
            self.poem_clusters,
            'poem_cluster',
            self.config.poem_gt_col
        )
        self.perf_monitor.record('poem_evaluation', eval_result)
        
        return self.poem_clusters
    
    def save_enriched_dataset(self):
        full_df = self.data_loader.get_full_data()
        
        full_df['predicted_verse_cluster'] = full_df['id'].apply(
            lambda x: self.verse_clusters.get(x, -1)
        )
        
        full_df['predicted_poem_cluster'] = full_df['idoriginal_poem'].apply(
            lambda x: self.poem_clusters.get(x, -1)
        )
        
        output_path = os.path.join(
            self.config.ensure_output_dir(), 
            'enriched_dataset.csv'
        )
        full_df.to_csv(output_path, index=False)
        
        return output_path
    
    def run(self):
        self.gpu_monitor.print_status()
        
        self.perf_monitor.start()
        
        best_verse_params = self.run_verse_clustering_on_subset()
        
        if best_verse_params is not None:
            best_threshold = best_verse_params['threshold']
            best_resolution = best_verse_params['resolution']
        else:
            best_threshold = self.config.verse_similarity_thresholds[0]
            best_resolution = self.config.leiden_resolutions[0]
        
        self.run_full_verse_clustering(best_threshold, best_resolution)
        
        best_poem_params = self.run_poem_clustering_on_subset()
        
        if best_poem_params is not None:
            best_poem_threshold = best_poem_params['threshold']
        else:
            best_poem_threshold = self.config.poem_similarity_thresholds[0]
        
        self.run_full_poem_clustering(best_poem_threshold)
        
        enriched_path = self.save_enriched_dataset()
        
        self.perf_monitor.end()
        report_path = self.perf_monitor.generate_report()
        
        print(f'\npipeline completed')
        print(f'  performance report: {report_path}')
        print(f'  enriched dataset: {enriched_path}')
        
        self.gpu_monitor.print_status()
        
        return {
            'verse_clusters': self.verse_clusters,
            'poem_clusters': self.poem_clusters,
            'enriched_dataset_path': enriched_path
        }

def main():
    config = Config()
    pipeline = Pipeline(config)
    results = pipeline.run()
    return results

if __name__ == '__main__':
    main()