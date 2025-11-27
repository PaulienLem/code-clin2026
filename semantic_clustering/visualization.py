import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

class Visualizer:
    def __init__(self, config):
        self.config = config
        self.output_dir = config.ensure_output_dir()
    
    def plot_verse_gridsearch(self, results_df, filename='verse_gridsearch.png'):
        if len(results_df) == 0:
            return
        
        has_ari = 'ari' in results_df.columns
        has_composite = 'composite_score' in results_df.columns
        
        n_thresholds = results_df['threshold'].nunique()
        n_resolutions = results_df['resolution'].nunique()
        
        if n_thresholds > 1 and n_resolutions > 1:
            self._plot_heatmap(results_df, filename, has_ari, has_composite)
        else:
            self._plot_line(results_df, filename, has_ari, has_composite)
    
    def _plot_heatmap(self, results_df, filename, has_ari=False, has_composite=False):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        if has_ari:
            metrics = [
                ('ari', 'ari'),
                ('nmi', 'nmi'),
                ('n_clusters', 'number of clusters'),
                ('modularity', 'modularity')
            ]
        else:
            metrics = [
                ('composite_score', 'composite score') if has_composite else ('stability', 'stability'),
                ('n_clusters', 'number of clusters'),
                ('modularity', 'modularity'),
                ('n_pairs', 'number of pairs')
            ]
        
        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            if metric not in results_df.columns:
                continue
            
            pivot = results_df.pivot_table(
                values=metric,
                index='resolution',
                columns='threshold',
                aggfunc='mean'
            )
            
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', ax=ax)
            ax.set_title(title)
            ax.set_xlabel('similarity threshold')
            ax.set_ylabel('leiden resolution')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_line(self, results_df, filename, has_ari=False, has_composite=False):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        if has_ari:
            metrics = [
                ('ari', 'ari'),
                ('nmi', 'nmi'),
                ('n_clusters', 'number of clusters'),
                ('modularity', 'modularity')
            ]
        else:
            metrics = [
                ('composite_score', 'composite score') if has_composite else ('stability', 'stability'),
                ('n_clusters', 'number of clusters'),
                ('modularity', 'modularity'),
                ('n_pairs', 'number of pairs')
            ]
        
        if 'threshold' in results_df.columns and results_df['threshold'].nunique() > 1:
            x_col = 'threshold'
            group_col = 'resolution'
        else:
            x_col = 'resolution'
            group_col = 'threshold'
        
        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            if metric not in results_df.columns:
                continue
            
            for group_val in results_df[group_col].unique():
                subset = results_df[results_df[group_col] == group_val]
                ax.plot(subset[x_col], subset[metric], marker='o', label=f'{group_col}={group_val}')
            
            ax.set_title(title)
            ax.set_xlabel(x_col)
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_poem_gridsearch(self, results_df, filename='poem_gridsearch.png'):
        if len(results_df) == 0:
            return
        
        has_ari = 'ari' in results_df.columns
        has_composite = 'composite_score' in results_df.columns
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        if has_ari:
            metrics = [
                ('ari', 'ari'),
                ('nmi', 'nmi'),
                ('n_clusters', 'number of clusters'),
                ('avg_cluster_size', 'average cluster size')
            ]
        else:
            metrics = [
                ('composite_score', 'composite score') if has_composite else ('silhouette', 'silhouette score'),
                ('n_clusters', 'number of clusters'),
                ('avg_cluster_size', 'average cluster size'),
                ('n_pairs', 'number of pairs')
            ]
        
        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            if metric not in results_df.columns:
                continue
            
            ax.plot(results_df['threshold'], results_df[metric], marker='o', linewidth=2)
            ax.set_title(title)
            ax.set_xlabel('similarity threshold')
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()