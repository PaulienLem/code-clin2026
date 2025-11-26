import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusteringVisualizer:
    @staticmethod
    def plot_heatmap(results_df: pd.DataFrame,
                     metric_col: str,
                     title: str = "Clustering quality heatmap",
                     figsize: Tuple[int, int] = (10, 8),
                     save_path: Optional[str] = None) -> plt.Figure:
        logger.info(f"Creating heatmap for {metric_col}")
        pivot_data = results_df.pivot(
            index='shingle_size',
            columns='threshold',
            values=metric_col
        )
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(pivot_data, 
                    annot=True, 
                    fmt='.3f', 
                    cmap='viridis',
                    cbar_kws={'label': metric_col},
                    ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Jaccard threshold', fontsize=12)
        ax.set_ylabel('Shingle size', fontsize=12)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved heatmap to {save_path}")
        return fig
    
    @staticmethod
    def plot_line_chart(results_df: pd.DataFrame,
                        metric_col: str,
                        group_by: str = 'shingle_size',
                        title: str = "Clustering quality by parameters",
                        figsize: Tuple[int, int] = (10, 6),
                        save_path: Optional[str] = None) -> plt.Figure:
        logger.info(f"Creating line chart for {metric_col}")
        fig, ax = plt.subplots(figsize=figsize)
        if group_by == 'shingle_size':
            x_col = 'threshold'
            for shingle_size in sorted(results_df['shingle_size'].unique()):
                subset = results_df[results_df['shingle_size'] == shingle_size]
                ax.plot(subset[x_col], subset[metric_col], 
                        marker='o', label=f'Shingle size {shingle_size}',
                        linewidth=2, markersize=6)
            ax.set_xlabel('Jaccard threshold', fontsize=12)
        else:
            x_col = 'shingle_size'
            for threshold in sorted(results_df['threshold'].unique()):
                subset = results_df[results_df['threshold'] == threshold]
                ax.plot(subset[x_col], subset[metric_col], 
                        marker='o', label=f'Threshold {threshold:.2f}',
                        linewidth=2, markersize=6)
            ax.set_xlabel('Shingle size', fontsize=12)
        ax.set_ylabel(metric_col, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved line chart to {save_path}")
        return fig
    
    @staticmethod
    def plot_cluster_size_distribution(cluster_labels: List[int],
                                       title: str = "Cluster size distribution",
                                       figsize: Tuple[int, int] = (10, 6),
                                       save_path: Optional[str] = None) -> plt.Figure:
        logger.info("Creating cluster size distribution plot")
        cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        ax1.hist(cluster_sizes.values, bins=30, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Cluster size', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Cluster size histogram', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax2.boxplot(cluster_sizes.values, vert=True)
        ax2.set_ylabel('Cluster size', fontsize=12)
        ax2.set_title('Cluster size box plot', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved cluster size distribution to {save_path}")
        return fig
    
    @staticmethod
    def plot_comparison_chart(verse_results: pd.DataFrame,
                              poem_results: pd.DataFrame,
                              metric_col: str,
                              title: str = "Verse vs Poem clustering quality",
                              figsize: Tuple[int, int] = (12, 6),
                              save_path: Optional[str] = None) -> plt.Figure:
        logger.info(f"Creating comparison chart for {metric_col}")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        if 'shingle_size' in verse_results.columns:
            pivot_verse = verse_results.pivot(
                index='shingle_size',
                columns='threshold',
                values=metric_col
            )
            sns.heatmap(pivot_verse, annot=True, fmt='.3f', cmap='viridis', ax=ax1)
            ax1.set_title('Verse-level clustering', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Threshold', fontsize=10)
            ax1.set_ylabel('Shingle size', fontsize=10)
        if 'threshold' in poem_results.columns:
            if len(poem_results) > 1:
                ax2.plot(poem_results['threshold'], poem_results[metric_col], 
                         marker='o', linewidth=2, markersize=8, color='darkblue')
                ax2.set_xlabel('Threshold', fontsize=10)
                ax2.set_ylabel(metric_col, fontsize=10)
                ax2.grid(True, alpha=0.3)
            ax2.set_title('Poem-level clustering', fontsize=12, fontweight='bold')
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison chart to {save_path}")
        return fig
    
    @staticmethod
    def create_all_visualizations(verse_results: pd.DataFrame,
                                  poem_results: pd.DataFrame,
                                  verse_metric: str,
                                  poem_metric: str,
                                  output_dir: str = '.') -> Dict[str, plt.Figure]:
        figures = {}
        fig1 = ClusteringVisualizer.plot_heatmap(
            verse_results, 
            verse_metric,
            title=f"Verse-level clustering: {verse_metric}",
            save_path=f"{output_dir}/verse_heatmap.png"
        )
        figures['verse_heatmap'] = fig1
        fig2 = ClusteringVisualizer.plot_line_chart(
            verse_results,
            verse_metric,
            title=f"Verse-level clustering: {verse_metric} by threshold",
            save_path=f"{output_dir}/verse_linechart.png"
        )
        figures['verse_linechart'] = fig2
        if len(poem_results) > 1:
            fig3 = ClusteringVisualizer.plot_line_chart(
                poem_results,
                poem_metric,
                group_by='threshold',
                title=f"Poem-level clustering: {poem_metric}",
                save_path=f"{output_dir}/poem_linechart.png"
            )
            figures['poem_linechart'] = fig3
        fig4 = ClusteringVisualizer.plot_comparison_chart(
            verse_results,
            poem_results,
            verse_metric,
            title="Verse vs Poem clustering quality",
            save_path=f"{output_dir}/comparison_chart.png"
        )
        figures['comparison'] = fig4
        logger.info(f"Created {len(figures)} visualizations in {output_dir}")
        return figures