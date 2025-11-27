import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, output_dir, has_verse_gt=False, has_poem_gt=False):
        self.output_dir = Path(output_dir)
        self.has_verse_gt = has_verse_gt
        self.has_poem_gt = has_poem_gt
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_visualizations(self):
        verse_results = pd.read_csv(self.output_dir / 'verse_grid_search_results.csv')
        poem_results = pd.read_csv(self.output_dir / 'poem_grid_search_results.csv')

        metric_col = 'ari' if self.has_verse_gt else 'score'
        poem_metric = 'ari' if self.has_poem_gt else 'score'

        if 'num_perm' in verse_results.columns and len(verse_results['num_perm'].unique()) > 1:
            self._plot_verse_3d_heatmap(verse_results, metric_col)

        self._plot_verse_line_charts(verse_results, metric_col)
        self._plot_poem_line_chart(poem_results, poem_metric)
        self._plot_comparison_chart(verse_results, poem_results, metric_col, poem_metric)

        logger.info(f"Saved visualizations to {self.output_dir}")

    def _plot_verse_3d_heatmap(self, results_df, metric_col):
        fig = plt.figure(figsize=(16, 12))

        unique_perms = sorted(results_df['num_perm'].unique())
        n_perms = len(unique_perms)

        for idx, num_perm in enumerate(unique_perms):
            subset = results_df[results_df['num_perm'] == num_perm]

            if len(subset) > 1:
                pivot = subset.pivot(
                    index='shingle_size',
                    columns='threshold',
                    values=metric_col
                )

                ax = fig.add_subplot(1, n_perms, idx + 1)
                sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis',
                            cbar_kws={'label': metric_col}, ax=ax)
                ax.set_title(f'num_perm={num_perm}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Threshold', fontsize=10)
                ax.set_ylabel('Shingle size', fontsize=10)

        fig.suptitle(f'Verse Clustering Grid Search: {metric_col}',
                     fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        fig.savefig(self.output_dir / 'verse_grid_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _plot_verse_line_charts(self, results_df, metric_col):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax1 = axes[0, 0]
        for num_perm in sorted(results_df['num_perm'].unique()):
            for shingle_size in sorted(results_df['shingle_size'].unique()):
                subset = results_df[(results_df['num_perm'] == num_perm) &
                                    (results_df['shingle_size'] == shingle_size)]
                if len(subset) > 1:
                    ax1.plot(subset['threshold'], subset[metric_col],
                             marker='o', label=f'p={num_perm},s={shingle_size}')
        ax1.set_xlabel('Threshold', fontsize=10)
        ax1.set_ylabel(metric_col, fontsize=10)
        ax1.set_title('Score by Threshold', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        for num_perm in sorted(results_df['num_perm'].unique()):
            for threshold in sorted(results_df['threshold'].unique()):
                subset = results_df[(results_df['num_perm'] == num_perm) &
                                    (results_df['threshold'] == threshold)]
                if len(subset) > 1:
                    ax2.plot(subset['shingle_size'], subset[metric_col],
                             marker='o', label=f'p={num_perm},t={threshold:.2f}')
        ax2.set_xlabel('Shingle Size', fontsize=10)
        ax2.set_ylabel(metric_col, fontsize=10)
        ax2.set_title('Score by Shingle Size', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=8, ncol=2)
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        for num_perm in sorted(results_df['num_perm'].unique()):
            for shingle_size in sorted(results_df['shingle_size'].unique()):
                subset = results_df[(results_df['num_perm'] == num_perm) &
                                    (results_df['shingle_size'] == shingle_size)]
                if len(subset) > 1:
                    ax3.plot(subset['threshold'], subset['n_clusters'],
                             marker='s', label=f'p={num_perm},s={shingle_size}')
        ax3.set_xlabel('Threshold', fontsize=10)
        ax3.set_ylabel('Number of Clusters', fontsize=10)
        ax3.set_title('Cluster Count by Threshold', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=8, ncol=2)
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        if 'avg_cluster_size' in results_df.columns:
            for num_perm in sorted(results_df['num_perm'].unique()):
                for shingle_size in sorted(results_df['shingle_size'].unique()):
                    subset = results_df[(results_df['num_perm'] == num_perm) &
                                        (results_df['shingle_size'] == shingle_size)]
                    if len(subset) > 1:
                        ax4.plot(subset['threshold'], subset['avg_cluster_size'],
                                 marker='^', label=f'p={num_perm},s={shingle_size}')
            ax4.set_xlabel('Threshold', fontsize=10)
            ax4.set_ylabel('Avg Cluster Size', fontsize=10)
            ax4.set_title('Avg Cluster Size by Threshold', fontsize=11, fontweight='bold')
            ax4.legend(fontsize=8, ncol=2)
            ax4.grid(True, alpha=0.3)

        fig.suptitle('Verse Clustering Grid Search Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(self.output_dir / 'verse_grid_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _plot_poem_line_chart(self, results_df, metric_col):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        ax1 = axes[0]
        ax1.plot(results_df['threshold'], results_df[metric_col],
                 marker='o', linewidth=2, markersize=8, color='darkblue')
        ax1.set_xlabel('Jaccard Threshold', fontsize=10)
        ax1.set_ylabel(metric_col, fontsize=10)
        ax1.set_title(f'{metric_col.upper()} by Threshold', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(results_df['threshold'], results_df['n_clusters'],
                 marker='s', linewidth=2, markersize=8, color='darkgreen')
        ax2.set_xlabel('Jaccard Threshold', fontsize=10)
        ax2.set_ylabel('Number of Clusters', fontsize=10)
        ax2.set_title('Cluster Count by Threshold', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        ax3 = axes[2]
        if 'avg_cluster_size' in results_df.columns:
            ax3.plot(results_df['threshold'], results_df['avg_cluster_size'],
                     marker='^', linewidth=2, markersize=8, color='darkred')
            ax3.set_xlabel('Jaccard Threshold', fontsize=10)
            ax3.set_ylabel('Avg Cluster Size', fontsize=10)
            ax3.set_title('Avg Cluster Size by Threshold', fontsize=11, fontweight='bold')
            ax3.grid(True, alpha=0.3)

        fig.suptitle('Poem Clustering Grid Search Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(self.output_dir / 'poem_grid_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _plot_comparison_chart(self, verse_results, poem_results, verse_metric, poem_metric):
        fig = plt.figure(figsize=(14, 5))

        ax1 = fig.add_subplot(1, 2, 1)
        top_verse = verse_results.nlargest(10, verse_metric)
        labels = [
            f"p{int(row['num_perm'])}_s{int(row['shingle_size'])}_t{row['threshold']:.2f}"
            for _, row in top_verse.iterrows()
        ]
        ax1.barh(range(len(top_verse)), top_verse[verse_metric], color='steelblue')
        ax1.set_yticks(range(len(top_verse)))
        ax1.set_yticklabels(labels, fontsize=8)
        ax1.set_xlabel(verse_metric, fontsize=10)
        ax1.set_title('Top 10 Verse Configurations', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(poem_results['threshold'], poem_results[poem_metric],
                 marker='o', linewidth=2, markersize=8, color='darkgreen')
        ax2.set_xlabel('Jaccard Threshold', fontsize=10)
        ax2.set_ylabel(poem_metric, fontsize=10)
        ax2.set_title('Poem Clustering Performance', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        fig.suptitle('Verse vs Poem Clustering Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(self.output_dir / 'comparison_chart.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
