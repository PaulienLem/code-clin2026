import psutil
import numpy as np
import logging
class ReportGenerator:
    def __init__(
        self,
        df,
        verse_clusters,
        poem_clusters,
        best_verse_params,
        timing,
        resources,
        output_dir,
        format_stage_resources_fn
    ):
        self.df = df
        self.verse_clusters = verse_clusters
        self.poem_clusters = poem_clusters
        self.best_verse_params = best_verse_params
        self.timing = timing
        self.resources = resources
        self.output_dir = output_dir
        self.format_stage_resources = format_stage_resources_fn

    def generate(self):
        logger = logging.getLogger(__name__)

        n_verse_clusters = len(set(self.verse_clusters.values()))
        n_poem_clusters = len(set(self.poem_clusters.values()))
        
        avg_cpu = np.mean(self.resources['cpu_percent']) if self.resources['cpu_percent'] else 0
        max_cpu = np.max(self.resources['cpu_percent']) if self.resources['cpu_percent'] else 0
        avg_mem = np.mean(self.resources['memory_percent']) if self.resources['memory_percent'] else 0
        max_mem = np.max(self.resources['memory_percent']) if self.resources['memory_percent'] else 0
        avg_mem_gb = np.mean(self.resources['memory_used_gb']) if self.resources['memory_used_gb'] else 0
        max_mem_gb = np.max(self.resources['memory_used_gb']) if self.resources['memory_used_gb'] else 0

        total_mem = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)

        report = [
            "Complete pipeline report",
            "\Dataset:",
            f"  Verses: {len(self.df):,}",
            f"  Poems: {self.df['idoriginal_poem'].nunique():,}",
            "\nBest verse parameters:",
            f"  num_perm: {self.best_verse_params['num_perm']}",
            f"  shingle_size: {self.best_verse_params['shingle_size']}",
            f"  threshold: {self.best_verse_params['threshold']:.2f}",
            "\nResults:",
            f"  Verse clusters: {n_verse_clusters:,}",
            f"  Poem clusters: {n_poem_clusters:,}",
            f"  Avg verses/cluster: {len(self.df)/n_verse_clusters:.1f}",
            f"  Avg poems/cluster: {len(self.poem_clusters)/n_poem_clusters:.1f}",
            "\nTiming (wall-clock):",
            f"  Data loading: {self.timing['load']:.1f}s",
            f"  Verse clustering: {self.timing['verse_clustering']:.1f}s",
            f"  Poem clustering: {self.timing['poem_clustering']:.1f}s",
            f"  TOTAL: {self.timing['total']:.1f}s",
            "\nThroughput:",
            f"  Verses/sec: {len(self.df)/self.timing['total']:.0f}",
            f"  Poems/sec: {len(self.poem_clusters)/self.timing['total']:.0f}",
            "\nComputational resources:",
            f"  System: {cpu_count} cores ({cpu_count_logical} logical), {total_mem:.1f} GB RAM",
            f"  CPU: avg={avg_cpu:.1f}%, max={max_cpu:.1f}%",
            f"  Memory: avg={avg_mem:.1f}% ({avg_mem_gb:.2f} GB), max={max_mem:.1f}% ({max_mem_gb:.2f} GB)",
            "\nResource usage by stage:",
            f"  Data loading: {self.format_stage_resources(0, 1)}",
            f"  Verse clustering: {self.format_stage_resources(1, len(self.resources['cpu_percent'])//2)}",
            f"  Poem clustering: {self.format_stage_resources(len(self.resources['cpu_percent'])//2, -1)}",
            "\nOutput files:",
            f"  {self.output_dir}/verse_clusters.csv",
            f"  {self.output_dir}/poem_clusters.csv",
            f"  {self.output_dir}/verse_grid_search_results.csv",
            f"  {self.output_dir}/poem_grid_search_results.csv",
        ]

        report_text = "\n".join(report)
        logger.info("\n" + report_text)

        with open(self.output_dir / 'performance_report.txt', 'w') as f:
            f.write(report_text)
