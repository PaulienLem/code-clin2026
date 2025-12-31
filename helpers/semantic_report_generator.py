from pathlib import Path
import logging

class SemanticReportGenerator:
    def __init__(self, system_info, resource_stats, timing_summary,
                 verse_summary, poem_summary, total_time, results_dir: Path,
                 logger: logging.Logger = None):
        self.system_info = system_info
        self.resource_stats = resource_stats
        self.timing_summary = timing_summary
        self.verse_summary = verse_summary
        self.poem_summary = poem_summary
        self.total_time = total_time
        self.results_dir = results_dir
        self.logger = logger

    def generate_report_lines(self):
        lines = []
        lines.append("="*80)
        lines.append("Comprehensive clustering performance report")
        lines.append("="*80)
        lines.append("")
        lines.append("System information")
        lines.append("-"*80)
        lines.append(f"Hostname:            {self.system_info['hostname']}")
        lines.append(f"Platform:            {self.system_info['platform']}")
        lines.append(f"Python Version:      {self.system_info['python_version']}")
        lines.append(f"Processor:           {self.system_info['processor']}")
        lines.append(f"CPU Cores (Physical):{self.system_info['cpu_count_physical']}")
        lines.append(f"CPU Cores (Logical): {self.system_info['cpu_count_logical']}")
        lines.append(f"Total RAM:           {self.system_info['total_ram_gb']:.2f} GB")
        lines.append(f"GPU Available:       {self.system_info['gpu_available']}")
        lines.append("")
        lines.append("Peak resource usage")
        lines.append("-"*80)
        lines.append(f"Peak RAM Usage:      {self.resource_stats['peak_ram_gb']:.2f} GB")
        lines.append(f"Average RAM Usage:   {self.resource_stats['avg_ram_gb']:.2f} GB")
        if self.system_info['gpu_available']:
            lines.append(f"Peak GPU Memory:     {self.resource_stats['peak_gpu_mem_gb']:.2f} GB")
            lines.append(f"Average GPU Memory:  {self.resource_stats['avg_gpu_mem_gb']:.2f} GB")
        lines.append("")
        lines.append("Timing breakdown (by stage)")
        lines.append("-"*80)
        total_measured = sum(self.timing_summary.values())
        for stage, duration in self.timing_summary.items():
            pct = (duration / total_measured * 100) if total_measured > 0 else 0
            lines.append(f"{stage:.<50} {duration:>8.1f}s ({pct:>5.1f}%)")
        lines.append(f"{'Total measured time':.<50} {total_measured:>8.1f}s")
        lines.append(f"{'Total wall clock time':.<50} {self.total_time:>8.1f}s ({self.total_time/60:>6.1f} min)")
        lines.append("")
        lines.append("Clustering results summary")
        lines.append("-"*80)
        lines.append("Verse-level:")
        for k, v in self.verse_summary.items():
            lines.append(f"  {k.replace('_',' ').title():<25} {v}")
        lines.append("")
        lines.append("Poem-level:")
        for k, v in self.poem_summary.items():
            lines.append(f"  {k.replace('_',' ').title():<25} {v}")
        lines.append("")
        lines.append("Performance metrics")
        lines.append("-"*80)
        verse_time = sum(self.timing_summary.get(k,0) for k in self.timing_summary if k.startswith(('01_','02_','03_','04_')))
        poem_time = sum(self.timing_summary.get(k,0) for k in self.timing_summary if k.startswith(('13_','14_','15_')))
        if verse_time > 0:
            lines.append(f"Verse clustering throughput:  {self.verse_summary['n_verses'] / verse_time:.1f} verses/sec")
        if poem_time > 0:
            lines.append(f"Poem clustering throughput:   {self.poem_summary['n_poems'] / poem_time:.1f} poems/sec")
        lines.append(f"Overall processing rate:      {self.verse_summary['n_verses'] / self.total_time:.1f} verses/sec")
        lines.append("")
        lines.append("="*80)
        lines.append("End of report")
        lines.append("="*80)
        return lines

    def save(self, filename='clustering_performance_report.txt'):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.results_dir / filename
        lines = self.generate_report_lines()
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))

        return report_path
