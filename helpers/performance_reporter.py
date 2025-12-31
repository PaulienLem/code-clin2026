
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd


class PerformanceReporter:

    def __init__(self, output_folder: str = "full_orthographic_results"):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        self.sections = []

    def add_header(self, title: str, width: int = 80):
        self.sections.append("=" * width)
        self.sections.append(title)
        self.sections.append("=" * width)
        self.sections.append("")

    def add_subheader(self, title: str, width: int = 80):
        self.sections.append(title)
        self.sections.append("-" * width)

    def add_line(self, content: str = ""):
        self.sections.append(content)

    def add_blank_line(self):
        self.sections.append("")

    def add_key_value(self, key: str, value: Any, width: int = 50):
        self.sections.append(f"{key:.<{width}} {value}")

    def add_system_information(self, system_info: Dict[str, Any]):
        self.add_subheader("System Information")

        self.add_key_value("Hostname:", system_info['hostname'])
        self.add_key_value("Platform:", system_info['platform'])
        self.add_key_value("Python Version:", system_info['python_version'])
        self.add_key_value("Processor:", system_info['processor'])
        self.add_key_value("CPU Cores (Physical):", system_info['cpu_count_physical'])
        self.add_key_value("CPU Cores (Logical):", system_info['cpu_count_logical'])
        self.add_key_value("Total RAM:", f"{system_info['total_ram_gb']:.2f} GB")
        self.add_key_value("Available RAM:", f"{system_info['available_ram_gb']:.2f} GB")
        self.add_key_value("GPU Available:", 'Yes' if system_info['has_gpu'] else 'No')

        if system_info['has_gpu']:
            self.add_key_value("GPU Count:", system_info['gpu_count'])
            self.add_key_value("GPU Memory:", f"{system_info['gpu_memory_gb']:.2f} GB")

        self.add_key_value("Timestamp:", system_info['timestamp'])
        self.add_blank_line()

    def add_resource_usage(self, resource_stats: Dict[str, float]):
        self.add_subheader("Peak Resource Usage")

        self.add_key_value("Peak RAM Usage:", f"{resource_stats['peak_ram_gb']:.2f} GB")
        self.add_key_value("Average RAM Usage:", f"{resource_stats['avg_ram_gb']:.2f} GB")
        self.add_key_value("Peak CPU Percent:", f"{resource_stats['peak_cpu_percent']:.1f}%")
        self.add_key_value("Average CPU Percent:", f"{resource_stats['avg_cpu_percent']:.1f}%")
        self.add_blank_line()

    def add_timing_breakdown(self, timing_summary: Dict[str, float], total_time: float):
        self.add_subheader("Timing Breakdown (by Stage)")

        total_measured = sum(timing_summary.values())

        for stage_name, duration in timing_summary.items():
            pct = (duration / total_measured * 100) if total_measured > 0 else 0
            formatted_value = f"{duration:>8.1f}s ({pct:>5.1f}%)"
            self.add_key_value(stage_name, formatted_value)

        self.add_key_value("Total Measured Time", f"{total_measured:>8.1f}s")
        self.add_key_value("Total Wall Clock Time",
                           f"{total_time:>8.1f}s ({total_time / 60:>6.1f} min)")
        self.add_blank_line()

    def add_timing_analysis(self, timing_summary: Dict[str, float]):
        self.add_subheader("Detailed Timing Analysis")

        verse_stages = [k for k in timing_summary.keys()
                        if k.startswith(('00_', '01_', '02_', '03_', '04_'))]
        verse_time = sum(timing_summary.get(k, 0) for k in verse_stages)

        poem_stages = [k for k in timing_summary.keys()
                       if k.startswith(('05_',)) or 'poem' in k.lower()]
        poem_time = sum(timing_summary.get(k, 0) for k in poem_stages)

        self.add_key_value("Verse-Level Clustering:",
                           f"{verse_time:>8.1f}s ({verse_time / 60:>6.1f} min)")
        self.add_key_value("Poem-Level Clustering:",
                           f"{poem_time:>8.1f}s ({poem_time / 60:>6.1f} min)")
        self.add_blank_line()

    def add_verse_clustering_results(self, verse_summary: Dict[str, Any]):
        self.add_subheader("Clustering Results Summary")
        self.add_line("Verse-Level:")

        self.add_line(f"  Total verses:             {verse_summary['n_verses']:,}")
        self.add_line(f"  Total clusters:           {verse_summary['n_clusters']:,}")
        self.add_line(f"  Multi-member clusters:    {verse_summary['n_multi_clusters']:,}")
        self.add_line(f"  Singletons:               {verse_summary['n_singletons']:,}")
        self.add_line(f"  Best shingle size:        {verse_summary['best_shingle_size']}")
        self.add_line(f"  Best threshold:           {verse_summary['best_threshold']:.3f}")
        self.add_blank_line()

    def add_poem_clustering_results(self, poem_threshold: float,
                                    poem_summary: Dict[str, Any] = None):
        self.add_line("Poem-Level:")
        self.add_line(f"  Selected threshold:       {poem_threshold:.3f}")

        if poem_summary:
            self.add_line(f"  Total poems:              {poem_summary.get('n_poems', 'N/A'):,}")
            self.add_line(f"  Poem clusters:            {poem_summary.get('n_poem_clusters', 'N/A'):,}")
            self.add_line(f"  Cross-dataset clusters:   {poem_summary.get('n_cross_dataset_clusters', 'N/A'):,}")

        self.add_blank_line()

    def add_performance_metrics(self, verse_summary: Dict[str, Any],
                                total_time: float, verse_time: float = None):
        self.add_subheader("Performance Metrics")

        if verse_time and verse_time > 0:
            throughput = verse_summary['n_verses'] / verse_time
            self.add_key_value("Verse clustering throughput:",
                               f"{throughput:.1f} verses/sec")

        overall_rate = verse_summary['n_verses'] / total_time
        self.add_key_value("Overall processing rate:",
                           f"{overall_rate:.1f} verses/sec")
        self.add_blank_line()

    def generate_report(self,
                        system_info: Dict[str, Any],
                        resource_stats: Dict[str, float],
                        timing_summary: Dict[str, float],
                        verse_summary: Dict[str, Any],
                        poem_threshold: float,
                        total_time: float,
                        poem_summary: Dict[str, Any] = None) -> str:

        self.sections = []

        self.add_header("Comprehensive Orthographic Clustering Performance Report")

        self.add_system_information(system_info)
        self.add_resource_usage(resource_stats)
        self.add_timing_breakdown(timing_summary, total_time)
        self.add_timing_analysis(timing_summary)
        self.add_verse_clustering_results(verse_summary)
        self.add_poem_clustering_results(poem_threshold, poem_summary)

        verse_stages = [k for k in timing_summary.keys()
                        if k.startswith(('00_', '01_', '02_', '03_', '04_'))]
        verse_time = sum(timing_summary.get(k, 0) for k in verse_stages)

        self.add_performance_metrics(verse_summary, total_time, verse_time)

        self.add_header("End of Report")

        report_path = self.output_folder / 'clustering_performance_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(self.sections))

        return str(report_path)

    def to_dataframe(self) -> pd.DataFrame:
        pass

    def to_markdown(self) -> str:
        markdown_sections = []
        for section in self.sections:
            if section.startswith("="):
                continue
            elif section.startswith("-"):
                continue
            elif "..." in section:
                markdown_sections.append(f"- {section}")
            else:
                markdown_sections.append(section)

        return '\n'.join(markdown_sections)

    def add_comparison_table(self, configs: List[Dict[str, Any]],
                             metrics: List[str]):
        pass

    def add_chart_reference(self, chart_path: str, description: str):
        self.add_line(f"See figure: {chart_path}")
        self.add_line(f"  {description}")
        self.add_blank_line()

    def export_json(self, filepath: str):
        import json
        data = {
            'system_info': getattr(self, 'system_info', None),
            'resource_stats': getattr(self, 'resource_stats', None),
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def add_custom_section(self, title: str, content: List[str]):
        self.add_subheader(title)
        for line in content:
            self.add_line(line)
        self.add_blank_line()