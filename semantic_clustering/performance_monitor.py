# import time
# import psutil
# import pickle
# import os
# from datetime import datetime

# class PerformanceMonitor:
#     def __init__(self, config):
#         self.config = config
#         self.start_time = None
#         self.metrics = {}
#         self.output_dir = config.ensure_output_dir()
        
#     def start(self):
#         self.start_time = time.time()
#         self.metrics['start_time'] = datetime.now().isoformat()
#         self.metrics['cpu_count'] = psutil.cpu_count()
#         self.metrics['memory_total_gb'] = psutil.virtual_memory().total / 1024**3
        
#     def record(self, stage, data):
#         summary = {}
#         if isinstance(data, dict):
#             for k, v in data.items():
#                 if isinstance(v, (int, float, str, bool)):
#                     summary[k] = v
#                 elif hasattr(v, '__len__'):
#                     summary[k] = len(v)
#         self.metrics[stage] = summary
    
#     def end(self):
#         self.metrics['end_time'] = datetime.now().isoformat()
#         self.metrics['total_runtime'] = time.time() - self.start_time
#         self.metrics['peak_memory_gb'] = psutil.Process().memory_info().rss / 1024**3
        
#     def generate_report(self, filename='performance_report.txt'):
#         report_lines = []
#         report_lines.append('=' * 80)
#         report_lines.append('performance report')
#         report_lines.append('=' * 80)
#         report_lines.append('')
        
#         report_lines.append('system configuration')
#         report_lines.append('-' * 80)
#         report_lines.append(f"cpu cores: {self.metrics.get('cpu_count', 'n/a')}")
#         report_lines.append(f"total memory: {self.metrics.get('memory_total_gb', 0):.2f} gb")
#         report_lines.append(f"peak memory usage: {self.metrics.get('peak_memory_gb', 0):.2f} gb")
#         report_lines.append('')
        
#         report_lines.append('execution timeline')
#         report_lines.append('-' * 80)
#         report_lines.append(f"start time: {self.metrics.get('start_time', 'n/a')}")
#         report_lines.append(f"end time: {self.metrics.get('end_time', 'n/a')}")
#         report_lines.append(f"total runtime: {self.metrics.get('total_runtime', 0):.2f} seconds")
#         report_lines.append('')
        
#         stage_order = [
#             'data_loading',
#             'stratified_sampling',
#             'model_loading',
#             'embedding',
#             'faiss_index',
#             'verse_gridsearch',
#             'full_verse_clustering',
#             'poem_representation_creation',
#             'poem_gridsearch',
#             'full_poem_clustering',
#             'verse_evaluation',
#             'poem_evaluation'
#         ]
        
#         report_lines.append('stage-by-stage breakdown')
#         report_lines.append('-' * 80)
        
#         for stage in stage_order:
#             if stage in self.metrics:
#                 data = self.metrics[stage]
#                 report_lines.append(f"\n{stage}:")
                
#                 if isinstance(data, dict):
#                     for key, value in data.items():
#                         if isinstance(value, float):
#                             report_lines.append(f"  {key}: {value:.4f}")
#                         else:
#                             report_lines.append(f"  {key}: {value}")
        
#         report_lines.append('')
#         report_lines.append('=' * 80)
        
#         report_text = '\n'.join(report_lines)
        
#         output_path = os.path.join(self.output_dir, filename)
#         with open(output_path, 'w') as f:
#             f.write(report_text)
        
#         pickle_path = os.path.join(self.output_dir, 'performance_metrics.pkl')
#         with open(pickle_path, 'wb') as f:
#             pickle.dump(self.metrics, f)
        
#         return output_path

import time
import psutil
import pickle
import os
from datetime import datetime

class PerformanceMonitor:
    def __init__(self, config):
        self.config = config
        self.start_time = None
        self.metrics = {}
        self.output_dir = config.ensure_output_dir()
        self.total_memory_gb = psutil.virtual_memory().total / 1024**3
        
    def start(self):
        self.start_time = time.time()
        self.metrics['start_time'] = datetime.now().isoformat()
        self.metrics['cpu_count'] = psutil.cpu_count()
        self.metrics['memory_total_gb'] = self.total_memory_gb
        
        try:
            import torch
            if torch.cuda.is_available():
                n_gpus = torch.cuda.device_count()
                self.metrics['n_gpus'] = n_gpus
                self.metrics['gpu_info'] = []
                for i in range(n_gpus):
                    props = torch.cuda.get_device_properties(i)
                    self.metrics['gpu_info'].append({
                        'id': i,
                        'name': props.name,
                        'memory_total_gb': props.total_memory / 1024**3
                    })
        except:
            pass
        
    def record(self, stage, data):
        summary = {}
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (int, float, str, bool)):
                    summary[k] = v
                elif hasattr(v, '__len__'):
                    summary[k] = len(v)
        self.metrics[stage] = summary
    
    def end(self):
        self.metrics['end_time'] = datetime.now().isoformat()
        self.metrics['total_runtime'] = time.time() - self.start_time
        peak_memory_gb = psutil.Process().memory_info().rss / 1024**3
        self.metrics['peak_memory_gb'] = peak_memory_gb
        self.metrics['peak_memory_pct'] = (peak_memory_gb / self.total_memory_gb * 100) if self.total_memory_gb > 0 else 0
        
        try:
            import torch
            if torch.cuda.is_available():
                n_gpus = torch.cuda.device_count()
                self.metrics['gpu_usage'] = []
                for i in range(n_gpus):
                    mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    self.metrics['gpu_usage'].append({
                        'id': i,
                        'memory_allocated_gb': mem_allocated,
                        'memory_allocated_pct': (mem_allocated / mem_total * 100) if mem_total > 0 else 0
                    })
        except:
            pass
        
    def generate_report(self, filename='performance_report.txt'):
        report_lines = []
        report_lines.append('=' * 80)
        report_lines.append('performance report')
        report_lines.append('=' * 80)
        report_lines.append('')
        
        report_lines.append('system configuration')
        report_lines.append('-' * 80)
        report_lines.append(f"cpu cores: {self.metrics.get('cpu_count', 'n/a')}")
        total_mem = self.metrics.get('memory_total_gb', 0)
        peak_mem = self.metrics.get('peak_memory_gb', 0)
        peak_pct = self.metrics.get('peak_memory_pct', 0)
        report_lines.append(f"total memory: {total_mem:.2f} gb")
        report_lines.append(f"peak memory usage: {peak_mem:.2f} gb ({peak_pct:.1f}%)")
        
        if 'gpu_info' in self.metrics:
            report_lines.append(f"\ngpus: {self.metrics.get('n_gpus', 0)}")
            for gpu in self.metrics['gpu_info']:
                report_lines.append(f"  gpu {gpu['id']}: {gpu['name']}")
                report_lines.append(f"    memory total: {gpu['memory_total_gb']:.1f} gb")
            
            if 'gpu_usage' in self.metrics:
                report_lines.append(f"\ngpu peak usage:")
                for gpu_usage in self.metrics['gpu_usage']:
                    gpu_id = gpu_usage['id']
                    mem_gb = gpu_usage['memory_allocated_gb']
                    mem_pct = gpu_usage['memory_allocated_pct']
                    report_lines.append(f"  gpu {gpu_id}: {mem_gb:.1f} gb ({mem_pct:.1f}%)")
        
        report_lines.append('')
        
        report_lines.append('execution timeline')
        report_lines.append('-' * 80)
        report_lines.append(f"start time: {self.metrics.get('start_time', 'n/a')}")
        report_lines.append(f"end time: {self.metrics.get('end_time', 'n/a')}")
        report_lines.append(f"total runtime: {self.metrics.get('total_runtime', 0):.2f} seconds")
        report_lines.append('')
        
        stage_order = [
            'data_loading',
            'stratified_sampling',
            'model_loading',
            'embedding',
            'faiss_index',
            'verse_gridsearch',
            'full_verse_clustering',
            'poem_representation_creation',
            'poem_gridsearch',
            'full_poem_clustering',
            'verse_evaluation',
            'poem_evaluation'
        ]
        
        report_lines.append('stage-by-stage breakdown')
        report_lines.append('-' * 80)
        
        for stage in stage_order:
            if stage in self.metrics:
                data = self.metrics[stage]
                report_lines.append(f"\n{stage}:")
                
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, float):
                            report_lines.append(f"  {key}: {value:.4f}")
                        else:
                            report_lines.append(f"  {key}: {value}")
        
        report_lines.append('')
        report_lines.append('=' * 80)
        
        report_text = '\n'.join(report_lines)
        
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        pickle_path = os.path.join(self.output_dir, 'performance_metrics.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.metrics, f)
        
        return output_path