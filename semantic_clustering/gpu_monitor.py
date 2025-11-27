import torch
import psutil
import time

class GPUMonitor:
    def __init__(self):
        self.has_gpu = torch.cuda.is_available()
        self.n_gpus = torch.cuda.device_count() if self.has_gpu else 0
        
    def get_gpu_info(self):
        if not self.has_gpu:
            return {
                'has_gpu': False,
                'n_gpus': 0
            }
        
        gpu_info = {
            'has_gpu': True,
            'n_gpus': self.n_gpus,
            'gpus': []
        }
        
        for i in range(self.n_gpus):
            props = torch.cuda.get_device_properties(i)
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            mem_total = props.total_memory / 1024**3
            
            gpu_info['gpus'].append({
                'id': i,
                'name': props.name,
                'memory_total_gb': mem_total,
                'memory_allocated_gb': mem_allocated,
                'memory_reserved_gb': mem_reserved,
                'memory_allocated_pct': (mem_allocated / mem_total * 100) if mem_total > 0 else 0,
                'memory_reserved_pct': (mem_reserved / mem_total * 100) if mem_total > 0 else 0,
                'compute_capability': f"{props.major}.{props.minor}"
            })
        
        return gpu_info
    
    def get_system_info(self):
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_usage_pct': cpu_percent,
            'memory_total_gb': mem.total / 1024**3,
            'memory_used_gb': mem.used / 1024**3,
            'memory_available_gb': mem.available / 1024**3,
            'memory_used_pct': mem.percent
        }
    
    def print_status(self):
        print('\n' + '=' * 80)
        print('computational resources')
        print('=' * 80)
        
        sys_info = self.get_system_info()
        print(f"\ncpu:")
        print(f"  cores: {sys_info['cpu_count']}")
        print(f"  usage: {sys_info['cpu_usage_pct']:.1f}%")
        
        print(f"\nsystem memory:")
        print(f"  total: {sys_info['memory_total_gb']:.1f} gb")
        print(f"  used: {sys_info['memory_used_gb']:.1f} gb ({sys_info['memory_used_pct']:.1f}%)")
        print(f"  available: {sys_info['memory_available_gb']:.1f} gb")
        
        gpu_info = self.get_gpu_info()
        if gpu_info['has_gpu']:
            print(f"\ngpus: {gpu_info['n_gpus']}")
            for gpu in gpu_info['gpus']:
                print(f"\n  gpu {gpu['id']}: {gpu['name']}")
                print(f"    memory total: {gpu['memory_total_gb']:.1f} gb")
                print(f"    memory allocated: {gpu['memory_allocated_gb']:.1f} gb ({gpu['memory_allocated_pct']:.1f}%)")
                print(f"    memory reserved: {gpu['memory_reserved_gb']:.1f} gb ({gpu['memory_reserved_pct']:.1f}%)")
                print(f"    compute capability: {gpu['compute_capability']}")
        else:
            print("\ngpus: none available")
        
        print('=' * 80)
