import re
import unicodedata
from typing import Dict
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasketch import MinHash, MinHashLSHForest
import multiprocessing as mp
import hashlib
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import defaultdict
from numba import njit, cuda
import psutil
import platform
import socket
from datetime import datetime
import threading
from pathlib import Path
import gc
class SystemResourceAnalyzer:
    def __init__(self):
        self.cpu_count_physical = psutil.cpu_count(logical=False) or mp.cpu_count()
        self.cpu_count_logical = psutil.cpu_count(logical=True) or mp.cpu_count()
        self.total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        self.available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
        self.has_gpu = self._check_gpu()
        self.gpu_count = self._get_gpu_count() if self.has_gpu else 0
        self.gpu_memory_gb = self._get_gpu_memory() if self.has_gpu else 0

    def _check_gpu(self):
        try:
            cuda.detect()
            return True
        except:
            return False

    def _get_gpu_count(self):
        try:
            return len(cuda.gpus)
        except:
            return 0

    def _get_gpu_memory(self):
        try:
            if cuda.gpus:
                return cuda.current_context().get_memory_info()[1] / (1024 ** 3)
            return 0
        except:
            return 0

    def get_optimal_workers(self, task_type='cpu_intensive'):
        if task_type == 'cpu_intensive':
            return min(self.cpu_count_logical, 64)
        elif task_type == 'io_intensive':
            return min(self.cpu_count_logical * 2, 128)
        elif task_type == 'memory_intensive':
            workers = int(self.available_ram_gb / 2)
            return max(min(workers, self.cpu_count_logical), 4)
        else:
            return self.cpu_count_logical

    def get_optimal_chunk_size(self, total_items, workers):
        base_chunk = max(50, total_items // (workers * 8))
        available_ram_per_worker_gb = self.available_ram_gb / workers * 0.7
        max_chunk_by_memory = int(available_ram_per_worker_gb * 100000)
        return min(base_chunk, max_chunk_by_memory, 50000)

    def should_use_gpu(self, data_size):
        if not self.has_gpu:
            return False
        return data_size > 10000 and self.gpu_memory_gb > 2

    def print_summary(self):
        print("=" * 80)
        print("SYSTEM RESOURCE ANALYSIS")
        print("=" * 80)
        print(f"CPU Cores (Physical): {self.cpu_count_physical}")
        print(f"CPU Cores (Logical):  {self.cpu_count_logical}")
        print(f"Total RAM:            {self.total_ram_gb:.2f} GB")
        print(f"Available RAM:        {self.available_ram_gb:.2f} GB")
        print(f"GPU Available:        {'Yes' if self.has_gpu else 'No'}")
        if self.has_gpu:
            print(f"GPU Count:            {self.gpu_count}")
            print(f"GPU Memory:           {self.gpu_memory_gb:.2f} GB per GPU")
        print("=" * 80)
        print(f"Optimal Workers (CPU Intensive):    {self.get_optimal_workers('cpu_intensive')}")
        print(f"Optimal Workers (Memory Intensive): {self.get_optimal_workers('memory_intensive')}")
        print(f"Optimal Workers (I/O Intensive):    {self.get_optimal_workers('io_intensive')}")
        print("=" * 80)
