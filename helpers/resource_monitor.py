import numpy as np
import time
import psutil
import threading

class ResourceMonitor:
    def __init__(self):
        self.monitoring = False
        self.thread = None
        self.peak_ram_gb = 0
        self.peak_cpu_percent = 0
        self.ram_samples = []
        self.cpu_samples = []
        self.process = psutil.Process()

    def _monitor_loop(self):
        while self.monitoring:
            ram_gb = self.process.memory_info().rss / (1024 ** 3)
            self.ram_samples.append(ram_gb)
            self.peak_ram_gb = max(self.peak_ram_gb, ram_gb)

            try:
                cpu_percent = self.process.cpu_percent(interval=0.1)
                self.cpu_samples.append(cpu_percent)
                self.peak_cpu_percent = max(self.peak_cpu_percent, cpu_percent)
            except:
                pass

            time.sleep(0.5)

    def start(self):
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=2)

    def get_stats(self):
        return {
            'peak_ram_gb': self.peak_ram_gb,
            'avg_ram_gb': np.mean(self.ram_samples) if self.ram_samples else 0,
            'peak_cpu_percent': self.peak_cpu_percent,
            'avg_cpu_percent': np.mean(self.cpu_samples) if self.cpu_samples else 0
        }
