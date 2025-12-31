
import time

class TimingLogger:
    def __init__(self):
        self.stages = {}
        self.current_stage = None
        self.stage_start = None

    def start_stage(self, name):
        self.current_stage = name
        self.stage_start = time.time()

    def end_stage(self):
        if self.current_stage and self.stage_start:
            duration = time.time() - self.stage_start
            self.stages[self.current_stage] = duration
            self.current_stage = None
            self.stage_start = None

    def get_summary(self):
        return self.stages.copy()
