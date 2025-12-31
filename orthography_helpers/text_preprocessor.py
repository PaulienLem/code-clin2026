#%% md
# 1. DBBE
#%%
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, v_measure_score
class TextPreprocessor:
    def __init__(self, lowercase=True, remove_punctuation=True, remove_diacritics=True):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_diacritics = remove_diacritics
        if remove_punctuation:
            self.punct_pattern = re.compile(r'[^\w\s]', re.UNICODE)
            self.remove_chars_pattern = re.compile(r'[\(\)\{\}]')

    def _remove_diacritics(self, text: str) -> str:
        return ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )

    def preprocess(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text) if pd.notna(text) else ''

        if self.remove_diacritics:
            text = self._remove_diacritics(text)
        if self.lowercase:
            text = text.lower()
        if self.remove_punctuation:
            text = self.remove_chars_pattern.sub('', text)
            text = self.punct_pattern.sub(' ', text)

        return ' '.join(text.split())

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        return [self.preprocess(t) for t in texts]
