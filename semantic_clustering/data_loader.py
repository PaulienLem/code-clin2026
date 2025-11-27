import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.df = None
        self.subset_df = None
        
    def load_data(self):
        start = time.time()
        self.df = pd.read_csv(self.config.csv_path)
        
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['verse', 'id', 'idoriginal_poem'])
        self.df = self.df[self.df['verse'].str.strip() != '']
        filtered_count = len(self.df)
        
        load_time = time.time() - start
        
        has_verse_gt = self.config.verse_gt_col in self.df.columns
        has_poem_gt = self.config.poem_gt_col in self.df.columns
        
        return {
            'n_verses': filtered_count,
            'n_poems': self.df['idoriginal_poem'].nunique(),
            'has_verse_gt': has_verse_gt,
            'has_poem_gt': has_poem_gt,
            'load_time': load_time,
            'filtered_out': initial_count - filtered_count
        }
    
    def get_stratified_subset(self, fraction=None):
        if fraction is None:
            fraction = self.config.subset_fraction
            
        start = time.time()
        
        self.df['idoriginal_poem'] = self.df['idoriginal_poem'].astype(str)
        
        poem_counts = self.df['idoriginal_poem'].value_counts()
        valid_poems = poem_counts[poem_counts >= 2].index
        df_filtered = self.df[self.df['idoriginal_poem'].isin(valid_poems)]
        
        if len(df_filtered) == 0:
            self.subset_df = self.df.sample(frac=fraction, random_state=self.config.random_seed)
        else:
            try:
                self.subset_df, _ = train_test_split(
                    df_filtered,
                    train_size=fraction,
                    stratify=df_filtered['idoriginal_poem'],
                    random_state=self.config.random_seed
                )
            except (ValueError, TypeError):
                self.subset_df = df_filtered.sample(frac=fraction, random_state=self.config.random_seed)
        
        subset_time = time.time() - start
        
        return {
            'subset_size': len(self.subset_df),
            'subset_poems': self.subset_df['idoriginal_poem'].nunique(),
            'subset_time': subset_time
        }
    
    def get_full_data(self):
        return self.df
    
    def get_subset_data(self):
        return self.subset_df