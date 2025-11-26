"""
Data loader for verse and poem clustering.
Handles CSV loading, validation, and preprocessing.
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerseDataLoader:
    """
    Loads and validates verse-level data from CSV files.
    
    Required columns: idoriginal_poem, id, verse, order, source_dataset
    Optional columns: idgroup (verse clusters), type_id (poem clusters)
    """
    
    REQUIRED_COLUMNS = ['idoriginal_poem', 'id', 'verse', 'order', 'source_dataset']
    OPTIONAL_COLUMNS = ['idgroup', 'type_id']
    
    def __init__(self, csv_path: str):
        """
        Initialize the data loader.
        
        Args:
            csv_path: Path to the CSV file
        """
        self.csv_path = csv_path
        self.df: Optional[pd.DataFrame] = None
        self.has_ground_truth: bool = False
        self.has_verse_gt: bool = False
        self.has_poem_gt: bool = False
        
    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        
        missing_cols = set(self.REQUIRED_COLUMNS) - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for ground truth
        self.has_verse_gt = 'idgroup' in self.df.columns
        self.has_poem_gt = 'type_id' in self.df.columns
        self.has_ground_truth = self.has_verse_gt or self.has_poem_gt
        
        logger.info(f"Loaded {len(self.df)} verses from {self.df['idoriginal_poem'].nunique()} poems")
        logger.info(f"Ground truth available - Verse: {self.has_verse_gt}, Poem: {self.has_poem_gt}")
        
        # Sort by poem and order for consistency
        self.df = self.df.sort_values(['idoriginal_poem', 'order']).reset_index(drop=True)

        self.df['verse'] = (
            self.df['verse']
            .fillna("")           # replace NaN with empty string
            .astype(str)          # ensure string
            .str.strip()          # trim whitespace
        )
        
        
        return self.df
    
    def get_sample(self, fraction: float = 0.01, random_state: int = 42) -> pd.DataFrame:
        """
        Get a random sample of verses for hyperparameter tuning.
        
        Args:
            fraction: Fraction of data to sample
            random_state: Random seed for reproducibility
            
        Returns:
            Sampled DataFrame
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        sample_df = self.df.sample(frac=fraction, random_state=random_state)
        logger.info(f"Created sample with {len(sample_df)} verses ({fraction*100:.1f}%)")
        
        return sample_df
    
    def get_poems_dataframe(self) -> pd.DataFrame:
        """
        Create a poem-level dataframe from verse-level data.
        
        Returns:
            DataFrame with one row per poem
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        poem_data = []
        for poem_id, group in self.df.groupby('idoriginal_poem'):
            poem_info = {
                'idoriginal_poem': poem_id,
                'verse_count': len(group),
                'source_dataset': group['source_dataset'].iloc[0],
                'verse_ids': list(group['id'].values)
            }
            
            if self.has_poem_gt:
                poem_info['type_id'] = group['type_id'].iloc[0]
            
            poem_data.append(poem_info)
        
        poems_df = pd.DataFrame(poem_data)
        logger.info(f"Created poems dataframe with {len(poems_df)} poems")
        
        return poems_df
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        stats = {
            'total_verses': len(self.df),
            'total_poems': self.df['idoriginal_poem'].nunique(),
            'avg_verses_per_poem': len(self.df) / self.df['idoriginal_poem'].nunique(),
            'has_verse_ground_truth': self.has_verse_gt,
            'has_poem_ground_truth': self.has_poem_gt,
        }
        
        if self.has_verse_gt:
            stats['verse_clusters'] = self.df['idgroup'].nunique()
            stats['avg_verses_per_cluster'] = len(self.df) / self.df['idgroup'].nunique()
        
        if self.has_poem_gt:
            poems_df = self.get_poems_dataframe()
            stats['poem_clusters'] = poems_df['type_id'].nunique()
            stats['avg_poems_per_cluster'] = len(poems_df) / poems_df['type_id'].nunique()
        
        return stats