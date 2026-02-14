import pandas as pd
import numpy as np

class TypeOptimizer:
    """Downcasts numeric types to save memory"""
    
    @staticmethod
    def optimize_floats(df: pd.DataFrame) -> pd.DataFrame:
        floats = df.select_dtypes(include=['float64']).columns
        df[floats] = df[floats].astype(np.float32)
        return df
    
    @staticmethod
    def optimize_ints(df: pd.DataFrame) -> pd.DataFrame:
        ints = df.select_dtypes(include=['int64']).columns
        for col in ints:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        return df
    
    @staticmethod
    def optimize_objects(df: pd.DataFrame) -> pd.DataFrame:
        # Columns with low cardinality (few unique values) are perfect for 'category'
        for col in df.select_dtypes(include=['object']).columns:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype('category')
        return df
    
    @classmethod
    def optimize(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Runs the full optimization suite."""
        
        df = cls.optimize_floats(df)
        df = cls.optimize_ints(df)
        df = cls.optimize_objects(df)
        
        return df