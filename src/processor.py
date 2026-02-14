import pandas as pd
import numpy as np
from pathlib import Path

from src.utils import TypeOptimizer

class DataProcessor:
    """Class for processing csv data into a format for regression"""
    def __init__(self, raw_dir: Path, processed_dir: Path):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.processed_dir.mkdir(parents = True, exist_ok = True)
    
    def process_pipeline(self):
        """Main processing pipeline"""
        print("Starting data processing")
        # main csv
        df = pd.read_csv(self.raw_dir / "application_train.csv")

        # Handle financial entry outliers
        df = self._clean_financials(df)

        # feature extraction
        df = self._add_ratios(df)

        output_file = self.processed_dir / "train_transformed.parquet"
        df.to_parquet(output_file, engine="pyarrow", index=False)
        print(f"Success! Processed data saved to: {output_file}")

    def _clean_financials(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles 0-income edge cases and creates binary flags."""
        # Create a flag for zero income
        df['FLAG_ZERO_INCOME'] = (df['AMT_INCOME_TOTAL'] == 0).astype(int)
        
        # Replace 0 with nan
        df['AMT_INCOME_TOTAL'] = df['AMT_INCOME_TOTAL'].replace(0, np.nan)
        df['AMT_INCOME_TOTAL'] = df['AMT_INCOME_TOTAL'].fillna(df['AMT_INCOME_TOTAL'].median())
        
        # nan for employed > 70 years, as its an outlier
        df['DAYS_EMPLOYED_ANOM'] = df["DAYS_EMPLOYED"] >= 25567
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace({25567: np.nan})

        return df
    
    def _add_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates standard credit risk ratios."""
        # Avoid division by zero with a small epsilon
        eps = 1e-6
        
        df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + eps)
        df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + eps)
        df['GOODS_PRICE_TO_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / (df['AMT_CREDIT'] + eps)
        
        # Age-based features
        df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
        
        return df