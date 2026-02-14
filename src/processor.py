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

        # deep copy to unfragment
        df = df.copy() 
    
        df = TypeOptimizer.optimize(df)

        output_file = self.processed_dir / "train_transformed.parquet"
        df.to_parquet(output_file, engine="pyarrow", index=False)
        print(f"Success! Processed data saved to: {output_file}")

    def _clean_financials(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles 0-income edge cases and creates binary flags using a dictionary to prevent fragmentation."""
        
        # Pre-calculate median 
        income_median = df['AMT_INCOME_TOTAL'].replace(0, np.nan).median()
        
        new_features = {}
        
        new_features['FLAG_ZERO_INCOME'] = (df['AMT_INCOME_TOTAL'] == 0).astype(int)
        
        # 70-year threshold for days employed seems fair
        new_features['DAYS_EMPLOYED_ANOM'] = (df["DAYS_EMPLOYED"] >= 25567).astype(int)
        
        # Use pd.concat to add all new columns at once so it doesnt complain
        df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        
        # Modifying existing columns in-place should be alright
        df['AMT_INCOME_TOTAL'] = df['AMT_INCOME_TOTAL'].replace(0, np.nan).fillna(income_median)
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace({365243: np.nan, 25567: np.nan})

        return df
    
    def _add_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates standard credit risk ratios using a non-fragmenting approach."""
        eps = 1e-6
        
        # Create a dictionary for the new features
        feature_dict = {
            'CREDIT_TO_INCOME_RATIO': df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + eps),
            'ANNUITY_TO_INCOME_RATIO': df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + eps),
            'GOODS_PRICE_TO_CREDIT_RATIO': df['AMT_GOODS_PRICE'] / (df['AMT_CREDIT'] + eps),
            'DAYS_EMPLOYED_PERCENT': df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
        }
        
        # Join everything at once
        # This prevents fragmentation warnings
        return pd.concat([df, pd.DataFrame(feature_dict, index=df.index)], axis=1)