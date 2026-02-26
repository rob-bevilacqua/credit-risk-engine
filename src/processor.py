import pandas as pd
import numpy as np
from pathlib import Path

from src.utils import TypeOptimizer

class DataProcessor:
    """
    Class for processing csv data into a format for regression
    Currently only makes use of the main "application_train" dataset
    Goal is to eventually join tables to improve performance
    """
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

        bureau_df = pd.read_csv(self.raw_dir / "bureau.csv")
        bureau_agg = self._agg_bureau(bureau_df)

        df = df.merge(bureau_agg, on="SK_ID_CURR", how="left")

        # feature extraction
        df = self._add_ratios(df)

        df = self._prune_columns(df)

        # deep copy to unfragment
        df = df.copy() 
    
        df = TypeOptimizer.optimize(df)

        output_file = self.processed_dir / "train_transformed.parquet"
        df.to_parquet(output_file, engine="pyarrow", index=False)
        print(f"Success! Processed data saved to: {output_file}")

    def _clean_financials(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles 0-income edge cases and creates binary flags using a dictionary to prevent fragmentation."""
        
        # Pre-calculate median and replace 0 with nan so it isnt factored to median
        income_median = df['AMT_INCOME_TOTAL'].replace(0, np.nan).median()
        
        new_features = {}
        
        # new features mapped to dict to avoid fragmentation
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
        # avoid /0 error
        eps = 1e-6
        
        #columns containing info on flag documents
        doc_cols = [c for c in df.columns if 'FLAG_DOCUMENT' in c]

        # Create a dictionary for the new features
        feature_dict = {
            'CREDIT_TO_INCOME_RATIO': df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + eps),
            'ANNUITY_TO_INCOME_RATIO': df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + eps),
            'GOODS_PRICE_TO_CREDIT_RATIO': df['AMT_GOODS_PRICE'] / (df['AMT_CREDIT'] + eps),
            'DAYS_EMPLOYED_PERCENT': df['DAYS_EMPLOYED'] / df['DAYS_BIRTH'],
            'BU_UTILIZATION_RATIO': df['BU_TOTAL_DEBT_SUM'] / (df['BU_TOTAL_LIMIT_SUM'] + eps),
            'DOCUMENTATION_COMPLETENESS': df[doc_cols].sum(axis=1) / len(doc_cols)
        }
        
        # Join everything at once
        # This prevents fragmentation warnings
        df = pd.concat([df, pd.DataFrame(feature_dict, index=df.index)], axis=1)
        
        #ratio for documents now so i can drop the old columns
        df = df.drop(columns = doc_cols)

        return df
    
    def _agg_bureau(self, bureau_df: pd.DataFrame) -> pd.DataFrame:
        """
        Collects past credit behaviour from bureau.csv, so it can be appended to the applicants application entry
        """
        eps = 1e-6
        #flag active loans
        bureau_df['IS_ACTIVE'] = (bureau_df['CREDIT_ACTIVE'] == 'Active').astype(int)
        
        agg_logic = {
        'SK_ID_BUREAU': 'count', # how many past loans
        'IS_ACTIVE': 'sum', # how many are active
        'DAYS_CREDIT': 'mean', # avg length of 
        'CREDIT_DAY_OVERDUE': 'mean', #avg length of overdue credit
        'AMT_CREDIT_MAX_OVERDUE': 'max', #longest
        'AMT_CREDIT_SUM': 'sum', 
        'AMT_CREDIT_SUM_DEBT': 'sum',
        'AMT_CREDIT_SUM_LIMIT': 'sum',
        'CNT_CREDIT_PROLONG': 'sum'
        }

        bureau_agg = bureau_df.groupby('SK_ID_CURR').agg(agg_logic)

        bureau_agg.columns = [
        'BU_LOAN_COUNT',
        'BU_ACTIVE_LOAN_COUNT',
        'BU_DAYS_CREDIT_MEAN',
        'BU_AVG_DAYS_OVERDUE',
        'BU_MAX_OVERDUE',
        'BU_TOTAL_CREDIT_SUM',
        'BU_TOTAL_DEBT_SUM',
        'BU_TOTAL_LIMIT_SUM',
        'BU_TOTAL_PROLONG_SUM'
        ]

        return bureau_agg
    
    def _get_correlated_columns(self, df: pd.DataFrame, threshold = 0.8):
        correlation_mat = df.corr().abs()

        upper = correlation_mat.where(
        np.triu(np.ones(correlation_mat.shape), k=1).astype(bool)
        )

        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        return to_drop
    
    def _prune_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates correlation between columns and prunes redundant variables.
        """
        exclude = ['SK_ID_CURR', 'TARGET']
        cols_to_check = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
        
        to_drop = self._get_correlated_columns(df[cols_to_check])

        return df.drop(columns=to_drop)