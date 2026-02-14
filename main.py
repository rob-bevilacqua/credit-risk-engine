from src.datafetch import Datafetcher
from pathlib import Path
from src.processor import DataProcessor
from src.model import RiskModel

def main():
    # Setup Paths
    base_path = Path(__file__).parent
    raw_data_dir = base_path / "data" / "raw"
    processed_data_dir = base_path / "data" / "processed"
    model_dir = base_path / "models"
    
    processed_file = processed_data_dir / "train_transformed.parquet"

    # Extraction to csv
    fetcher = Datafetcher()
    # Check if data exists before downloading
    if not (raw_data_dir / "application_train.csv").exists():
        fetcher.download_data(raw_data_dir)

    # Transformation CSV -> Optimized Parquet
    processor = DataProcessor(raw_data_dir, processed_data_dir)
    processor.process_pipeline()

    # Modeling Parquet -> Logistic Regression
    # This performs the split, scaling, and training
    trainer = RiskModel(model_dir)
    trainer.run_training_pipeline(processed_file)

    print("\nFull Pipeline execution complete.")

if __name__ == "__main__":
    main()
