from src.datafetch import Datafetcher
from pathlib import Path
from src.processor import DataProcessor

def main():
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    fetch = Datafetcher()
    if not any(data_dir.iterdir()):
        fetch.download_data(data_dir)
    # data downloaded and unzipped
    # next task, transform data
    proc_dir = Path("data/processed")
    proc = DataProcessor(data_dir, proc_dir)
    proc.process_pipeline()
    print("Hello from credit-risk-engine!")


if __name__ == "__main__":
    main()
