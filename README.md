# credit-risk-engine
Quantatative risk framework to estimate expected credit loss by modelling probability of default on a home credit default dataset

## Requirements

1. [uv](https://docs.astral.sh/uv/): An extremely fast Python package manager used for this project.
2. **Kaggle Authentication**: Required for automated data ingestion.
   - Follow the [Kaggle CLI documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md) for setup.
   - Alternatively, manually download the CSVs from the [Home Credit Default Risk Competition](https://www.kaggle.com/competitions/home-credit-default-risk/overview).

# Getting Started
## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/rob-bevilacqua/credit-risk-engine.git
   cd credit-risk-engine
   ```
2. Sync packages
    ```bash
    uv sync
    ```
3. run the ingestion pipeline
    ```bash
    uv run main.py
    ```