# credit-risk-engine
Quantatative risk framework to estimate expected credit loss by modelling probability of default on a home credit default dataset. Just a way for me to get more familiar with credit risk engineering.

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
3. run the model pipeline (Must have kaggle api key or csv in ./data/raw folder)
    ```bash
    uv run main.py
    ```

## Project Roadmap
- [x] Establish baseline Logistic Regression model (AUC: 0.725)
- [ ] **Bureau Data Integration**: Aggregate `bureau.csv` to capture historical credit behavior
- [ ] **Automated Feature Selection**: Implement correlation filters to prune redundant variables
- [ ] **Ratio Synthesis**: Develop debt-to-income and annuity-to-credit features

### Phase 2: Mathematical Research (Low-Level)
- [ ] **NumPy Engine**: Develop a vectorized Logistic Regression class from scratch
- [ ] **Optimization Suite**: Implement manual Sigmoid, Log-Loss, and Gradient Descent functions
- [ ] **Benchmarking**: Create a validation framework to compare manual model weights against `scikit-learn`

### Phase 3: Systems & Performance Engineering
- [ ] **Go Port**: Re-implement core math utilities in Go for compiled performance
- [ ] **Concurrency Lab**: Use Goroutines to benchmark parallel processing against Python’s GIL
- [ ] **Streaming Statistics**: Implement Welford’s Algorithm for out-of-core feature scaling
