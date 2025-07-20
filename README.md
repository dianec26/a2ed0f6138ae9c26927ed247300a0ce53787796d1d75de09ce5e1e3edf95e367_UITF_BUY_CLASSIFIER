# a2ed0f6138ae9c26927ed247300a0ce53787796d1d75de09ce5e1e3edf95e367_UITF_BUY_CLASSIFIER

### Project Overview: 
This project addresses a buy classification problem tailored for daily trading decisions in a Unit Investment Trust Fund (UITF) environment. The goal is to identify whether a "buy" signal should be triggered on a given day, based on price movements and technical indicators that capture short-term dips. Specifically, the model incorporates three key signals aligned with a "buy low" strategy: RSI Recovery (detecting rebound after oversold conditions), Dip Recovery (identifying short-term price recovery after a significant drop), and Near Support (recognizing when price approaches a recent 30-day low). These signals are well-suited to UITFs, which are priced only once daily and are not influenced by intraday fluctuations.

The dataset consists of daily Net Asset Value per Unit (NAVPU) prices of the ATRAM Global Financials Feeder Fund, scraped and cleaned from UITF.com.ph. I chose this fund because it is part of my personal investment portfolio, which I currently manage using an intuitive "buy when it seems low" approach. By formalizing this strategy through machine learning, I aim to validate and enhance my buying decisions using structured signals grounded in technical analysis.

### Data: 
- found in the data folder (data < 1mb)

### Folder Structure:
.<br>
└── uitf-buy-classifier/<br>
├── data/                        # Data files (raw, processed, etc.)
├── logs/          
├── models/          
├── reports/       
├── src/   
└── deploy/                      # Containerization and orchestration support
    ├── docker/                  # Dockerfiles and build assets
    │   ├── Dockerfile           # Container definition for ML app
    │   └── requirements.txt     # Python dependencies
    │
    └── airflow/                 # Airflow orchestration
        ├── dags/                # DAG definitions for ML workflows
        │   └── example_dag.py   # Example DAG script
        │
        ├── logs/                # Log folder (placeholder, auto-managed by Airflow)
        │   └── .gitkeep         # Keeps the directory in version control
        │
        └── config/              # (Optional) Custom Airflow settings
            └── airflow_local_settings.py
  
#### Folder Description
- data/: Contains the cleaned and feature-engineered datasets used to build and evaluate the model.
- logs/: Stores training and experiment logs for tracking progress and debugging.
- models/: Holds all saved model files, including experiment outputs and the preprocessing scaler, to ensure reproducibility.
- reports/: Includes visualizations, evaluation metrics, and summaries generated after model training.
- src/: Contains all source code such as scripts for data processing, feature engineering, training, and evaluation.

### Setup Instructions (Setting up UV): 
1. Install uv: `pip install uv`
2. Initialize uv: `uv init`
3. Define dependencies in pyproject.toml (optional if you already have the .toml file): `
   dependencies = [
    "loguru>=0.7.3",
    "matplotlib>=3.10.3",
    "pandas==2.0.0",
    "scikit-learn==1.3.0",
    "autogluon-tabular==1.0.0",
    "autogluon==1.0.0",
    "ipykernel>=6.29.5"
]`
4. Synchronize the libraries: `uv sync`

### Running the Pipeline:
1. Set up uv as seen in **Setup Instructions Section**
2. Activate the virtual environment: `source .venv/bin/activate`
3. Run the pipeline script: `Python src/run_pipeline.py`
      - script flow: data_processing.py > model_training.py > evaluation.py

### Precommit Configuration:
- repo: https://github.com/pre-commit/pre-commit-hooks:
  - checks for the trailing-whitespace, end-of-file-fixer and spell-check for the README.md file
- repo: https://github.com/psf/black:
  - Runs Black, an opinionated Python code formatter, to automatically format Python files to a consistent style (e.g., consistent indentation, line length, and quote usage).
 
### Reflection:
During the development of this project, one challenge I encountered was configuring pre-commit hooks, particularly ensuring that all lines adhered to the 88-character limit. This required manually reviewing and adjusting multiple lines while experimenting with different formatting tools to find a configuration that balanced readability and compliance. Additionally, switching Python versions within a virtual environment proved tricky—I initially attempted to change versions without deactivating the environment, which led to version conflicts until I properly deactivated and reconfigured it. Finally, installing AutoGluon posed dependency issues due to strict version requirements for scikit-learn and pandas, which I resolved by explicitly specifying compatible versions in the pyproject.toml file to maintain a consistent and functional environment.
