# a2ed0f6138ae9c26927ed247300a0ce53787796d1d75de09ce5e1e3edf95e367_UITF_BUY_CLASSIFIER

### Project Overview: 
This project addresses a buy classification problem tailored for daily trading decisions in a Unit Investment Trust Fund (UITF) environment. The goal is to identify whether a "buy" signal should be triggered on a given day, based on price movements and technical indicators that capture short-term dips. Specifically, the model incorporates three key signals aligned with a "buy low" strategy: RSI Recovery (detecting rebound after oversold conditions), Dip Recovery (identifying short-term price recovery after a significant drop), and Near Support (recognizing when price approaches a recent 30-day low). These signals are well-suited to UITFs, which are priced only once daily and are not influenced by intraday fluctuations.

The dataset consists of daily Net Asset Value per Unit (NAVPU) prices of the ATRAM Global Financials Feeder Fund, scraped and cleaned from UITF.com.ph. I chose this fund because it is part of my personal investment portfolio, which I currently manage using an intuitive "buy when it seems low" approach. By formalizing this strategy through machine learning, I aim to validate and enhance my buying decisions using structured signals grounded in technical analysis.

### Data: 
- found in the data folder (data < 1mb)

### Folder Structure:
<pre><code>
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
        ├── dags/                # DAG definitions for ML workflows<br>
        │   └── example_dag.py   # Example DAG script<br>
        │
        ├── logs/                # Log folder (placeholder, auto-managed by Airflow)
        │   └── .gitkeep         # Keeps the directory in version control
        │
        └── config/              # (Optional) Custom Airflow settings
            └── airflow_local_settings.py
</code></pre>
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

### Docker Set up Instructions
1. Create a dockerfile 
2. Specify the Python version ```FROM python:3.10.13-slim```
3. Install Python
<code> RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*
</code>
4. Install UV ```RUN pip install uv```
5. Copy pyrproject.toml .  ```COPY pyproject.toml .```
6. Uv sync ```RUN uv sync```
7. Copy scripts ```COPY /src/ src/```
8. Run pipeline ```CMD ["python", "src/run_pipeline.py"]```
    
#### Containerize Your ML Pipeline with Docker:
- Run and build dockerfile (note we run this inside root folder)
    - Create a Docker ml-pipeline image using the dockerfile
      <pre> <code>
        docker build -f deploy/docker/Dockerfile -t a2ed0f6138ae9c26927ed247300a0ce53787796d1d75de09ce5e1e3edf95e367-ml-pipeline .
       </code></pre>
    - Mount the data and model and run the pipeline in the docker image
      <pre> <code> docker run --rm \
          -v "$(pwd)/data:/app/data" \
          -v "$(pwd)/models:/app/models" \
          a2ed0f6138ae9c26927ed247300a0ce53787796d1d75de09ce5e1e3edf95e367-ml-pipeline </code> </pre>

### Airflow Docker Compose Set up:
1. Extract the Docker Compose and follow the instructions here
    - https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html
2. For Local implementation, change the following:
    - Change ```CeleryExecutor``` to ```LocalExecutor```
    - For lightweight docker contianers: set ```AIRFLOW__CORE__LOAD_EXAMPLES: 'false'```
    - Remove the following block of code since we are using a local implementation:
      - ```flower```
      - ```airflow-worker```

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



