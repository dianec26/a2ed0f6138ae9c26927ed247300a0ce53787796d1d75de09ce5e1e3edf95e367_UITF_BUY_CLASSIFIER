# a2ed0f6138ae9c26927ed247300a0ce53787796d1d75de09ce5e1e3edf95e367_UITF_BUY_CLASSIFIER

## Project Overview: 
This project addresses a buy classification problem tailored for daily trading decisions in a Unit Investment Trust Fund (UITF) environment. The goal is to identify whether a "buy" signal should be triggered on a given day, based on price movements and technical indicators that capture short-term dips. Specifically, the model incorporates three key signals aligned with a "buy low" strategy: RSI Recovery (detecting rebound after oversold conditions), Dip Recovery (identifying short-term price recovery after a significant drop), and Near Support (recognizing when price approaches a recent 30-day low). These signals are well-suited to UITFs, which are priced only once daily and are not influenced by intraday fluctuations.

The dataset consists of daily Net Asset Value per Unit (NAVPU) prices of the ATRAM Global Financials Feeder Fund, scraped and cleaned from UITF.com.ph. I chose this fund because it is part of my personal investment portfolio, which I currently manage using an intuitive "buy when it seems low" approach. By formalizing this strategy through machine learning, I aim to validate and enhance my buying decisions using structured signals grounded in technical analysis.

## Data: 
- found in the data folder (data < 1mb)

## Folder Structure:
<pre><code>
└── uitf-buy-classifier/<br>
├── data/                            # Data files (raw, processed, etc.) 
├── logs/                            # Model logs an artifact, logs from loguru and model performance
├── models/          
├── reports/       
├── src/   
└── deploy/                          # Containerization and orchestration support
    ├── docker/                      # Dockerfiles and build assets
    │   └── dockerfile               # Container definition for ML app
    │
    └── airflow/                     # Airflow orchestration
        ├── dags/                    # DAG definitions for ML workflows<br>
        │   └── ml_pipeline_dag,py   # Example DAG script<br>
        │
        ├── logs/                    # Log folder (placeholder, auto-managed by Airflow)
        │   └── .gitkeep             # Keeps the directory in version control
        │
        ├── plugins/
        │
        └── config/                  # (Optional) Custom Airflow settings
            └── airflow_local_settings.py
</code></pre>

#### Folder Description
- data/: Contains the cleaned and feature-engineered datasets used to build and evaluate the model.
- logs/: Stores training and experiment logs for tracking progress and debugging.
- models/: Holds all saved model files, including experiment outputs and the preprocessing scaler, to ensure reproducibility.
- reports/: Includes visualizations, evaluation metrics, and summaries generated after model training.
- src/: Contains all source code such as scripts for data processing, feature engineering, training, and evaluation.

## Environment Setups:

### Local Setup (Setting up UV): 
1. Install uv: `pip install uv`
2. Initialize uv: `uv init`
3. Define dependencies in pyproject.toml (optional if you already have the .toml file):
<pre><code>
   dependencies = [
    "loguru>=0.7.3",
    "matplotlib>=3.10.3",
    "pandas==2.0.0",
    "scikit-learn==1.3.0",
    "autogluon-tabular==1.0.0",
    "autogluon==1.0.0",
    "ipykernel>=6.29.5"
]` </code> </pre>
5. Synchronize the libraries: `uv sync` 
   
### Running the Pipeline:
1. Set up uv as seen in **Setup Instructions Section**
2. Activate the virtual environment: `source .venv/bin/activate`
3. Run the pipeline script: `Python src/run_pipeline.py`
      - script flow: data_processing.py > model_training.py > evaluation.py

## Docker
**Note: This was only used for running the ml-pipeline; this wasn't used as container for airflow **

### Docker Main Process: 
1. Docker Installation: install Docker Desktop (https://docs.docker.com/engine/install/)
3. Dockerfile Creation:  Docker builds images by reading the instructions from a Dockerfile. A Dockerfile is a text file containing instructions for building your source code. see [Dockerfile Description](# Dockerfile Description)
4. Image Building: Build an image using the docker build command. This reads the Dockerfile and packages the application into an image that can be run as a container.
5. Volume Strategy: Volumes are used to persist and share data between the host machine and the Docker container. This allows the pipeline to read input data and write outputs (such as models or logs) without modifying the container image.

### Steps to run a Dockerfile :
1. Dockerfile
2. Build/run commands
3. Volume Strategies
   
#### 1. Dockerfile:
This project uses a lightweight Docker image to run a Python 3.10 application with minimal overhead. <br>
1. Base Image: python:3.10.13-slim provides a small and secure foundation for running Python apps.
2. System Packages: build-essential, gcc, and curl are installed to support any Python packages that require compilation.
3. Dependency Management: The project uses uv for fast, reproducible dependency resolution via pyproject.toml.
    - Note: uv install caused issues during Docker build, likely due to how it tries to create or manage virtual environments in a non-standard context. To avoid this, uv sync was used instead, which installs dependencies as declared in the lockfile without resolving them again.
4. App Structure: Source code is placed in /src/, and the main entry point is src/run_pipeline.py.
5. Startup Command: The application runs using the Python binary inside the .venv created by uv.
   
This setup avoids unnecessary layers and complexity while ensuring fast dependency management and a consistent runtime environment.

#### 2. Build/run commands
Run and build the Dockerfile (note: this is run from the root folder):
- Create a Docker ml-pipeline image using the Dockerfile:
<pre> <code>
docker build -f deploy/docker/Dockerfile -t a2ed0f6138ae9c26927ed247300a0ce53787796d1d75de09ce5e1e3edf95e367-ml-pipeline .
</code></pre>

#### 3.  Volume Strategies
Mount the data and model directories and run the pipeline in the Docker image:
- data volume: Mounts the local data/ folder into the container at /app/data, so the pipeline can access raw input files.
- models volume: Mounts the local models/ folder into the container at /app/models, allowing the pipeline to save trained models back to the host machine.
<pre> <code>docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" 2ed0f6138ae9c26927ed247300a0ce53787796d1d75de09ce5e1e3edf95e367-ml-pipeline</code></pre>
- Implication: Clear separation between code (in the image) and runtime data (in mounted volumes), which aligns with container best practices.
  

## Airflow 

### Docker Compose Setup:
1. Fetch the Docker Compose and follow the instructions here
    - https://airflow.apache.org/docs/apache-airflow/3.0.3/docker-compose.yaml
2. Customization docker-compoase.yaml:
    - For Local implementation, change the following:
       1. Change ```CeleryExecutor``` to ```LocalExecutor```
       2. For lightweight docker contianers: set ```AIRFLOW__CORE__LOAD_EXAMPLES: 'false'```
       3. Remove the following block of code since we are using a local implementation:
          - ```flower```
          - ```airflow-worker```
       4. Set the pip requirements in x-airflow-common
            <pre><code>
            _PIP_ADDITIONAL_REQUIREMENTS: > loguru==0.7.3 matplotlib==3.10.3 pandas==2.0.0 scikit-learn==1.3.0 xgboost==2.0.3 numpy==1.24.3
            </code></pre>
       5. Mounting the docker-compose can use the scripts:
        <pre><code>
            #format is (local path): (path inside the docker image)
              volumes:
                - ./data:/opt/airflow/data
                - ./pyproject.toml:/opt/airflow/pyproject.toml
                - ./src:/opt/airflow/src
                - ./deploy/airflow/dags:/opt/airflow/dags
                - ./deploy/airflow/logs:/opt/airflow/logs
                - ./deploy/airflow/config:/opt/airflow/config
                - ./deploy/airflow/plugins:/opt/airflow/plugins
        </code></pre>
3. Set environment: ```echo -e "AIRFLOW_UID=$(id -u)" > .env```
4. Initialize database: ```docker compose up airflow-init```
5. Run docker compose: ```docker compose up```
6. Shutdown docker compose: ```docker compose down -v```
**Basic commands:**
    - Cleaning up: ```docker compose down --volumes --rmi all```
    
      
note: install airflow in the docker image `uv pip install "apache-airflow==3.0.3”`
### DAG Design:

#### Dag Structure
The ml_pipeline_dag is a machine learning workflow designed to automate the complete ML lifecycle for UITF data processing. This DAG orchestrates data preprocessing, model training, and evaluation in a sequential pipeline using Apache Airflow.

#### Airflow DAG
<pre><code> get_data_task >> model_training_task >> model_evaluation_task</code></pre>

#### Tasks
1.
2.
3. 


**Dependencies**

#### Scheduling Rationale
- Scheduling Settings:
<pre><code>
Schedule: None (Manual triggering only)
Start Date: January 1, 2025
Catchup: False (No historical runs)
</code></pre>
- On-demand execution: This ML pipeline pends on data availability rather than fixed schedules. Additionally since the data is static thus would only require to run the model once tthe data is updated.




## Precommit Configuration:
- repo: https://github.com/pre-commit/pre-commit-hooks:
  - checks for the trailing whitespace, ,end-of-file fixer and spell-check for the README.md file
- repo: https://github.com/psf/black:
  - Runs Black, an opinionated Python code formatter, to automatically format Python files to a consistent style (e.g., consistent indentation, line length, and quote usage).
- repo: https://github.com/adrienverge/yamllint.git
   - Check for syntax validity, but for weirdnesses like key repetition and cosmetic problems such as line length, trailing spaces, indentation
     
## Reflection:
During the development of this project, one challenge I encountered was configuring pre-commit hooks, particularly ensuring that all lines adhered to the 88-character limit. This required manually reviewing and adjusting multiple lines while experimenting with different formatting tools to find a configuration that balanced readability and compliance. Switching Python versions within a virtual environment proved tricky—I initially attempted to change versions without deactivating the environment, which led to version conflicts until I properly deactivated and reconfigured it. Finally, installing AutoGluon posed dependency issues due to strict version requirements for scikit-learn and pandas, which I resolved by explicitly specifying compatible versions in the pyproject.toml file to maintain a consistent and functional environment.



### Resources
- Docker
    - Docker Concepts: https://github.com/dianec26/a2ed0f6138ae9c26927ed247300a0ce53787796d1d75de09ce5e1e3edf95e367_UITF_BUY_CLASSIFIER/blob/hw2-docker-airflow/README.md
    - Docker Compose: https://adamtheautomator.com/docker-compose-tutorial/