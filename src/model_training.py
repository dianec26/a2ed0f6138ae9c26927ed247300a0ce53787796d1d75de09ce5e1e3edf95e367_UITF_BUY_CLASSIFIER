import sys
import pickle
import time
from datetime import datetime
from loguru import logger
from autogluon.tabular import TabularPredictor


def train_model(train_data):
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
    )
    logger.add(
        "logs/data_preprocessing_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {function} | {message}",
        level="DEBUG",
    )

    to_drop_final = ["original_label", "buy_reason", "buy_strength"]
    train = train_data.drop(to_drop_final, axis=1)
    process_start = datetime.now()
    logger.info(
        f"Starting model training at {process_start.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    predictor = TabularPredictor(label="buy", path="../").fit(train)
    # sleep for 5 seconds to save all models
    time.sleep(5)
    process_end = datetime.now()
    logger.info(
        f"Finished model training at {process_end.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    predictor.feature_importance(train).to_csv("logs/model_feature_importance.csv")
    logger.info("Logging feature importance")
    rankings = predictor.leaderboard()
    rankings.to_csv("logs/model_leaderboard.csv")

    logger.info("Logging model leaderboard")

    path = "models/"
    print(path + rankings["model"][0] + "/model.pkl")
    with open(path + rankings["model"][0] + "/model.pkl", "rb") as f:
        best_model = pickle.load(f)

    logger.info("Returning best model")

    return best_model
