from data_preprocessing import preprocess_data
from model_training import train_model
from evaluation import evaluate_model
import time


def main():
    train_data, test_data = preprocess_data()
    model = train_model(train_data)
    # need sleep to save all models
    time.sleep(5)
    scores = evaluate_model(model, test_data)
    print(scores)


if __name__ == "__main__":
    main()
