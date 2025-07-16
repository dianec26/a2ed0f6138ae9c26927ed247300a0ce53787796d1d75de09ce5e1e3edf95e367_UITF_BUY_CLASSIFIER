from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(model, test_data):
    """
    Calculate comprehensive classification metrics

    Args:
        y_true: Actual/true labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional, for ROC AUC)

    Returns:
        dict: Dictionary containing all metrics
    """
    y_true = test_data["buy"].copy()
    y_pred = model.predict(test_data.drop("buy", axis=1))

    metrics = {}

    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    metrics["f1"] = f1_score(y_true, y_pred, average="weighted")
    metrics["precision"] = precision_score(y_true, y_pred, average="weighted")
    metrics["recall"] = recall_score(y_true, y_pred, average="weighted")
    return metrics
