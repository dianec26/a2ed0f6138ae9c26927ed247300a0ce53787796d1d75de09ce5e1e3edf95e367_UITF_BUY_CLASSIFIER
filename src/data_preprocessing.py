import os
import sys
import time
import pandas as pd
import joblib
from datetime import datetime
from loguru import logger
from sklearn.preprocessing import StandardScaler
import feature_engineering as fe


def preprocess_data():
    # Configure loguru logger
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

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Start timing the entire process
    start_time = time.time()
    process_start = datetime.now()
    logger.info(
        f"Starting data preprocessing pipeline at\
            {process_start.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # Read the data
    logger.info("Reading input data from CSV file")
    data_read_start = time.time()
    df = pd.read_csv("data/ATRGFNP_hist_with_buy_signals.csv")
    data_read_time = time.time() - data_read_start
    logger.success(
        f"Data loaded successfully: {df.shape[0]} rows, \
            {df.shape[1]} columns (took {data_read_time:.2f}s)"
    )

    # Combine all feature engineering
    logger.info("Starting feature engineering for multiple timeframes")
    durations = [3, 7, 14, 21, 30]
    logger.info(f"Processing {len(durations)} timeframes: {durations} days")

    arr_df = []
    for i, days in enumerate(durations, 1):
        duration_start = time.time()
        logger.debug(f"Processing {days}-day features ({i}/{len(durations)})")

        feature_df, feature_cols = fe.calculate_technical_features(
            df, price_col="navpu_value", duration_days=days
        )
        arr_df.append(feature_df[feature_cols])

        duration_time = time.time() - duration_start
        logger.info(
            f"{days}-day features completed: {len(feature_cols)}\
            features created (took {duration_time:.2f}s)"
        )

    # Combine feature dataframes
    logger.info("Combining all engineered features")
    combine_start = time.time()
    feat_eng_df = pd.concat(arr_df, axis=1)
    combine_time = time.time() - combine_start
    logger.success(
        f"Feature combination completed: {feat_eng_df.shape[1]} \
            total engineered features (took {combine_time:.2f}s)"
    )

    # Combine original df and feature engineering
    logger.info("Merging original data with engineered features")
    merge_start = time.time()
    final_df = pd.concat([df, feat_eng_df], axis=1)
    initial = final_df.shape[0]
    merge_time = time.time() - merge_start
    logger.success(
        f"Data merge completed: {final_df.shape[0]} rows, \
            {final_df.shape[1]} columns (took {merge_time:.2f}s)"
    )

    # Remove null rows and problematic columns
    logger.info("Starting data cleaning process")
    cleaning_start = time.time()
    nans_to_drop = ["ma_30", "future_return_7d", "future_return_30d", "pct_change_30d"]

    final_df = final_df.dropna(subset=nans_to_drop)
    final_df = final_df.drop(["sr_position_3d"], axis=1)
    logger.debug(f"Dropping rows with NaN values in columns: {nans_to_drop}")
    final_df = final_df.reset_index(drop=True)

    rows_removed = initial - final_df.shape[0]
    cleaning_time = time.time() - cleaning_start
    print("Prev shape", initial)
    print("Removed rows", initial - final_df.shape[0])
    logger.info(
        f"Data cleaning completed: removed {rows_removed} \
            rows (took {cleaning_time:.2f}s)"
    )

    final_df = final_df.reset_index(drop=True)
    logger.debug("DataFrame index reset completed")

    # show distribution
    # display(final_df["buy"].value_counts())

    # Split train test
    logger.info("Starting train/test split process")
    split_start = time.time()
    train_len = int(final_df.shape[0] * 0.8)
    logger.debug(
        f"Train length calculated: {train_len} \
            rows (80% of {final_df.shape[0]})"
    )

    # Temporal split (first 80% for train, last 20% for test)
    train_df = final_df.iloc[:train_len].copy().set_index("date")
    test_df = final_df.iloc[train_len:].copy().set_index("date")

    split_time = time.time() - split_start
    print(
        f"Train set: {len(train_df)} \
          rows ({len(train_df)/len(final_df):.1%})"
    )
    print(
        f"Test set: {len(test_df)} \
          rows ({len(test_df)/len(final_df):.1%})"
    )
    logger.success(f"Temporal split completed (took {split_time:.2f}s)")

    # Define feature columns (exclude non-feature columns)
    logger.info("Defining feature columns")
    exclude_cols = ["original_label", "buy", "buy_reason", "buy_strength"]
    feature_cols = [
        col
        for col in final_df.columns
        if col not in exclude_cols and final_df[col].dtype in ["float64", "int64"]
    ]

    print(f"Total features: {len(feature_cols)}")
    print("Target variable: buy")
    logger.success(
        f"Feature selection completed: {len(feature_cols)} features identified"
    )

    # Separate features and target
    logger.info("Separating features and targets")
    feature_split_start = time.time()

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    feature_split_time = time.time() - feature_split_start
    logger.success(
        f"Feature/target separation completed (took {feature_split_time:.2f}s)"
    )

    # Initialize and fit scaler on training data only (prevent data leakage)
    logger.info("Starting feature scaling process")
    scaling_start = time.time()

    scaler = StandardScaler()
    logger.debug("Fitting StandardScaler on training data only")
    scaler.fit(X_train)

    # Transform both train and test sets using the fitted scaler
    logger.debug("Transforming training and test features")
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrames with original column names and indices
    X_train_scaled = pd.DataFrame(
        X_train_scaled, columns=feature_cols, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled, columns=feature_cols, index=X_test.index
    )

    scaling_time = time.time() - scaling_start
    logger.success(f"Feature scaling completed (took {scaling_time:.2f}s)")

    # Create complete scaled dataframes
    logger.info("Creating complete scaled dataframes")
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()

    # Replace the feature columns with scaled versions
    train_df_scaled[feature_cols] = X_train_scaled
    test_df_scaled[feature_cols] = X_test_scaled

    # Save the scaler for future use
    logger.info("Saving fitted scaler")
    save_start = time.time()

    scaler_path = "models/scaler.pkl"
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, scaler_path)

    save_time = time.time() - save_start
    print(f"\nScaler saved to: {scaler_path}")
    logger.success(f"Scaler saved to: {scaler_path} (took {save_time:.2f}s)")

    # Calculate total processing time
    total_time = time.time() - start_time

    # Final summary with timing
    logger.info("DATA PREPROCESSING PIPELINE COMPLETED")
    logger.info(
        f"Total processing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)"
    )
    logger.info(f"Original dataset: {initial} rows")
    logger.info(f"After cleaning: {len(final_df)} rows")
    logger.info(f"Training features shape: {X_train_scaled.shape}")
    logger.info(f"Test features shape: {X_test_scaled.shape}")
    logger.info(f"Total engineered features: {len(feature_cols)}")
    logger.info(f"Pipeline completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return train_df_scaled, test_df_scaled
