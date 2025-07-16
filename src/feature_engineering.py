def calculate_technical_features(df, price_col="navpu_value", duration_days=7):
    """
    Calculate technical indicators for a specific duration

    Args:
        df (pd.DataFrame): DataFrame with price data
        price_col (str): Column name containing price data
        duration_days (int): Number of days for calculations

    Returns:
        pd.DataFrame: DataFrame with new technical features added
    """
    df_copy = df.copy()

    # Create base name for features
    period_name = f"{duration_days}d"

    print(f"Calculating technical features for {duration_days} days period...")

    # 1. Percent Change
    pct_change_col = f"pct_change_{period_name}"
    df_copy[pct_change_col] = df_copy[price_col].pct_change(periods=duration_days)

    # 2. Moving Average
    ma_col = f"ma_{period_name}"
    df_copy[ma_col] = (
        df_copy[price_col].rolling(window=duration_days, min_periods=1).mean()
    )

    # 3. Support (Rolling Minimum)
    support_col = f"support_{period_name}"
    df_copy[support_col] = (
        df_copy[price_col].rolling(window=duration_days, min_periods=1).min()
    )

    # 4. Resistance (Rolling Maximum)
    resistance_col = f"resistance_{period_name}"
    df_copy[resistance_col] = (
        df_copy[price_col].rolling(window=duration_days, min_periods=1).max()
    )

    # 5. Additional derived features

    # Distance from moving average (normalized)
    ma_distance_col = f"ma_distance_{period_name}"
    df_copy[ma_distance_col] = (df_copy[price_col] - df_copy[ma_col]) / df_copy[ma_col]

    # Position within support-resistance range
    sr_position_col = f"sr_position_{period_name}"
    df_copy[sr_position_col] = (df_copy[price_col] - df_copy[support_col]) / (
        df_copy[resistance_col] - df_copy[support_col]
    )

    # Volatility (rolling standard deviation)
    volatility_col = f"volatility_{period_name}"
    df_copy[volatility_col] = (
        df_copy[price_col].rolling(window=duration_days, min_periods=1).std()
    )

    # Price relative to support/resistance
    price_vs_support_col = f"price_vs_support_{period_name}"
    df_copy[price_vs_support_col] = (
        df_copy[price_col] - df_copy[support_col]
    ) / df_copy[support_col]

    price_vs_resistance_col = f"price_vs_resistance_{period_name}"
    df_copy[price_vs_resistance_col] = (
        df_copy[price_col] - df_copy[resistance_col]
    ) / df_copy[resistance_col]

    # Moving average trend (slope)
    ma_trend_col = f"ma_trend_{period_name}"
    df_copy[ma_trend_col] = df_copy[ma_col].pct_change()

    new_features = [
        pct_change_col,
        ma_col,
        support_col,
        resistance_col,
        ma_distance_col,
        sr_position_col,
        volatility_col,
        price_vs_support_col,
        price_vs_resistance_col,
        ma_trend_col,
    ]

    print(f"✓ Added {len(new_features)} features for {duration_days}-day period")

    return df_copy, new_features
