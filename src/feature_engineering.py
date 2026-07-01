"""
Feature engineering: rolling stats, rate-of-change, and failure labels.

Key idea: a single sensor reading isn't very predictive, but the TREND
(is it rising/falling over the last N cycles?) is. This is what turns raw
sensor logs into ML-ready features.
"""

import pandas as pd

SENSOR_COLS = ["temperature", "vibration", "pressure", "rotation_speed", "oil_quality"]
WINDOW = 10                 # rolling window size (cycles)
FAILURE_HORIZON = 30        # label = 1 if failure within next N cycles


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grouped = df.groupby("unit_id")

    for col in SENSOR_COLS:
        df[f"{col}_roll_mean"] = grouped[col].transform(
            lambda s: s.rolling(WINDOW, min_periods=1).mean()
        )
        df[f"{col}_roll_std"] = grouped[col].transform(
            lambda s: s.rolling(WINDOW, min_periods=1).std().fillna(0)
        )
        # rate of change vs WINDOW cycles ago
        df[f"{col}_rate_of_change"] = grouped[col].transform(
            lambda s: s.diff(WINDOW).fillna(0)
        )

    return df


def add_failure_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # RUL already computed in generate_data.py; if using real C-MAPSS data,
    # compute RUL as (max_cycle - cycle) per unit first.
    df["failure_within_horizon"] = (df["RUL"] <= FAILURE_HORIZON).astype(int)
    return df


def build_features(in_path: str = "data/processed/clean_data.csv",
                    out_path: str = "data/processed/features.csv") -> pd.DataFrame:
    df = pd.read_csv(in_path)
    df = add_rolling_features(df)
    df = add_failure_label(df)
    df.to_csv(out_path, index=False)
    print(f"Feature-engineered data saved -> {out_path} ({len(df)} rows, "
          f"{df.shape[1]} columns)")
    print(f"Class balance:\n{df['failure_within_horizon'].value_counts(normalize=True)}")
    return df


if __name__ == "__main__":
    build_features()
