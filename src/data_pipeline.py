"""
Data ingestion + cleaning pipeline.
Takes raw sensor CSV -> handles missing values -> ensures correct types
-> sorts by unit/cycle -> outputs clean dataframe.

This is the "Data Engineer" evidence for your project.
"""

import pandas as pd
import os


def load_raw_data(path: str = "data/raw/sensor_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # sort so rolling/window features are computed in correct time order
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)

    # handle missing sensor values: forward-fill within each unit, then
    # fall back to column median for any remaining gaps
    sensor_cols = [c for c in df.columns if c not in
                   ("unit_id", "cycle", "max_cycle", "RUL")]

    df[sensor_cols] = (
        df.groupby("unit_id")[sensor_cols]
        .apply(lambda g: g.ffill().bfill())
        .reset_index(drop=True)
    )
    df[sensor_cols] = df[sensor_cols].fillna(df[sensor_cols].median())

    # drop any fully duplicate rows
    df = df.drop_duplicates()

    return df


def run_pipeline(raw_path: str = "data/raw/sensor_data.csv",
                  out_path: str = "data/processed/clean_data.csv") -> pd.DataFrame:
    df = load_raw_data(raw_path)
    df = clean_data(df)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Cleaned data saved -> {out_path} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    run_pipeline()
