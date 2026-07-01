"""
Generates synthetic multi-sensor time-series data mimicking NASA's C-MAPSS
turbofan engine dataset structure. Each 'unit' runs for a random number of
cycles until failure, with sensor readings that drift/degrade as failure
approaches, plus realistic noise.

Run this first if you don't have the real NASA dataset downloaded.
Output: data/raw/sensor_data.csv
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N_UNITS = 100          # number of machines/engines
SENSORS = ["temperature", "vibration", "pressure", "rotation_speed", "oil_quality"]
MIN_CYCLES, MAX_CYCLES = 120, 350


def generate_unit(unit_id: int) -> pd.DataFrame:
    n_cycles = np.random.randint(MIN_CYCLES, MAX_CYCLES)
    cycles = np.arange(1, n_cycles + 1)

    df = pd.DataFrame({"unit_id": unit_id, "cycle": cycles})

    # degradation fraction: 0 at start, 1 at failure
    degradation = cycles / n_cycles

    for sensor in SENSORS:
        baseline = np.random.uniform(40, 60)
        drift_strength = np.random.uniform(15, 35)
        noise = np.random.normal(0, 1.5, size=n_cycles)
        # sensor value rises (or falls) nonlinearly as failure approaches
        direction = np.random.choice([1, -1])
        trend = direction * drift_strength * (degradation ** 2)
        df[sensor] = baseline + trend + noise

    df["max_cycle"] = n_cycles
    df["RUL"] = n_cycles - cycles  # remaining useful life
    return df


def main():
    all_units = [generate_unit(uid) for uid in range(1, N_UNITS + 1)]
    data = pd.concat(all_units, ignore_index=True)

    os.makedirs("data/raw", exist_ok=True)
    out_path = "data/raw/sensor_data.csv"
    data.to_csv(out_path, index=False)
    print(f"Generated {len(data)} rows across {N_UNITS} units -> {out_path}")
    print(data.head())


if __name__ == "__main__":
    main()
