"""
Plotting the results from using TSAI / A ML model to predict on un/compressed data sets
"""

import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_PATH = "data/tsai_results/BIDMC32_dim_RESP.csv"

def plot_ml_results(filepath: str) -> None:
    df = pd.read_csv(filepath, delimiter=",")

    title = filepath.split("/")[-1].split(".")[0]

    df["rmse"] = df["rmse"].astype(np.float64)

    try:
        df[["e", "eb", "label"]] = df.apply(
            to_float,
            axis=1,
            result_type="expand"
        )
    except ValueError as e:
        print(f"error: {e}")
        return

    fig, ax = plt.subplots(figsize=(16, 6))

    # Unique eb values for coloring
    unique_eb = sorted(df["eb"].dropna().unique())

    cmap = plt.get_cmap("viridis")

    color_map = {
        eb: cmap(i / max(len(unique_eb) - 1, 1))
        for i, eb in enumerate(unique_eb)
    }

    labels = []
    colors = []

    for _, row in df.iterrows():

        # TRAIN / TEST
        if pd.notna(row["label"]):
            labels.append(row["label"])
            colors.append("gray")

        # Compressed rows
        else:
            labels.append(f"e={row['e']}")
            colors.append(color_map[row["eb"]])

    # Numeric x positions
    x = np.arange(len(df))

    # Bar plot
    ax.bar(x, df["rmse"], color=colors)

    # Keep duplicate e labels
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    # Legend for eb colors
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_map[eb])
        for eb in unique_eb
    ]

    legend_labels = [f"eb={eb}" for eb in unique_eb]

    ax.legend(handles, legend_labels, title="eb")

    ax.set_xlabel("e")
    ax.set_ylabel("RMSE")
    ax.set_title(title)

    plt.tight_layout()
    plt.show()


def to_float(row):
    filename = str(row["filename"])

    # Extract e and eb
    gr = re.search(
        r'e([-+]?\d*\.?\d+)_eb([-+]?\d*\.?\d+)',
        filename
    )

    if gr:
        return float(gr.group(1)), float(gr.group(2)), None

    # Extract TRAIN / TEST
    gr2 = re.search(r'(TRAIN|TEST)', filename)

    if gr2:
        return np.nan, np.nan, gr2.group(1)

    raise ValueError(f"Could not parse filename: {filename}")

if __name__ == "__main__":
    plot_ml_results(RESULTS_PATH)