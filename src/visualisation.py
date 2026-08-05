import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import math

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.serif": "Helvetica",
    "font.size": 10
})

snr_labels = {
    ("HouseholdPowerConsumption1", "0"): {
        0.0: "Uncompressed",
        4.0: "40 dB",
        8.0: "30 dB",
        16.0: "25 dB",
        32.0: "20 dB",
    },
    ("HouseholdPowerConsumption1", "1"): {
        0.0: "Uncompressed",
        0.0625: "40 dB",
        0.125: "35 dB",
        0.25: "30 dB",
        0.5: "25 dB",
        1.0: "20 dB",
    },
    ("HouseholdPowerConsumption1", "4"): {
        0.0: "Uncompressed",
        1.0: "40 dB",
        2.0: "20 dB",
    },
    ("HouseholdPowerConsumption2", "0"): {
        0.0: "Uncompressed",
        4.0: "40 dB",
        8.0: "30 dB",
        16.0: "25 dB",
        32.0: "20 dB",
    },
    ("HouseholdPowerConsumption2", "1"): {
        0.0: "Uncompressed",
        0.0625: "40 dB",
        0.125: "35 dB",
        0.25: "30 dB",
        0.5: "25 dB",
        1.0: "20 dB",
    },
    ("BIDMC32", "AVR"): {
        0.0: "Uncompressed",
        0.00390625: "40 dB",
        0.0078125: "35 dB",
        0.015625: "30 dB",
        0.03125: "25 dB",
        0.0625: "20 dB",
    }, 
    ("BIDMC32", "II"): {
        0.0: "Uncompressed",
        0.015625: "40 dB",
        0.03125: "30 dB",
        0.0625: "25 dB",
        0.125: "20 dB",
    },
    ("BIDMC32", "PLETH"): {
        0.0: "Uncompressed",
        0.0078125: "35 dB",
        0.015625: "30 dB",
        0.03125: "25 dB",
        0.0625: "20 dB",
    },
    ("BIDMC32", "RESP"): {
        0.0: "Uncompressed",
        0.0078125: "40 dB",
        0.015625: "35 dB",
        0.03125: "25 dB",
        0.0625: "20 dB",
    }, 
    ("BIDMC32", "V"): {
        0.0: "Uncompressed",
        0.015625: "40 dB",
        0.03125: "30 dB",
        0.0625: "25 dB",
        0.125: "20 dB",
    },
    ("BeijingPM10Quality", "0"): {
        0.0: "Uncompressed",
        2.0: "40 dB",
        1.0: "20 dB"
    }, 
    ("BeijingPM10Quality", "1") : {
        0.0: "Uncompressed",
        8.0: "20 dB",
        4.0: "25 dB",
        2.0: "30 dB",
        1.0: "40 dB"
    },
    ("BeijingPM10Quality", "2"): {
        0.0: "Uncompressed",
        256.0: "20 dB",
        128.0: "25 dB",
        64.0: "30 dB",
        32.0: "40 dB"
    },
    ("BeijingPM10Quality", "3"): {
        0.0: "Uncompressed",
        8.0: "20 dB",
        4.0: "30 dB",
        2.0: "40 dB"
    }
}

def plot_compression_per_dimension(
    df: pd.DataFrame,
    datasets: list[str],
    model: str = "InceptionTime",
):
    """
    Plot compression ratio vs SNR for every dimension of one or more datasets.

    Rows    : datasets
    Columns : dimensions

    Blue line   = Base-only compression ratio
    Orange line = Total compression ratio
    """

    # ------------------------------------------------------------------
    # Mapping: dataset -> dimension -> epsilon -> SNR label
    # ------------------------------------------------------------------

    snr_labels = {
        "BIDMC32": {"AVR": {
                0.00390625: "40 dB",
                0.0078125: "35 dB",
                0.015625: "30 dB",
                0.03125: "25 dB",
                0.0625: "20 dB",
            }, "II": {
                0.015625: "40 dB",
                0.03125: "30 dB",
                0.0625: "25 dB",
                0.125: "20 dB",
            }, "PLETH": {
                0.0078125: "35 dB",
                0.015625: "30 dB",
                0.03125: "25 dB",
                0.0625: "20 dB",
            }, "RESP": {
                0.0078125: "40 dB",
                0.015625: "35 dB",
                0.03125: "25 dB",
                0.0625: "20 dB",
            }, "V": {
                0.015625: "40 dB",
                0.03125: "30 dB",
                0.0625: "25 dB",
                0.125: "20 dB",
            }},
        "HouseholdPowerConsumption1": {"0": {
                4.0: "40 dB",
                8.0: "30 dB",
                16.0: "25 dB",
                32.0: "20 dB"
            }, "1": {
                0.0625: "40 dB",
                0.125: "35 dB",
                0.25: "30 dB",
                0.5: "25 dB",
                1.0: "20 dB",
            }, "4": {
                1.0: "40 dB",
                2.0: "20 dB"
            }},
        "HouseholdPowerConsumption2": {"0": {
                4.0: "40 dB",
                8.0: "30 dB",
                16.0: "25 dB",
                32.0: "20 dB"
            }, "1": {
                0.0625: "40 dB",
                0.125: "35 dB",
                0.25: "30 dB",
                0.5: "25 dB",
                1.0: "20 dB",
            }},
        "BeijingPM10Quality": {"0": {
            2.0: "40 dB",
            1.0: "20 dB"
        }, "1" : {
            8.0: "20 dB",
            4.0: "25 dB",
            2.0: "30 dB",
            1.0: "40 dB"
        }, "2": {
            256.0: "20 dB",
            128.0: "25 dB",
            64.0: "30 dB",
            32.0: "40 dB"
        }, "3": {
            8.0: "20 dB",
            4.0: "30 dB",
            2.0: "40 dB"
        }}
    }

    # ------------------------------------------------------------------

    # Build a list of all (dataset, dimension) pairs
    plots = []

    for dataset in datasets:
        dims = sorted(
            df.loc[df["dataset"] == dataset, "dimension"].unique()
        )

        for dim in dims:
            if dataset == "BeijingPM10Quality" and dim == "4":
                break
            plots.append((dataset, dim))

    nplots = len(plots)
    ncols = 3
    nrows = math.ceil(nplots / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3*ncols, 2*nrows),
        sharey=False,
    )

    axes = axes.flatten()

    for ax, (dataset, dim) in zip(axes, plots):

        subset = df[
            (df["dataset"] == dataset)
            & (df["split"] == "test")
            & (df["compressed"])
            & (df["model"] == model)
            & (df["window_length"] == 100)
            & (df["cycles"] == 50)
            & (df["batch_size"] == 16)
            & (df["preprocessing"] == "INTERPOLATION")
        ]

        data = (
            subset[subset["dimension"].astype(str) == dim]
            .sort_values("base_epsilon")
        )

        labels = [
            snr_labels[dataset][dim][eps]
            for eps in data["base_epsilon"]
        ]

        ax.plot(
            labels,
            data["compression_ratio_base_only"],
            marker="o",
            label="Base only",
        )

        ax.plot(
            labels,
            data["compression_ratio"],
            marker="s",
            label="Total",
        )

        ax.set_title(f"{dataset} dimension {dim}")

        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.show()

def plot_compression_per_dimension_one_dataset(df, dataset, model="InceptionTime"):
    snr_labels = {"AVR": {
        0.00390625: "40 dB",
        0.0078125: "35 dB",
        0.015625: "30 dB",
        0.03125: "25 dB",
        0.0625: "20 dB",
    }, "II": {
        0.015625: "40 dB",
        0.03125: "30 dB",
        0.0625: "25 dB",
        0.125: "20 dB",
    }, "PLETH": {
        0.0078125: "35 dB",
        0.015625: "30 dB",
        0.03125: "25 dB",
        0.0625: "20 dB",
    }, "RESP": {
        0.0078125: "40 dB",
        0.015625: "35 dB",
        0.03125: "25 dB",
        0.0625: "20 dB",
    }, "V": {
        0.015625: "40 dB",
        0.03125: "30 dB",
        0.0625: "25 dB",
        0.125: "20 dB",
    }}

    subset = df[
        (df["dataset"] == dataset)
        & (df["split"] == "test")
        & (df["compressed"])
        & (df["model"] == model)
        & (df["window_length"] == 50)
        & (df["batch_size"] == 16)
    ].copy()

    dimensions = sorted(subset["dimension"].unique())

    fig, axes = plt.subplots(
        1,
        len(dimensions),
        figsize=(5 * len(dimensions), 4),
        sharey=True,
    )

    if len(dimensions) == 1:
        axes = [axes]

    for ax, dim in zip(axes, dimensions):

        data = (
            subset[subset["dimension"] == dim]
            .sort_values("base_epsilon")
        )
        data["snr"] = data["base_epsilon"].map(snr_labels[dim])

        ax.plot(
            data["snr"],
            data["compression_ratio_base_only"],
            marker="o",
            label="Base only",
        )

        ax.plot(
            data["snr"],
            data["compression_ratio"],
            marker="s",
            label="Total",
        )

        ax.set_title(f"{dataset} dimension {dim}")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Compression Ratio")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
    )

    plt.tight_layout(rect=(0, 0.08, 1, 0.95))
    plt.show()

def plot_preprocessing_comparison(
    df: pd.DataFrame,
    dataset: str,
    dimension: str | int,
    model: str = "InceptionTime",
):
    """
    Compare preprocessing methods using grouped bar plots.

    x-axis: preprocessing method
    grouped by: base epsilon
    y-axis: Compressionn ratio

    Includes the uncompressed baseline (base epsilon = 0).
    """

    subset = df[ # type: ignore
        (df["dataset"] == dataset)
        & (df["dimension"].astype(str) == str(dimension))
        & (df["model"] == model)
        & (df["split"] == "test")
        & (df["compression_ratio"] > 1)
    ].copy()

    # keep only test rows
    subset = subset.sort_values(["base_epsilon", "preprocessing"]) # type: ignore

    pivot = subset.pivot_table( # type: ignore
        index="base_epsilon",
        columns="preprocessing",
        values="compression_ratio",
        aggfunc="first"
    )

    snr_labels = {
        4.0: "40 dB",
        8.0: "30/35 dB",
        16.0: "25 dB",
        32.0: "20 dB",
    }

    pivot = pivot.rename(index=snr_labels) # type: ignore
    pivot = pivot.reindex([ # type: ignore
        "20 dB",
        "25 dB",
        "30/35 dB",
        "40 dB",
    ])

    ax = pivot.plot( # type: ignore
        kind="bar",
        edgecolor="white"
    )

    for container in ax.containers: # type: ignore
        ax.bar_label( # type: ignore
            container,
            fmt="%.2f",
            fontsize=8,
            padding=3
        )

    ax.set_xlabel(None) # type: ignore
    ax.set_ylabel("Compression Ratio") # type: ignore
    ax.set_title(f"{dataset} (dim {dimension})") # type: ignore


    plt.legend( # type: ignore
        title="Preprocessing",
        loc="upper right"
    )
    plt.xticks(rotation=0, ha="right") # type: ignore

    plt.tight_layout()
    plt.show() # type: ignore

def plot_preprocessing_comparison_multimodel(
    df: pd.DataFrame,
    dataset: str,
    models: list[str],
    dimension: str = "1"
):
    fig, axes = plt.subplots( # type: ignore
        1,
        len(models),
        figsize=(5 * len(models), 5),
        sharey=True,
    )

    if len(models) == 1:
        axes = [axes]

    snr_labels = {
        0.0625: "40 dB",
        0.125: "35 dB",
        0.25: "30 dB",
        0.5: "25 dB",
        1.0: "20 dB",
    }

    for ax, model in zip(axes, models):

        subset = df[ # type: ignore
            (df["dataset"] == dataset)
            & (df["dimension"] == dimension)
            & (df["model"] == model)
            & (df["split"] == "test")
        ].copy()

        baseline_rmse = subset.loc[ # type: ignore
            subset["compression_ratio"] == 1.0,
            "rmse"
        ].iloc[0]

        compressed = subset[subset["compression_ratio"] > 1] # type: ignore

        pivot = compressed.pivot_table( # type: ignore
            index="base_epsilon",
            columns="preprocessing",
            values="rmse",
            aggfunc="first",
        )

        pivot = pivot.rename(index=snr_labels) # type: ignore
        pivot = pivot.reindex(["20 dB", "25 dB", "30 dB", "35 dB", "40 dB"]) # type: ignore

        pivot.plot( # type: ignore
            kind="bar",
            ax=ax,
            edgecolor="white",
            legend=False
        )

        ax.axhline(
            baseline_rmse,
            color="black",
            linestyle="--",
            linewidth=1.0,
        )

        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%.1f",
                fontsize=7,
                padding=2,
            )

        ax.set_title(f"{model}")
        ax.set_xlabel(None)
        ax.tick_params(axis="x", rotation=0)

    axes[0].set_ylabel("RMSE")

    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(
        Line2D(
            [0], [0],
            color="black",
            linestyle="--",
            linewidth=1.5,
        )
    )
    labels.append("Uncompressed")
    fig.legend( # type: ignore
        handles,
        labels,
        #title="Preprocessing",
        loc="lower center",
        ncol=len(labels),
        frameon=False,
    )

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    plt.show() # type: ignore

def plot_preprocessing_comparison_multidim(
    df: pd.DataFrame,
    dataset: str,
    dimensions: list[str],
    model: str = "InceptionTime",
):
    fig, axes = plt.subplots( # type: ignore
        1,
        len(dimensions),
        figsize=(5 * len(dimensions), 5),
        sharey=True,
    )

    if len(dimensions) == 1:
        axes = [axes]

    snr_labels = {"0": {
        4.0: "40 dB",
        8.0: "35 dB",
        16.0: "30 dB",
        32.0: "20 dB"
    }, "1": {
        0.0625: "40 dB",
        0.125: "35 dB",
        0.25: "30 dB",
        0.5: "25 dB",
        1.0: "20 dB",
    }, "4": {
        1.0: "40 dB",
        2.0: "20 dB"
    }}
    reindex = {
        "0": ["20 dB", "30 dB", "35 dB", "40 dB"],
        "1": ["20 dB", "25 dB", "30 dB", "35 dB", "40 dB"],
        "4": ["20 dB", "40 dB"]
    }

    for ax, dimension in zip(axes, dimensions):
        subset = df[ # type: ignore
            (df["dataset"] == dataset)
            & (df["dimension"] == dimension)
            & (df["model"] == model)
            & (df["split"] == "test")
        ].copy()

        compressed = subset[subset["compression_ratio"] > 1] # type: ignore

        pivot = compressed.pivot_table( # type: ignore
            index="base_epsilon",
            columns="preprocessing",
            values="compression_ratio",
            aggfunc="first",
        )

        pivot = pivot.rename(index=snr_labels[dimension]) # type: ignore
        pivot = pivot.reindex(reindex[dimension]) # type: ignore

        pivot.plot( # type: ignore
            kind="bar",
            ax=ax,
            edgecolor="white",
            legend=False
        )

        ax.set_title(f"{dataset} dimension {dimension}")
        ax.set_xlabel(None)
        ax.tick_params(axis="x", rotation=0)

    axes[0].set_ylabel("Compression Ratio")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend( # type: ignore
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    plt.show() # type: ignore

def plot_compression_ratio_per_batchsize(
    df,
    dataset,
    dimension,
    model,
):

    subset = filter_results(df, dataset, dimension, model, 40, None, None)
    subset = subset[subset["compressed"]]

    # Fixed ordering of the bins
    eps_order = sorted(subset["base_epsilon"].unique())

    # Marker cycle
    markers = ["o", "^", "s", "D", "v", "P", "X", "*", "<", ">"]
    plt.figure(figsize=(7, 5))

    for marker, (wl, group) in zip(markers, subset.groupby("batch_size")):

        group = group.sort_values("base_epsilon")

        plt.plot(
            group["base_epsilon"].astype(str),   # categorical x-axis
            group["rmse"],
            marker=marker,
            label=f"{wl}",
        )

    snr_map = {
        0.00390625: "40 dB",
        0.0078125: "35 dB",
        0.015625: "30 dB",
        0.03125: "25 dB",
        0.0625: "20 dB",
    }

    plt.xticks(
        range(len(eps_order)),
        [snr_map[e] for e in eps_order]
    )

    plt.xlabel("SNR")
    plt.ylabel("RMSE")
    plt.title(f"{dataset} - dimension {dimension} - {model}")
    plt.grid(alpha=0.3)
    plt.legend(title="Batch size")
    plt.tight_layout()
    plt.show()

def filter_results(
    df,
    dataset: str,
    dimension: str,
    model: str,
    window_length: int | None =None,
    cycles: int | None =None,
    batch_size: int | None =16,
):
    subset = df[
        (df["dataset"] == dataset)
        & (df["dimension"].astype(str) == str(dimension))
        & (df["model"] == model)
        & (df["split"] == "test")
    ].copy()

    if window_length is not None:
        subset = subset[subset["window_length"] == window_length]

    if cycles is not None:
        subset = subset[subset["cycles"] == cycles]

    if batch_size is not None:
        subset = subset[subset["batch_size"] == batch_size]

    return subset.sort_values("base_epsilon")

def plot_rmse(
    df,
    dataset,
    dimension,
    model,
):

    subset = filter_results(df, dataset, dimension, model)

    plt.figure(figsize=(7,5))

    for wl, group in subset.groupby("window_length"):

        plt.plot(
            group["base_epsilon"],
            group["rmse"],
            marker="o",
            label=f"WL={wl}"
        )

    plt.xlabel("Base ε")
    plt.ylabel("RMSE")
    plt.title(dataset)
    plt.grid(alpha=.3)
    plt.legend(title="Window length")
    plt.tight_layout()
    plt.show()

def plot_compression_ratio_per_wl(
    df,
    dataset,
    dimension,
    model,
):

    subset = filter_results(df, dataset, dimension, model)
    #subset = subset[subset["compressed"]]

    # Fixed ordering of the bins
    eps_order = sorted(subset["base_epsilon"].unique())

    # Marker cycle
    markers = ["o", "^", "s", "D", "v", "P", "X", "*", "<", ">"]
    plt.figure(figsize=(7, 5))

    for marker, (wl, group) in zip(markers, subset.groupby("window_length")):

        group = group.sort_values("base_epsilon")

        plt.plot(
            group["base_epsilon"].astype(str),   # categorical x-axis
            group["rmse"],
            marker=marker,
            label=f"{wl}",
        )

    snr_map = {
        0.0: "Uncompressed",
        0.00390625: "40 dB",
        0.0078125: "35 dB",
        0.015625: "30 dB",
        0.03125: "25 dB",
        0.0625: "20 dB",
    }

    plt.xticks(
        range(len(eps_order)),
        [snr_map[e] for e in eps_order]
    )

    plt.xlabel("SNR")
    plt.ylabel("RMSE")
    plt.title(f"{dataset} - dimension {dimension} - {model}")
    plt.grid(alpha=0.3)
    plt.legend(title="Window length")
    plt.tight_layout()
    plt.show()

def plot_inference_time(
    df,
    experiments,
    model="InceptionTime",
    cycles=50,
    batch_size=16,
):

    fig, ax = plt.subplots(figsize=(9, 5))

    width = 0.8 / len(experiments)

    all_labels = None

    for i, exp in enumerate(experiments):

        subset = (
            df[
                (df["dataset"] == exp["dataset"])
                & (df["dimension"].astype(str) == str(exp["dimension"]))
                & (df["model"] == model)
                & (df["window_length"] == exp["window_length"])
                & (df["cycles"] == cycles)
                & (df["batch_size"] == batch_size)
                & (df["preprocessing"] == exp["preprocessing"])
                & (df["split"] == "test")
            ]
            .sort_values("base_epsilon")
        )

        # convert epsilons -> SNR labels
        mapping = snr_labels[(exp["dataset"], str(exp["dimension"]))]

        subset["snr"] = subset["base_epsilon"].map(mapping)

        if all_labels is None:
            all_labels = subset["snr"].tolist()
            x = np.arange(len(all_labels))

        label = exp.get(
            "label",
            f"{exp['dataset']} dim {exp['dimension']}"
        )

        ax.bar(
            x + i * width,
            subset["inference_time"],
            width=width,
            label=label,
            edgecolor="white"
        )
        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%.1f",
                fontsize=7,
                padding=2,
            )

    ax.set_xticks(x + width * (len(experiments) - 1) / 2)
    ax.set_xticklabels(all_labels)

    ax.set_ylabel("Inference Time (ms)")
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
    )

    fig.suptitle(model)

    plt.tight_layout(rect=(0, 0.15, 1, 1))
    plt.show()

def filter_results_2(
    df,
    *,
    dataset=None,
    dimension=None,
    model=None,
    preprocessing=None,
    window_length=None,
    cycles=None,
    batch_size=None,
):
    subset = df.copy()

    if dataset is not None:
        subset = subset[subset["dataset"] == dataset]

    if dimension is not None:
        subset = subset[subset["dimension"].astype(str) == str(dimension)]

    if model is not None:
        subset = subset[subset["model"] == model]

    if preprocessing is not None:
        subset = subset[subset["preprocessing"] == preprocessing]

    if window_length is not None:
        subset = subset[subset["window_length"] == window_length]

    if cycles is not None:
        subset = subset[subset["cycles"] == cycles]

    if batch_size is not None:
        subset = subset[subset["batch_size"] == batch_size]
    return subset

def plot_rmse_improvement(
    df: pd.DataFrame,
    model: str,
    preprocessing: str,
    window_length: int,
    cycles: int,
    batch_size: int,
    snr_mapping: dict[tuple[str, str], dict[float, str]],
):
    # Extract unique datasets and their dimensions from snr_mapping
    datasets = sorted({key[0] for key in snr_mapping.keys()})
    dimensions_per_dataset = {}
    for dataset in datasets:
        dimensions_per_dataset[dataset] = sorted({key[1] for key in snr_mapping.keys() if key[0] == dataset})
    plt.figure(figsize=(12, 6))

    # Collect all unique SNR labels across all datasets and dimensions
    all_snr_labels = set()
    for (dataset, dim), mapping in snr_mapping.items():
        all_snr_labels.update(mapping.values())
    all_snr_labels = sorted(all_snr_labels)
    n_labels = len(all_snr_labels)

    # For each dataset, plot its dimensions' bars
    for dataset_idx, dataset in enumerate(datasets):
        n_dim = len(dimensions_per_dataset[dataset])
        width = 0.8 / (n_dim * len(datasets))  # Adjust width for all datasets and dimensions

        for dim_idx, dim in enumerate(dimensions_per_dataset[dataset]):
            wl = 50 if dataset == "BIDMC32" else 100
            prepro = None if dataset == "BIDMC32" else preprocessing
            subset = filter_results_2(
                df,
                dataset=dataset,
                dimension=dim,
                model=model,
                preprocessing=prepro,
                window_length=wl,
                cycles=cycles,
                batch_size=batch_size,
            )

            baseline = subset[
                (subset["split"] == "test")
                & (~subset["compressed"])
            ]["rmse"].iloc[0]

            compressed = subset[
                (subset["split"] == "test")
                & (subset["compressed"])
            ].sort_values("base_epsilon")

            delta = baseline - compressed["rmse"]

            # Map base_epsilon to SNR label using snr_mapping
            labels = [
                snr_mapping[(dataset, str(dim))][e]
                for e in compressed["base_epsilon"]
            ]

            # Only plot for labels that exist in all_snr_labels
            delta_filtered = []
            labels_filtered = []
            for label, d in zip(labels, delta):
                if label in all_snr_labels:
                    delta_filtered.append(d)
                    labels_filtered.append(label)
            x_pos = [all_snr_labels.index(label) for label in labels_filtered]
            # Offset for dataset and dimension
            x_pos = np.array(x_pos) + (
                (dataset_idx * n_dim + dim_idx) - (n_dim * len(datasets)) / 2 + 0.5
            ) * width

            plt.bar(
                x_pos,
                delta_filtered,
                width=width,
                label=f"{dataset} Dim {dim}",
                edgecolor="white",
            )

    plt.xticks(np.arange(n_labels), all_snr_labels, rotation=45, ha='right')
    plt.ylabel("RMSE improvement")
    plt.xlabel("SNR")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='-')

    plt.tight_layout()
    plt.show()

def plot_tradeoff(
    df,
    dataset,
    dimensions,
    model,
    preprocessing,
    window_length,
    cycles,
    batch_size,
    snr_mapping,
):
    plt.figure(figsize=(7, 5))

    for dim in dimensions:

        subset = filter_results_2(
            df,
            dataset=dataset,
            dimension=dim,
            model=model,
            preprocessing=preprocessing,
            window_length=window_length,
            cycles=cycles,
            batch_size=batch_size,
        )

        baseline = subset[
            (subset["split"] == "test")
            & (~subset["compressed"])
        ]["rmse"].iloc[0]

        compressed = (
            subset[
                (subset["split"] == "test")
                & (subset["compressed"])
            ]
            .sort_values("base_epsilon")
        )

        improvement = baseline - compressed["rmse"]

        # Scatter for this dimension
        plt.scatter(
            compressed["compression_ratio_base_only"],
            improvement,
            s=70
        )

        # Label every point
        for _, row in compressed.iterrows():

            snr = snr_mapping[(dataset, str(dim))][row["base_epsilon"]]

            plt.annotate(
                f"dim {dim} ({snr})",
                (
                    row["compression_ratio_base_only"],
                    baseline - row["rmse"],
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
            )

    plt.axhline(0, color="black")

    plt.xlabel("Compression Ratio Base Only")
    plt.ylabel("RMSE Improvement")
    plt.title(dataset)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_improvement_heatmap(
    df,
    experiments,
    model,
    preprocessing,
    window_length,
    cycles,
    batch_size,
    snr_mapping,
):
    """
    experiments : list of dicts, e.g.

    [
        {
            "dataset": "HouseholdPowerConsumption1",
            "dimensions": ["0", "1", "4"],
        },
        {
            "dataset": "HouseholdPowerConsumption2",
            "dimensions": ["0", "1"],
        },
        {
            "dataset": "BIDMC32",
            "dimensions": ["AVR", "II", "PLETH"],
        },
    ]
    """

    rows = []

    for exp in experiments:

        dataset = exp["dataset"]

        preprop = None if dataset == "BIDMC32" else preprocessing
        wl = 50 if dataset == "BIDMC32" else window_length
        subset = filter_results_2(
            df,
            dataset=dataset,
            model=model,
            preprocessing=preprop,
            window_length=wl,
            cycles=cycles,
            batch_size=batch_size,
        )

        for dim in exp["dimensions"]:

            data = subset[
                subset["dimension"].astype(str) == str(dim)
            ]

            baseline = data[
                (data["split"] == "test")
                & (~data["compressed"])
            ]["rmse"].iloc[0]

            compressed = data[
                (data["split"] == "test")
                & (data["compressed"])
            ].copy()

            compressed["SNR"] = compressed["base_epsilon"].map(
                snr_mapping[(dataset, str(dim))]
            )

            compressed["improvement"] = (
                baseline - compressed["rmse"]
            )

            rows.append(
                compressed.set_index("SNR")["improvement"]
                .rename(f"{dataset} dim {dim}")
            )

    heat = pd.DataFrame(rows)

    plt.figure(figsize=(8, max(4, len(rows) * 0.5)))

    norm = TwoSlopeNorm(vcenter=0)
    plt.imshow(
        heat,
        cmap=plt.get_cmap("RdYlGn", 9),   # 9 distinct color levels
        norm=norm,
        aspect="auto",
    )

    plt.xticks(
        range(len(heat.columns)),
        heat.columns,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(heat.index)),
        heat.index,
    )

    plt.colorbar(label="RMSE improvement")

    plt.tight_layout()
    plt.show()

def plot_best_safe_compression(
    df,
    experiments,
    model,
    preprocessing,
    window_length,
    cycles,
    batch_size,
):
    """
    experiments example:

    [
        {
            "dataset": "HouseholdPowerConsumption1",
            "dimensions": ["0", "1", "4"],
        },
        {
            "dataset": "HouseholdPowerConsumption2",
            "dimensions": ["0", "1"],
        },
        {
            "dataset": "BIDMC32",
            "dimensions": ["AVR", "II", "PLETH"],
        },
    ]
    """

    labels = []
    best_ratios = []

    for exp in experiments:

        dataset = exp["dataset"]

        preprop = None if dataset == "BIDMC32" else preprocessing
        wl = 50 if dataset == "BIDMC32" else window_length
        subset = filter_results_2(
            df,
            dataset=dataset,
            model=model,
            preprocessing=preprop,
            window_length=wl,
            cycles=cycles,
            batch_size=batch_size,
        )

        for dim in exp["dimensions"]:

            data = subset[
                subset["dimension"].astype(str) == str(dim)
            ]

            if data.empty:
                continue

            baseline = data[
                (data["split"] == "test")
                & (~data["compressed"])
            ]["rmse"].iloc[0]

            candidates = data[
                (data["split"] == "test")
                & (data["compressed"])
                & (data["rmse"] <= baseline)
            ]

            if len(candidates):
                best_ratio = candidates["compression_ratio"].max()

                # Only plot if compression is actually beneficial
                if best_ratio > 1:
                    labels.append(f"{dataset}\n{dim}")
                    best_ratios.append(best_ratio)

    plt.figure(figsize=(max(8, len(labels) * 0.7), 5))

    plt.bar(
        labels,
        best_ratios,
        edgecolor="white",
    )

    plt.ylabel("Maximum Safe Compression Ratio")
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("data/all_results.csv")
    # plot_tradeoff(
    #     df=df,
    #     dataset="BeijingPM10Quality",
    #     dimensions=["2", "3"],
    #     model="InceptionTime",
    #     preprocessing="INTERPOLATION",
    #     window_length=100,
    #     cycles=50,
    #     batch_size=16,
    #     snr_mapping=snr_labels
    # ) 
    # plot_improvement_heatmap(
    #     df,
    #     experiments=[
    #         {
    #             "dataset": "HouseholdPowerConsumption1",
    #             "dimensions": ["0", "1", "4"],
    #         },
    #         {
    #             "dataset": "HouseholdPowerConsumption2",
    #             "dimensions": ["0", "1"],
    #         },
    #         {
    #             "dataset": "BIDMC32",
    #             "dimensions": ["AVR", "II", "PLETH", "RESP", "V"],
    #         },
    #     ],
    #     model="InceptionTime",
    #     preprocessing="INTERPOLATION",
    #     window_length=100,
    #     cycles=50,
    #     batch_size=16,
    #     snr_mapping=snr_labels,
    # )
    plot_best_safe_compression(
    df,
    experiments=[
        {
            "dataset": "HouseholdPowerConsumption1",
            "dimensions": ["0", "1", "4"],
        },
        {
            "dataset": "HouseholdPowerConsumption2",
            "dimensions": ["0", "1"],
        },
        {
            "dataset": "BIDMC32",
            "dimensions": ["AVR", "II", "PLETH", "RESP", "V"],
        },
    ],
    model="InceptionTime",
    preprocessing="INTERPOLATION",
    window_length=100,
    cycles=50,
    batch_size=16,
)









    # plot_rmse_improvement(
    #     df=df,
    #     model="InceptionTime",
    #     preprocessing="INTERPOLATION",
    #     window_length=100,
    #     cycles=50,
    #     batch_size=16,
    #     snr_mapping=snr_labels
    # ) TODO it looks so shit
    # plot_preprocessing_comparison_multidim(
    #     df,
    #     dataset="HouseholdPowerConsumption1",
    #     dimensions=["0", "1", "4"]
    # )
    # plot_preprocessing_comparison_multimodel(
    #     df,
    #     dataset="HouseholdPowerConsumption1",
    #     models=["InceptionTime", "MLP", "FCN"]
    # )
    # plot_compression_ratio_per_wl(
    #     df,
    #     dataset="BIDMC32",
    #     dimension="AVR",
    #     model="InceptionTime"
    # )
    # plot_compression_per_dimension_one_dataset(df, "BIDMC32")
    # plot_compression_ratio_per_batchsize(
    #     df,
    #     dataset="BIDMC32",
    #     dimension="AVR",
    #     model="InceptionTime"
    # )
    # plot_compression_per_dimension(
    #     df,
    #     datasets=[
    #         "HouseholdPowerConsumption1",
    #         "HouseholdPowerConsumption2",
    #         "BeijingPM10Quality"
    #     ],
    #     model="InceptionTime",
    # )
    # experiments = [
    #     {
    #         "dataset": "HouseholdPowerConsumption1",
    #         "dimension": "1",
    #         "window_length": 100,
    #         "preprocessing": "INTERPOLATION",
    #         "label": "HPC1 dim1 WL100 INTERPOLATION"
    #     },
    #     {
    #         "dataset": "HouseholdPowerConsumption1",
    #         "dimension": "1",
    #         "window_length": 150,
    #         "preprocessing": "INTERPOLATION",
    #         "label": "HPC1 dim1 WL150 INTERPOLATION"
    #     },
    #     {
    #         "dataset": "HouseholdPowerConsumption1",
    #         "dimension": "1",
    #         "window_length": 200,
    #         "preprocessing": "INTERPOLATION",
    #         "label": "HPC1 dim1 WL200 INTERPOLATION"
    #     },
    #     {
    #         "dataset": "HouseholdPowerConsumption1",
    #         "dimension": "1",
    #         "window_length": 100,
    #         "preprocessing": "KNN",
    #         "label": "HPC1 dim1 WL100 KNN"
    #     },
    #     {
    #         "dataset": "HouseholdPowerConsumption1",
    #         "dimension": "1",
    #         "window_length": 100,
    #         "preprocessing": "MEAN",
    #         "label": "HPC1 dim1 WL100 MEAN"
    #     },
        # {
        #     "dataset": "HouseholdPowerConsumption2",
        #     "dimension": "1",
        #     "window_length": 100,
        #     "preprocessing": "INTERPOLATION",
        #     "label": "HPC2 dim1 WL100 INTERPOLATION"
        # }
    # ]
    # plot_inference_time(
    #     df,
    #     experiments
    # )