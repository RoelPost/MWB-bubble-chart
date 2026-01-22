"""Standalone script to plot machine-level relative NOx differences by main group."""

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

MIN_VALID_YEAR = 2000
EXCLUDED_MACHINES = {2035, 2039, 2083, 2225}
NOX_TOTAL_COLUMN = "NOxTotal"
NOX_AUB_COLUMN = "NOxMAF_AUB_NoxEmission"
MOTORBELASTING_COLUMN = "NOxMAF_motorbelasting"
BELASTINGTYPE_COLUMN = "Belastingtype"
CATEGORY_A_PREFIX = "A"
DATA_PATH = Path(
    "/Users/roelpost/DeveloperTools/MWB bubble chart/Exploratory/data/noxdagdata 2025-12-01T160303.csv"
)


def _mode_or_first_nonnull(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return None
    mode = non_null.mode()
    return mode.iloc[0] if not mode.empty else non_null.iloc[0]


def filter_by_bouwjaar(df: pd.DataFrame, min_year: int) -> pd.DataFrame:
    """Remove rows with ConstructionYear earlier than min_year."""
    if "ConstructionYear" not in df.columns:
        return df.copy()
    mask = df["ConstructionYear"].isna() | (df["ConstructionYear"] >= min_year)
    return df.loc[mask].copy()


def compute_machine_relative_nox(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate NOx totals and compute relative difference per machine."""
    required_cols = {"MachineId", NOX_TOTAL_COLUMN, NOX_AUB_COLUMN}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns required for machine aggregation: {missing}")

    agg_map = {
        "NOxTotal_sum": (NOX_TOTAL_COLUMN, "sum"),
        "NOxAUB_sum": (NOX_AUB_COLUMN, "sum"),
        "records": ("MachineId", "size"),
        "MainGroupLabel": ("MainGroupLabel", _mode_or_first_nonnull),
    }
    if MOTORBELASTING_COLUMN in df.columns:
        agg_map["Motorbelasting_mean"] = (MOTORBELASTING_COLUMN, "mean")

    grouped = df.groupby("MachineId").agg(**agg_map).reset_index()

    valid = grouped[
        grouped["NOxAUB_sum"].notna() & (grouped["NOxAUB_sum"] != 0)
    ].copy()
    valid["relative_diff"] = (valid["NOxTotal_sum"] - valid["NOxAUB_sum"]) / valid[
        "NOxAUB_sum"
    ]
    return valid


def filter_category_a(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows whose Belastingtype starts with CATEGORY_A_PREFIX (case-insensitive)."""
    if BELASTINGTYPE_COLUMN not in df.columns:
        raise ValueError(f"Column '{BELASTINGTYPE_COLUMN}' not found; cannot filter category A.")
    mask = df[BELASTINGTYPE_COLUMN].astype(str).str.startswith(
        CATEGORY_A_PREFIX, na=False, case=False
    )
    return df.loc[mask].copy()


def plot_machine_relative_by_main_group(df: pd.DataFrame):
    """Bar chart with MainGroupLabel shown per bar; each machine is a separate bar (single color, no legend)."""
    valid = compute_machine_relative_nox(df)
    if valid.empty:
        print("No machines have both NOxTotal and NOxMAF_AUB_NoxEmission data.")
        return

    valid["MainGroupLabel"] = valid["MainGroupLabel"].fillna("Unknown")
    valid["MachineId_str"] = valid["MachineId"].astype(str)

    # Order bars by relative difference (high to low) for readability.
    valid = valid.sort_values("relative_diff", ascending=False)

    # Build a combined label so each bar is unique on the axis.
    y_labels = valid.apply(
        lambda row: f"{row['MachineId_str']} ({row['MainGroupLabel']})", axis=1
    )

    base_color = "#4c78a8"

    fig = go.Figure(
        go.Bar(
            x=valid["relative_diff"],
            y=y_labels,
            orientation="h",
            text=valid["MachineId_str"],
            textposition="outside",
            marker_color=base_color,
            showlegend=False,
            hovertemplate=(
                "MachineId: %{text}<br>"
                "Machine (group): %{y}<br>"
                "Relative diff: %{x:.2f}<br>"
                "NOxTotal sum: %{customdata[0]:.2f}<br>"
                "NOxAUB sum: %{customdata[1]:.2f}<extra></extra>"
            ),
            customdata=valid[["NOxTotal_sum", "NOxAUB_sum"]],
        )
    )

    fig.update_layout(
        title="Machine-level relative difference (NOxTotal vs NOxMAF AUB) per machine type",
        xaxis_title="Relative difference (NOxTotal - AUB) / AUB",
        yaxis_title="Machine (MachineId with main group)",
        bargap=0.2,
        plot_bgcolor="white",
        yaxis=dict(categoryorder="array", categoryarray=y_labels),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    fig.show()


def plot_machine_relative_simple(df: pd.DataFrame):
    """Machine-level relative difference (NOxTotal vs NOxMAF AUB) without motorbelasting coloring."""
    valid = compute_machine_relative_nox(df)
    if valid.empty:
        print("No machines have both NOxTotal and NOxMAF_AUB_NoxEmission data.")
        return

    valid["MachineId_str"] = valid["MachineId"].astype(str)
    valid = valid.sort_values("relative_diff", ascending=False)

    fig = go.Figure(
        go.Bar(
            x=valid["MachineId_str"],
            y=valid["relative_diff"],
            marker_color="#4c78a8",
            text=valid["relative_diff"].round(2),
            textposition="outside",
            hovertemplate=(
                "MachineId: %{x}<br>"
                "Relative diff: %{y:.2f}<br>"
                "NOxTotal sum: %{customdata[0]:.2f}<br>"
                "NOxAUB sum: %{customdata[1]:.2f}<extra></extra>"
            ),
            customdata=valid[["NOxTotal_sum", "NOxAUB_sum"]],
            showlegend=False,
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Machine-level relative difference (NOxTotal vs NOxMAF AUB)",
        xaxis_title="MachineId",
        yaxis_title="Relative difference (NOxTotal - AUB) / AUB",
        plot_bgcolor="white",
    )
    fig.show()


def plot_category_a_relative_by_main_group(df: pd.DataFrame):
    """Helper to plot the category A subset."""
    category_a_df = filter_category_a(df)
    if category_a_df.empty:
        print("No category A machines found to plot.")
        return
    print(f"Plotting category A machines: {category_a_df['MachineId'].nunique()} machines.")
    plot_machine_relative_by_main_group(category_a_df)


def load_data() -> Optional[pd.DataFrame]:
    data_path = DATA_PATH
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return None
    df = pd.read_csv(data_path)
    df = filter_by_bouwjaar(df, MIN_VALID_YEAR)
    if "MachineId" in df.columns:
        df = df[~df["MachineId"].isin(EXCLUDED_MACHINES)].copy()
    return df


if __name__ == "__main__":
    df_nox = load_data()
    if df_nox is not None:
        plot_machine_relative_simple(df_nox)
        plot_machine_relative_by_main_group(df_nox)
        plot_category_a_relative_by_main_group(df_nox)
