# %% [markdown]
# # Exploratory Analysis - NOx Data

# %%
import pandas as pd
from pathlib import Path
from typing import Optional

# %%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
from statsmodels.nonparametric.smoothers_lowess import lowess

FOCUS_COLUMNS = {
    "Stage": ["Stage", "EngineClassificationLabel"],
    "Machinegroep": ["Machinegroep"],
}
NOXMAF_COLUMNS = {
    "NOxPerLiter": "NOxMAF_NOxPerLiter",
    "Motorbelasting": "NOxMAF_motorbelasting",
}
NOXMAF_ABSOLUTE_COLUMN = "NOxTotal"
PLOTLY_STAGE_FILTER = "Stage-V+D"
MIN_VALID_YEAR = 2000
EXCLUDED_MACHINES = {2035, 2039, 2083, 2225}
GRAMS_PER_KILOGRAM = 1000.0
NOX_G_PER_L_THRESHOLD_LOW = 0.4
NOX_G_PER_L_THRESHOLD_MEDIUM = 6.0
HOURS_CANDIDATE_COLUMNS = [
    "totalHoursPerDay",
    "CANBUS_validated_duration",
    "FF_validated_duration",
    "duration_from_rows",
]
FUEL_CANDIDATE_COLUMNS = [
    "totalFuelPerDay",
    "CANBUS_validated_fuel",
    "FF_validated_fuel",
    "FuelMassFlow",
]
HOURS_MIN_THRESHOLD = 1.0
FUEL_MIN_THRESHOLD = 1.0
MOTORBELASTING_INVALID_FRACTION_THRESHOLD = 0.1  # Exclude if >10% of readings are invalid
EXHAUST_COLUMNS = ["NOxTotal", "NOxMAF_NOxPerLiter", "NOxMAF_AUB_NoxEmission"]


def find_machines_with_invalid_sensor_data(
    df: pd.DataFrame,
    motorbelasting_col: str = "NOxMAF_motorbelasting",
    invalid_fraction_threshold: float = MOTORBELASTING_INVALID_FRACTION_THRESHOLD,
) -> set:
    """Identify machines that regularly have motorbelasting > 1 or negative exhaust values.

    Returns a set of MachineIds that should be excluded.
    """
    if "MachineId" not in df.columns:
        return set()

    excluded_machines = set()

    # Check for machines with frequent motorbelasting > 1 (100%)
    if motorbelasting_col in df.columns:
        motorbelasting_stats = df.groupby("MachineId").apply(
            lambda g: (g[motorbelasting_col] > 1).sum() / len(g)
            if len(g) > 0 else 0
        )
        high_motorbelasting_machines = set(
            motorbelasting_stats[motorbelasting_stats > invalid_fraction_threshold].index
        )
        if high_motorbelasting_machines:
            print(
                f"Found {len(high_motorbelasting_machines)} machines with >{invalid_fraction_threshold*100:.0f}% "
                f"of readings having motorbelasting > 100%: {sorted(high_motorbelasting_machines)}"
            )
        excluded_machines.update(high_motorbelasting_machines)

    # Check for machines with frequent negative exhaust values
    for exhaust_col in EXHAUST_COLUMNS:
        if exhaust_col not in df.columns:
            continue
        negative_stats = df.groupby("MachineId").apply(
            lambda g: (g[exhaust_col] < 0).sum() / len(g)
            if len(g) > 0 else 0
        )
        negative_exhaust_machines = set(
            negative_stats[negative_stats > invalid_fraction_threshold].index
        )
        if negative_exhaust_machines:
            print(
                f"Found {len(negative_exhaust_machines)} machines with >{invalid_fraction_threshold*100:.0f}% "
                f"of readings having negative {exhaust_col}: {sorted(negative_exhaust_machines)}"
            )
        excluded_machines.update(negative_exhaust_machines)

    return excluded_machines


def build_focus_dataframe(df: pd.DataFrame, focus_map: dict) -> pd.DataFrame:
    """Return dataframe with requested focus columns, using fallbacks if needed."""
    focus_df = pd.DataFrame(index=df.index)
    missing_targets = []

    for target, candidates in focus_map.items():
        source_col = next((col for col in candidates if col in df.columns), None)
        if source_col is None:
            missing_targets.append(target)
            continue
        focus_df[target] = df[source_col]

    if missing_targets:
        raise ValueError(f"Missing expected columns: {missing_targets}")

    return focus_df


def build_stage_groep_label(df_focus: pd.DataFrame) -> pd.Series:
    """Create Stage+Groep label string."""
    stage = df_focus["Stage"].fillna("Stage-Unknown").astype(str)
    groep = df_focus["Machinegroep"].fillna("Groep-Unknown").astype(str)
    return stage + "+" + groep


def filter_by_bouwjaar(df: pd.DataFrame, min_year: int) -> pd.DataFrame:
    """Remove rows with ConstructionYear earlier than min_year."""
    if "ConstructionYear" not in df.columns:
        return df.copy()
    mask = df["ConstructionYear"].isna() | (df["ConstructionYear"] >= min_year)
    dropped = len(df) - mask.sum()
    if dropped > 0:
        print(f"Filtered out {dropped} rows with ConstructionYear before {min_year}.")
    return df.loc[mask].copy()


def prepare_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric+dummified categorical features suitable for correlation."""
    numeric_cols = df.select_dtypes(include=[np.number])
    categorical_cols = df.select_dtypes(exclude=[np.number])

    if categorical_cols.shape[1] > 0:
        dummy_df = pd.get_dummies(
            categorical_cols,
            prefix=categorical_cols.columns,
            dummy_na=True,
            drop_first=False,
        )
    else:
        dummy_df = pd.DataFrame(index=df.index)

    feature_matrix = pd.concat([numeric_cols, dummy_df], axis=1)

    # Drop columns with constant values which break correlations
    keep_mask = feature_matrix.nunique(dropna=False) > 1
    return feature_matrix.loc[:, keep_mask]


def build_hours_series(df: pd.DataFrame) -> pd.Series:
    """Create consolidated hours values using best available duration column."""
    hours = pd.Series(np.nan, index=df.index, dtype=float)
    for col in HOURS_CANDIDATE_COLUMNS:
        if col not in df.columns:
            continue
        candidate = pd.to_numeric(df[col], errors="coerce")
        hours = hours.combine_first(candidate)
    return hours


def build_fuel_series(df: pd.DataFrame) -> pd.Series:
    """Create consolidated fuel values using best available fuel column."""
    fuel = pd.Series(np.nan, index=df.index, dtype=float)
    for col in FUEL_CANDIDATE_COLUMNS:
        if col not in df.columns:
            continue
        candidate = pd.to_numeric(df[col], errors="coerce")
        fuel = fuel.combine_first(candidate)
    return fuel


def _mode_or_first_nonnull(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return None
    mode = non_null.mode()
    return mode.iloc[0] if not mode.empty else non_null.iloc[0]


def plot_correlation_heatmap(corr: pd.DataFrame, title: str):
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, square=True)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_noxmaf_scatter(
    df: pd.DataFrame, stage_groep: pd.Series, stage_filter: Optional[str] = None
):
    """Scatter NOxMAF NOxPerLiter vs motorbelasting colored by Stage+Groep."""
    missing_cols = [name for name in NOXMAF_COLUMNS.values() if name not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing NOxMAF columns: {missing_cols}")

    plot_df = pd.DataFrame(
        {
            "Motorbelasting": df[NOXMAF_COLUMNS["Motorbelasting"]],
            "NOxPerLiter": df[NOXMAF_COLUMNS["NOxPerLiter"]],
            "Stage+Groep": stage_groep,
            "MachineId": df.get("MachineId"),
            "MainGroupLabel": df.get("MainGroupLabel", "Unknown"),
        }
    ).dropna(subset=["Motorbelasting", "NOxPerLiter"])
    if stage_filter:
        plot_df = plot_df[plot_df["Stage+Groep"] == stage_filter]
        if plot_df.empty:
            print(f"No NOxMAF rows available for Stage+Groep = {stage_filter}.")
            return

    plt.figure(figsize=(12, 6))
    title_suffix = f" — Stage+Groep {stage_filter}" if stage_filter else ""
    ax = sns.scatterplot(
        data=plot_df,
        x="Motorbelasting",
        y="NOxPerLiter",
        hue="Stage+Groep",
        palette="husl",
        alpha=0.6,
        s=50,
    )
    plt.title(f"NOxPerLiter (kg/L) vs Motorbelasting (%) {title_suffix}")
    plt.xlabel("Motorbelasting (%)")
    plt.ylabel("NOxPerLiter (kg/L)")
    plt.axhline(0.0016, color="gray", linestyle="--", linewidth=1)
    scatter_legend = plt.legend(
        title="Stage+Groep", bbox_to_anchor=(1.02, 1), loc="upper left"
    )
    plt.gca().add_artist(scatter_legend)

    line_handles = []
    for main_group, subset in plot_df.groupby("MainGroupLabel"):
        if subset.shape[0] < 5:
            continue
        sorted_subset = subset.sort_values("Motorbelasting")
        smoothed = lowess(
            sorted_subset["NOxPerLiter"],
            sorted_subset["Motorbelasting"],
            frac=0.3,
            return_sorted=True,
        )
        (line,) = plt.plot(
            smoothed[:, 0],
            smoothed[:, 1],
            linewidth=2,
            label=f"{main_group} trend",
        )
        line_handles.append(line)

    if line_handles:
        plt.legend(
            handles=line_handles,
            title="MainGroupLabel trend",
            loc="lower right",
        )

    plt.tight_layout()
    plt.show()

    if "MachineId" in plot_df:
        counts = plot_df.groupby("Stage+Groep")
        counts = counts["MachineId"].nunique().sort_values(ascending=False)
        print("\nStage+Groep machine counts (based on plotted points):")
        print(counts)


def plot_noxmaf_per_machine_by_maingroup(df: pd.DataFrame):
    """Scatter NOxPerLiter vs motorbelasting aggregated to one point per machine."""
    required_cols = list(NOXMAF_COLUMNS.values()) + ["MachineId"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns for per-machine plot: {missing_cols}")

    plot_df = (
        df.groupby(["MachineId", "MainGroupLabel"])
        .agg(
            Motorbelasting=(NOXMAF_COLUMNS["Motorbelasting"], "mean"),
            NOxPerLiter=(NOXMAF_COLUMNS["NOxPerLiter"], "mean"),
            records=("MachineId", "size"),
        )
        .reset_index()
    )
    plot_df["MainGroupLabel"] = plot_df["MainGroupLabel"].fillna("Unknown")
    plot_df = plot_df.dropna(subset=["Motorbelasting", "NOxPerLiter"])

    if plot_df.empty:
        print("No per-machine NOxMAF data available after aggregation.")
        return

    plt.figure(figsize=(12, 6))
    ax = sns.scatterplot(
        data=plot_df,
        x="Motorbelasting",
        y="NOxPerLiter",
        hue="MainGroupLabel",
        palette="tab10",
        s=70,
        alpha=0.7,
    )
    plt.title("NOxPerLiter (kg/L) vs Motorbelasting (%) — 1 punt per machine")
    plt.xlabel("Motorbelasting (%)")
    plt.ylabel("NOxPerLiter (kg/L)")
    plt.axhline(0.0016, color="gray", linestyle="--", linewidth=1)
    plt.legend(title="MainGroupLabel", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    counts = (
        plot_df.groupby("MainGroupLabel")["MachineId"]
        .nunique()
        .sort_values(ascending=False)
    )
    print("\nUnique machines per MainGroupLabel (plotted points):")
    print(counts)


def plot_noxmaf_plotly(df: pd.DataFrame, stage_groep: pd.Series, stage_filter: str):
    """Interactive Plotly scatter filtered to a specific Stage+Groep."""
    missing_cols = [name for name in NOXMAF_COLUMNS.values() if name not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing NOxMAF columns: {missing_cols}")

    if "MainGroupLabel" not in df.columns:
        raise ValueError("MainGroupLabel column is required for Plotly coloring.")

    base_df = pd.DataFrame(
        {
            "Motorbelasting": df[NOXMAF_COLUMNS["Motorbelasting"]],
            "NOxPerLiter": df[NOXMAF_COLUMNS["NOxPerLiter"]],
            "Stage+Groep": stage_groep,
            "MainGroupLabel": df["MainGroupLabel"].fillna("Unknown"),
            "MachineId": df.get("MachineId"),
        }
    )

    filtered = base_df[(base_df["Stage+Groep"] == stage_filter)].dropna(
        subset=["Motorbelasting", "NOxPerLiter"]
    )

    if filtered.empty:
        print(f"No data available for Stage+Groep = {stage_filter}")
        return

    fig = px.scatter(
        filtered,
        x="Motorbelasting",
        y="NOxPerLiter",
        color="MainGroupLabel",
        hover_data=["Stage+Groep", "MachineId"],
        title=f"NOxPerLiter vs Motorbelasting (Stage+Groep = {stage_filter})",
        labels={
            "Motorbelasting": "Motorbelasting (%)",
            "NOxPerLiter": "NOxPerLiter (kg/L)",
            "MainGroupLabel": "Main Group",
        },
    )
    fig.update_traces(marker=dict(size=7, opacity=0.6))
    fig.add_hline(
        y=0.0016,
        line_dash="dash",
        line_color="gray",
        annotation_text="1.6 g/L threshold",
        annotation_position="top left",
    )
    fig.update_layout(legend_title_text="MainGroupLabel")
    fig.show()


def plot_noxmaf_plotly_absolute(
    df: pd.DataFrame, stage_groep: pd.Series, stage_filter: str
):
    """Plotly scatter for Stage+Groep showing absolute NOx emissions."""
    required_cols = list(NOXMAF_COLUMNS.values()) + [NOXMAF_ABSOLUTE_COLUMN]
    missing_cols = [name for name in required_cols if name not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing NOxMAF columns: {missing_cols}")
    if "MainGroupLabel" not in df.columns:
        raise ValueError("MainGroupLabel column is required for Plotly coloring.")

    base_df = pd.DataFrame(
        {
            "Motorbelasting": df[NOXMAF_COLUMNS["Motorbelasting"]],
            "NOxAbsolute": df[NOXMAF_ABSOLUTE_COLUMN],
            "Stage+Groep": stage_groep,
            "MainGroupLabel": df["MainGroupLabel"].fillna("Unknown"),
            "MachineId": df.get("MachineId"),
        }
    )

    filtered = base_df[(base_df["Stage+Groep"] == stage_filter)].dropna(
        subset=["Motorbelasting", "NOxAbsolute"]
    )

    if filtered.empty:
        print(f"No data available for Stage+Groep = {stage_filter}")
        return

    fig = px.scatter(
        filtered,
        x="Motorbelasting",
        y="NOxAbsolute",
        color="MainGroupLabel",
        hover_data=["Stage+Groep", "MachineId"],
        title=f"Absolute NOx vs Motorbelasting (Stage+Groep = {stage_filter})",
        labels={
            "Motorbelasting": "Motorbelasting (%)",
            "NOxAbsolute": "NOx emission (absolute)",
            "MainGroupLabel": "Main Group",
        },
    )
    fig.update_traces(marker=dict(size=7, opacity=0.6))
    fig.update_layout(legend_title_text="MainGroupLabel")
    fig.show()


def plot_noxmaf_plotly_colored_by_hours(
    df: pd.DataFrame, stage_groep: pd.Series, stage_filter: str
):
    """Interactive Plotly scatter filtered to Stage+Groep and colored by hours."""
    hours = build_hours_series(df)
    if hours.isna().all():
        print("No hours-like columns available to color Plotly scatter.")
        return

    base_df = pd.DataFrame(
        {
            "Motorbelasting": df[NOXMAF_COLUMNS["Motorbelasting"]],
            "NOxPerLiter": df[NOXMAF_COLUMNS["NOxPerLiter"]],
            "Stage+Groep": stage_groep,
            "Hours": hours,
            "MachineId": df.get("MachineId"),
            "MainGroupLabel": df.get("MainGroupLabel", "Unknown"),
        }
    )

    filtered = base_df[(base_df["Stage+Groep"] == stage_filter)].dropna(
        subset=["Motorbelasting", "NOxPerLiter", "Hours"]
    )

    if filtered.empty:
        print(
            f"No data available for Stage+Groep = {stage_filter} with non-null hours values."
        )
        return

    fig = px.scatter(
        filtered,
        x="Motorbelasting",
        y="NOxPerLiter",
        color="Hours",
        color_continuous_scale="Viridis",
        hover_data=["Stage+Groep", "MachineId", "MainGroupLabel"],
        title=f"NOxPerLiter vs Motorbelasting (Stage+Groep = {stage_filter}) — colored by hours",
        labels={
            "Motorbelasting": "Motorbelasting (%)",
            "NOxPerLiter": "NOxPerLiter (kg/L)",
            "Hours": "Hours",
        },
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.add_hline(
        y=0.0016,
        line_dash="dash",
        line_color="gray",
        annotation_text="1.6 g/L threshold",
        annotation_position="top left",
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Hours"))
    fig.show()


def plot_noxmaf_absolute_colored_by_hours(
    df: pd.DataFrame, stage_groep: pd.Series, stage_filter: str
):
    """Absolute NOx scatter colored by hours for a specific Stage+Groep."""
    required_cols = list(NOXMAF_COLUMNS.values()) + [NOXMAF_ABSOLUTE_COLUMN]
    missing_cols = [name for name in required_cols if name not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing NOxMAF columns: {missing_cols}")

    hours = build_hours_series(df)
    if hours.isna().all():
        print("No hours-like columns available to color absolute NOx scatter.")
        return

    base_df = pd.DataFrame(
        {
            "Motorbelasting": df[NOXMAF_COLUMNS["Motorbelasting"]],
            "NOxAbsolute": df[NOXMAF_ABSOLUTE_COLUMN],
            "Stage+Groep": stage_groep,
            "Hours": hours,
            "MachineId": df.get("MachineId"),
            "MainGroupLabel": df.get("MainGroupLabel", "Unknown"),
        }
    )

    filtered = base_df[(base_df["Stage+Groep"] == stage_filter)].dropna(
        subset=["Motorbelasting", "NOxAbsolute", "Hours"]
    )

    if filtered.empty:
        print(
            f"No data available for Stage+Groep = {stage_filter} with non-null hours values."
        )
        return

    fig = px.scatter(
        filtered,
        x="Motorbelasting",
        y="NOxAbsolute",
        color="Hours",
        color_continuous_scale="Viridis",
        hover_data=["Stage+Groep", "MachineId", "MainGroupLabel"],
        title=f"Absolute NOx vs Motorbelasting (Stage+Groep = {stage_filter}) — colored by hours",
        labels={
            "Motorbelasting": "Motorbelasting (%)",
            "NOxAbsolute": "NOx emission (absolute)",
            "Hours": "Hours",
        },
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(coloraxis_colorbar=dict(title="Hours"))
    fig.show()


def plot_noxmaf_plotly_colored_by_fuel(
    df: pd.DataFrame, stage_groep: pd.Series, stage_filter: str
):
    """Interactive Plotly scatter filtered to Stage+Groep and colored by fuel use."""
    fuel = build_fuel_series(df)
    if fuel.isna().all():
        print("No fuel-like columns available to color Plotly scatter.")
        return

    base_df = pd.DataFrame(
        {
            "Motorbelasting": df[NOXMAF_COLUMNS["Motorbelasting"]],
            "NOxPerLiter": df[NOXMAF_COLUMNS["NOxPerLiter"]],
            "Stage+Groep": stage_groep,
            "Fuel": fuel,
            "MachineId": df.get("MachineId"),
            "MainGroupLabel": df.get("MainGroupLabel", "Unknown"),
        }
    )

    filtered = base_df[(base_df["Stage+Groep"] == stage_filter)].dropna(
        subset=["Motorbelasting", "NOxPerLiter", "Fuel"]
    )

    if filtered.empty:
        print(
            f"No data available for Stage+Groep = {stage_filter} with non-null fuel values."
        )
        return

    fig = px.scatter(
        filtered,
        x="Motorbelasting",
        y="NOxPerLiter",
        color="Fuel",
        color_continuous_scale="Plasma",
        hover_data=["Stage+Groep", "MachineId", "MainGroupLabel"],
        title=f"NOxPerLiter vs Motorbelasting (Stage+Groep = {stage_filter}) — colored by fuel use",
        labels={
            "Motorbelasting": "Motorbelasting (%)",
            "NOxPerLiter": "NOxPerLiter (kg/L)",
            "Fuel": "Fuel use",
        },
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.add_hline(
        y=0.0016,
        line_dash="dash",
        line_color="gray",
        annotation_text="1.6 g/L threshold",
        annotation_position="top left",
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Fuel"))
    fig.show()


def compute_machine_relative_nox(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate NOx totals and compute relative difference per machine."""
    required_cols = {"MachineId", "NOxTotal", "NOxMAF_AUB_NoxEmission"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns required for machine aggregation: {missing}")

    agg_map = {
        "NOxTotal_sum": ("NOxTotal", "sum"),
        "NOxAUB_sum": ("NOxMAF_AUB_NoxEmission", "sum"),
        "records": ("MachineId", "size"),
        "MainGroupLabel": ("MainGroupLabel", _mode_or_first_nonnull),
    }
    if "Machinegroep" in df.columns:
        agg_map["Machinegroep"] = ("Machinegroep", _mode_or_first_nonnull)
    motor_col = NOXMAF_COLUMNS["Motorbelasting"]
    if motor_col in df.columns:
        agg_map["Motorbelasting_mean"] = (motor_col, "mean")

    grouped = df.groupby("MachineId").agg(**agg_map).reset_index()

    valid = grouped[
        grouped["NOxAUB_sum"].notna() & (grouped["NOxAUB_sum"] != 0)
    ].copy()
    valid = valid[
        valid["NOxAUB_sum"].notna() & (valid["NOxAUB_sum"] != 0)
    ].copy()
    valid["relative_diff"] = (
        valid["NOxTotal_sum"] - valid["NOxAUB_sum"]
    ) / valid["NOxAUB_sum"]
    return valid


def plot_machine_relative_nox_difference(df: pd.DataFrame):
    """Bar chart of machine-level relative difference between NOxTotal and AUB NOx."""
    valid = compute_machine_relative_nox(df)
    if valid.empty:
        print("No machines have both NOxTotal and NOxMAF_AUB_NoxEmission data.")
        return
    if "Motorbelasting_mean" not in valid.columns:
        raise ValueError(
            "Motorbelasting column is missing; cannot color by motorbelasting."
        )

    valid["MachineId_str"] = valid["MachineId"].astype(str)
    valid = valid.sort_values("relative_diff")

    fig = px.bar(
        valid,
        x="MachineId_str",
        y="relative_diff",
        color="Motorbelasting_mean",
        color_continuous_scale="Viridis",
        hover_data={
            "NOxTotal_sum": ":.2f",
            "NOxAUB_sum": ":.2f",
            "records": True,
            "MainGroupLabel": True,
            "Motorbelasting_mean": ":.1f",
        },
        labels={
            "MachineId_str": "MachineId",
            "relative_diff": "Relative difference (NOxTotal vs AUB)",
            "Motorbelasting_mean": "Gemiddelde motorbelasting (%)",
        },
        title="Machine-level relative NOx difference (NOxTotal vs NOxMAF AUB) — gekleurd op motorbelasting",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis_title="MachineId",
        yaxis_title="Relative difference (NOxTotal - AUB) / AUB",
        coloraxis_colorbar=dict(title="Motorbelasting (%)"),
    )
    fig.show()


def plot_machine_relative_nox_difference_value_colored(df: pd.DataFrame):
    """Bar chart of relative NOx difference colored by the value itself."""
    valid = compute_machine_relative_nox(df)
    if valid.empty:
        print("No machines have both NOxTotal and NOxMAF_AUB_NoxEmission data.")
        return

    valid["MachineId_str"] = valid["MachineId"].astype(str)
    valid["Machinegroep"] = valid.get("Machinegroep", "Unknown").fillna("Unknown")
    valid["MainGroupLabel"] = valid["MainGroupLabel"].fillna("Unknown")
    valid["Machinegroep_A"] = valid["Machinegroep"].astype(str).str.upper().eq("A")
    valid = valid.sort_values("relative_diff", ascending=False)

    fig = px.bar(
        valid,
        x="MachineId_str",
        y="relative_diff",
        color="relative_diff",
        color_continuous_scale="RdYlGn_r",
        pattern_shape="Machinegroep_A",
        pattern_shape_map={True: "/", False: ""},
        hover_data={
            "NOxTotal_sum": ":.2f",
            "NOxAUB_sum": ":.2f",
            "records": True,
            "MainGroupLabel": True,
            "Machinegroep": True,
        },
        labels={
            "MachineId_str": "MachineId",
            "relative_diff": "Relative difference (NOxTotal - AUB) / AUB",
        },
        title="Verschil gemeten waarden tov AUB methode per machine",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis_title="Machine type (MainGroupLabel)",
        yaxis_title="Relative difference (NOxTotal - AUB) / AUB",
        coloraxis_colorbar=dict(title="Relative diff"),
        plot_bgcolor="white",
        legend_title_text="Machinegroep A",
        xaxis=dict(
            categoryorder="array",
            categoryarray=valid["MachineId_str"],
        ),
    )
    fig.update_xaxes(tickvals=valid["MachineId_str"], ticktext=valid["MainGroupLabel"])
    fig.show()


def print_average_machine_relative_diff(df: pd.DataFrame):
    """Print unweighted average of machine-level relative differences."""
    valid = compute_machine_relative_nox(df)
    if valid.empty:
        print("No machines available to compute average relative difference.")
        return
    avg_rel_diff = valid["relative_diff"].mean()
    avg_abs_rel_diff = valid["relative_diff"].abs().mean()
    print(
        f"Average machine-level relative difference (unweighted): {avg_rel_diff:.4f} "
        f"(mean absolute: {avg_abs_rel_diff:.4f})"
    )


def plot_noxmaf_scatter_bouwjaar(df: pd.DataFrame):
    """Scatter colored by ConstructionYear (Bouwjaar)."""
    missing_cols = [name for name in NOXMAF_COLUMNS.values() if name not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing NOxMAF columns: {missing_cols}")

    if "ConstructionYear" not in df.columns:
        raise ValueError("ConstructionYear column is required for Bouwjaar scatter.")

    plot_df = pd.DataFrame(
        {
            "Motorbelasting": df[NOXMAF_COLUMNS["Motorbelasting"]],
            "NOxPerLiter": df[NOXMAF_COLUMNS["NOxPerLiter"]],
            "Bouwjaar": df["ConstructionYear"],
        }
    ).dropna(subset=["Motorbelasting", "NOxPerLiter", "Bouwjaar"])

    if plot_df.empty:
        print("No rows with Bouwjaar available for plotting.")
        return

    plot_df["Bouwjaar"] = plot_df["Bouwjaar"].astype(int)
    ordered_years = sorted(plot_df["Bouwjaar"].unique())
    cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, len(ordered_years))))
    norm = BoundaryNorm(np.arange(len(ordered_years) + 1) - 0.5, cmap.N)
    year_index = plot_df["Bouwjaar"].apply(lambda y: ordered_years.index(y))

    plt.figure(figsize=(12, 6))
    plt.scatter(
        plot_df["Motorbelasting"],
        plot_df["NOxPerLiter"],
        c=year_index,
        cmap=cmap,
        norm=norm,
        alpha=0.6,
        s=45,
    )
    plt.title("NOxPerLiter (kg/L) vs Motorbelasting (%) — colored by Bouwjaar")
    plt.xlabel("Motorbelasting (%)")
    plt.ylabel("NOxPerLiter (kg/L)")
    plt.axhline(0.0016, color="gray", linestyle="--", linewidth=1)
    cbar = plt.colorbar(
        ticks=range(len(ordered_years)),
    )
    cbar.ax.set_yticklabels([str(year) for year in ordered_years])
    cbar.ax.set_ylabel("Bouwjaar (ConstructionYear)")
    plt.tight_layout()
    plt.show()


def plot_machine_nox_in_grams_per_liter(df: pd.DataFrame):
    """Horizontal bar chart of average NOx per machine in grams per liter colored by machine type."""
    required_cols = [NOXMAF_COLUMNS["NOxPerLiter"], "MachineId"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns for machine NOx chart: {missing_cols}")

    agg_map = {
        "NOx_kg_per_liter": (NOXMAF_COLUMNS["NOxPerLiter"], "mean"),
    }
    if "MainGroupLabel" in df.columns:
        agg_map["MainGroupLabel"] = ("MainGroupLabel", _mode_or_first_nonnull)

    per_machine = df.groupby("MachineId").agg(**agg_map).reset_index()
    per_machine = per_machine.dropna(subset=["NOx_kg_per_liter"])
    if per_machine.empty:
        print("No NOxPerLiter values available to plot per machine.")
        return

    per_machine["NOx_g_per_liter"] = per_machine["NOx_kg_per_liter"] * GRAMS_PER_KILOGRAM
    per_machine["MachineId_str"] = per_machine["MachineId"].astype(str)
    if "MainGroupLabel" in per_machine.columns:
        machine_type_series = per_machine["MainGroupLabel"].fillna("Unknown")
    else:
        machine_type_series = pd.Series("Unknown", index=per_machine.index)
    per_machine["MachineType"] = machine_type_series
    per_machine = per_machine.sort_values("NOx_g_per_liter", ascending=False)

    unique_types = per_machine["MachineType"].unique()
    palette = sns.color_palette("tab20", n_colors=len(unique_types))
    type_to_color = {m_type: palette[i] for i, m_type in enumerate(unique_types)}
    colors = per_machine["MachineType"].map(type_to_color)

    fig_height = max(6, 0.35 * len(per_machine))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    bars = ax.barh(per_machine["MachineId_str"], per_machine["NOx_g_per_liter"], color=colors)
    ax.axvline(
        NOX_G_PER_L_THRESHOLD_LOW,
        color="#2ca02c",
        linestyle="--",
        linewidth=1,
    )
    ax.axvline(
        NOX_G_PER_L_THRESHOLD_MEDIUM,
        color="#d62728",
        linestyle="--",
        linewidth=1,
    )
    ax.set_xlabel("NOx (gram per liter)")
    ax.set_ylabel("MachineId")
    ax.set_title("Gemiddelde NOx per machine (g/L)")
    ax.invert_yaxis()

    text_offset = 0.05 * per_machine["NOx_g_per_liter"].max()
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + text_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f} g/L",
            va="center",
            ha="left",
        )

    band_handles = [Patch(facecolor=color, label=label) for label, color in type_to_color.items()]
    ax.legend(
        handles=band_handles,
        title="Machine type",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load NOx dataset from the Exploratory directory
    data_dir = Path(__file__).resolve().parent
    df_nox = pd.read_csv(data_dir / "data/noxdagdata 2026-01-15T111711.csv")# "nox_data.csv")
    df_nox = filter_by_bouwjaar(df_nox, MIN_VALID_YEAR)
    if "MachineId" in df_nox.columns:
        initial_rows = len(df_nox)
        df_nox = df_nox[~df_nox["MachineId"].isin(EXCLUDED_MACHINES)].copy()
        removed = initial_rows - len(df_nox)
        if removed > 0:
            print(f"Removed {removed} rows for manually excluded machines: {sorted(EXCLUDED_MACHINES)}")

    # Filter out machines with invalid sensor data (motorbelasting > 100% or negative exhausts)
    if "MachineId" in df_nox.columns:
        invalid_sensor_machines = find_machines_with_invalid_sensor_data(df_nox)
        if invalid_sensor_machines:
            initial_rows = len(df_nox)
            df_nox = df_nox[~df_nox["MachineId"].isin(invalid_sensor_machines)].copy()
            removed = initial_rows - len(df_nox)
            print(
                f"Removed {removed} rows for {len(invalid_sensor_machines)} machines "
                f"with invalid sensor data (motorbelasting > 100% or negative exhaust values)"
            )

    # Filter out individual rows with invalid sensor values
    initial_rows = len(df_nox)
    invalid_row_mask = pd.Series(False, index=df_nox.index)

    # Exclude rows with motorbelasting > 1 (100%)
    motorbelasting_col = NOXMAF_COLUMNS["Motorbelasting"]
    if motorbelasting_col in df_nox.columns:
        high_motorbelasting = df_nox[motorbelasting_col] > 1
        invalid_row_mask |= high_motorbelasting
        count = high_motorbelasting.sum()
        if count > 0:
            print(f"  - {count} rows with motorbelasting > 100%")

    # Exclude rows with negative exhaust values
    for exhaust_col in EXHAUST_COLUMNS:
        if exhaust_col not in df_nox.columns:
            continue
        negative_exhaust = df_nox[exhaust_col] < 0
        invalid_row_mask |= negative_exhaust
        count = negative_exhaust.sum()
        if count > 0:
            print(f"  - {count} rows with negative {exhaust_col}")

    if invalid_row_mask.sum() > 0:
        df_nox = df_nox[~invalid_row_mask].copy()
        removed = initial_rows - len(df_nox)
        print(f"Removed {removed} individual rows with invalid sensor values (motorbelasting > 100% or negative exhausts)")

    hours_series = build_hours_series(df_nox)
    fuel_series = build_fuel_series(df_nox)
    valid_mask = (
        hours_series.notna()
        & (hours_series >= HOURS_MIN_THRESHOLD)
        & fuel_series.notna()
        & (fuel_series >= FUEL_MIN_THRESHOLD)
    )
    removed_mask_count = (~valid_mask).sum()
    if removed_mask_count > 0:
        print(
            f"Filtered out {removed_mask_count} rows with hours < {HOURS_MIN_THRESHOLD} "
            f"or fuel < {FUEL_MIN_THRESHOLD}."
        )
    df_nox = df_nox.loc[valid_mask].copy()

    df_focus = build_focus_dataframe(df_nox, FOCUS_COLUMNS)
    stage_groep_labels = build_stage_groep_label(df_focus)

    numeric_cols = df_nox.select_dtypes(include=[np.number])
    combined_df = pd.concat([df_focus, numeric_cols], axis=1)

    print("=" * 60)
    print("NOX DATA PREVIEW (Stage, Machinegroep, numeric features)")
    print("=" * 60)
    print(combined_df.head())

    features = prepare_feature_matrix(combined_df)
    corr_matrix = features.corr()

    print("\nCorrelation matrix shape:", corr_matrix.shape)
    plot_correlation_heatmap(corr_matrix, "NOx Feature Correlation (Stage/Machinegroep + Numeric)")

    print("\n" + "=" * 60)
    print("NOxMAF Emissions vs Motorbelasting")
    print("=" * 60)
    plot_noxmaf_scatter(df_nox, stage_groep_labels, stage_filter=PLOTLY_STAGE_FILTER)

    print("\n" + "=" * 60)
    print("NOxMAF Emissions colored by Bouwjaar (ConstructionYear)")
    print("=" * 60)
    plot_noxmaf_scatter_bouwjaar(df_nox)

    print("\n" + "=" * 60)
    print(f"Plotly NOxMAF Emissions (Stage+Groep = {PLOTLY_STAGE_FILTER})")
    print("=" * 60)
    plot_noxmaf_plotly(df_nox, stage_groep_labels, PLOTLY_STAGE_FILTER)

    print("\n" + "=" * 60)
    print(
        f"Plotly NOxMAF Absolute Emissions (Stage+Groep = {PLOTLY_STAGE_FILTER})"
    )
    print("=" * 60)
    plot_noxmaf_plotly_absolute(df_nox, stage_groep_labels, PLOTLY_STAGE_FILTER)

    print("\n" + "=" * 60)
    print(
        f"Plotly NOxMAF Emissions (Stage+Groep = {PLOTLY_STAGE_FILTER}) colored by hours"
    )
    print("=" * 60)
    plot_noxmaf_plotly_colored_by_hours(df_nox, stage_groep_labels, PLOTLY_STAGE_FILTER)

    print("\n" + "=" * 60)
    print(
        f"Plotly NOxMAF Emissions (Stage+Groep = {PLOTLY_STAGE_FILTER}) colored by fuel use"
    )
    print("=" * 60)
    plot_noxmaf_plotly_colored_by_fuel(df_nox, stage_groep_labels, PLOTLY_STAGE_FILTER)

    print("\n" + "=" * 60)
    print(
        f"Absolute NOx vs Motorbelasting (Stage+Groep = {PLOTLY_STAGE_FILTER}) colored by hours"
    )
    print("=" * 60)
    plot_noxmaf_absolute_colored_by_hours(
        df_nox, stage_groep_labels, PLOTLY_STAGE_FILTER
    )

    print("\n" + "=" * 60)
    print("Machine-level relative differences (NOxTotal vs NOxMAF AUB)")
    print("=" * 60)
    plot_machine_relative_nox_difference(df_nox)
    plot_machine_relative_nox_difference_value_colored(df_nox)
    print_average_machine_relative_diff(df_nox)

    print("\n" + "=" * 60)
    print("NOxPerLiter vs Motorbelasting — per machine, gekleurd per MainGroupLabel")
    print("=" * 60)
    plot_noxmaf_per_machine_by_maingroup(df_nox)

    print("\n" + "=" * 60)
    print("NOx per machine (g/L) met duidelijke grenswaarden")
    print("=" * 60)
    plot_machine_nox_in_grams_per_liter(df_nox)

# %%
