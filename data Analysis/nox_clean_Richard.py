"""
NOx Analysis for Boskalis Pilot - Clean Version

This script provides a structured three-level analysis of NOx emissions data:
1. MACHINE-LEVEL: Overview of average motorbelasting and NOx per machine
2. MACHINE-DAY LEVEL: Detailed daily patterns with fitted curves per stage+group
3. SINGLE-MACHINE: Deep dive into temporal patterns for a specific machine

Data source: Latest noxdagdata CSV from data folder
"""

# %% ============================================================================
# IMPORTS AND CONFIGURATION
# ==============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import curve_fit
from IPython.display import display

from nox_helpers import (
    certification_limit,
    certification_limit_for_stage_group,
    nox_per_liter_polynomial,
    model_fit_function,
    get_tno_coefficients,
    get_adblue_percentages,
    STAGE_GROUP_COLORS,
)

# Global plotting configuration
X_MIN = 0.05  # Minimum x-axis value for certification limit lines (5%)


# %% ============================================================================
# FILTER CONFIGURATION
# ==============================================================================
# All filter thresholds in one place for easy adjustment

# Master filters (applied before data quality filters)
FILTER_FMS = "FleetsonlineV2"  # None to disable, or "FleetsonlineV2" / "GPS Buddy"
FILTER_PILOT = None #"Boskalis"  # None to disable, or specific pilot name like "Boskalis"

# Exclusion lists
SUSPICIOUS_MACHINES = [
    1614,  # Reason: unrealistic high NOx for belasting > 30%
    2412,  # Reason: unrealistic high NOx for belasting > 30%
    2480, 2481 ## werkschip aandrijving
]
SUSPICIOUS_PILOTS = [
    "SOMA",
    "Fontys",
]

# Date range filter (None to disable)
MIN_DATEKEY = None              # Minimum datekey (e.g., 20260101 for Jan 1, 2026), None to disable
MAX_DATEKEY = None              # Maximum datekey (e.g., 20260131 for Jan 31, 2026), None to disable

# Data quality filter thresholds
MIN_DURATION_HOURS = 1          # Filter 1: Minimum duration per day (hours)
MIN_FUEL_MASS_FLOW = 1          # Filter 1: Minimum fuel mass flow (L)
MIN_MOTORBELASTING = 0          # Filter 1: Minimum motorbelasting (0 = 0%)
MAX_MOTORBELASTING = 1          # Filter 1: Maximum motorbelasting (1 = 100%)
MIN_UNIQUE_DAYS = 1            # Filter 2: Minimum unique datekeys per machine
MAX_FF_FPH = 200                # Filter 4: Maximum FF fuel per hour (L/h)
MAX_NOXMAF_FPH = 200            # Filter 5: Maximum NOxMAF fuel per hour (L/h)
FPH_CONSISTENCY_PCT = 30        # Filter 6: Max deviation between NOxMAF_FPH and FF_FPH (%)
DURATION_CONSISTENCY_PCT = 10   # Filter 7: Max deviation between FF_validated_duration and duration_from_rows_valid (%)

# Curve fitting configuration
MIN_MOTORBELASTING_FOR_FIT = 0.05  # Minimum motorbelasting for curve fitting (5%)


# %% ============================================================================
# DATA LOADING AND MASTER FILTERING
# ==============================================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

# Load the latest noxdagdata CSV from the data folder
data_folder = Path("data")
noxdagdata_files = sorted(data_folder.glob("noxdagdata *.csv"), reverse=True)

if not noxdagdata_files:
    raise FileNotFoundError(f"No noxdagdata CSV files found in {data_folder}")

DATA_PATH = noxdagdata_files[0]
print(f"Loading latest CSV: {DATA_PATH.name}")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"CSV not found: {DATA_PATH}")

nox_df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(nox_df):,} rows × {len(nox_df.columns)} columns\n")
print(f"Before master filters: {nox_df['MachineId'].nunique()} unique machines")

# Apply master filters (configured in FILTER CONFIGURATION section)
if FILTER_PILOT:
    nox_df = nox_df[nox_df["Pilot"] == FILTER_PILOT].copy()
    print(f"After {FILTER_PILOT} filter: {nox_df['MachineId'].nunique()} unique machines")

if FILTER_FMS:
    nox_df = nox_df[nox_df["SensorSupplier"] == FILTER_FMS].copy()
    print(f"After {FILTER_FMS} filter: {nox_df['MachineId'].nunique()} unique machines")

if SUSPICIOUS_MACHINES:
    n_before = nox_df["MachineId"].nunique()
    nox_df = nox_df[~nox_df["MachineId"].isin(SUSPICIOUS_MACHINES)].copy()
    n_after = nox_df["MachineId"].nunique()
    print(f"After excluding {len(SUSPICIOUS_MACHINES)} suspicious machines: {n_after} unique machines (removed {n_before - n_after})")

if SUSPICIOUS_PILOTS:
    n_before = nox_df["MachineId"].nunique()
    nox_df = nox_df[~nox_df["Pilot"].isin(SUSPICIOUS_PILOTS)].copy()
    n_after = nox_df["MachineId"].nunique()
    print(f"After excluding {len(SUSPICIOUS_PILOTS)} suspicious pilots: {n_after} unique machines (removed {n_before - n_after})")

# Create Stage+Groep combination column
nox_df["Stage+Groep"] = (
    nox_df["EngineClassificationLabel"].astype(str) + "+" +
    nox_df["Machinegroep"].astype(str)
)

# Create MerkType column (Brand + Type)
nox_df["MerkType"] = (
    nox_df["BrandLabel"].fillna("Unknown").astype(str) + " " +
    nox_df["TypeOfEquipment"].fillna("Unknown").astype(str)
)

# Convert NOx data from kg/L to g/L (multiply by 1000)
print("\nConverting NOx values from kg/L to g/L...")
nox_columns_to_convert = ["NOxMAF_NOxPerLiter", "FF_NOxPerLiter", "CANBUS_NOxPerLiter"]
for col in nox_columns_to_convert:
    if col in nox_df.columns:
        nox_df[col] = nox_df[col] * 1000
        print(f"  ✓ Converted {col} from kg/L to g/L")


# %% ===========================================================================
# Data Quality filters
# ==============================================================================

print("\n" + "=" * 80)
print("APPLYING DATA QUALITY FILTERS")
print("=" * 80)

n_machines_start = nox_df["MachineId"].nunique()
print(f"Starting with: {n_machines_start} machines\n")

# Create motorbelasting and hour buckets
mb_bins = [0.0, 0.10, 0.2, 0.3, 0.4, 1.0]
nox_df["motorbelasting_bucket_interval"] = pd.cut(nox_df["FF_motorbelasting"], bins=mb_bins, right=False)

hour_bins = pd.interval_range(start=0, end=nox_df["duration_from_rows"].max() + 0.5, freq=0.5, closed='left')
nox_df["hour_bucket_interval"] = pd.cut(nox_df["duration_from_rows"], bins=hour_bins)

# Filter 1: Duration and fuel mass flow
n_before = nox_df["MachineId"].nunique()
machines_before_filter1 = set(nox_df["MachineId"].unique())
filter_mask = (
    (nox_df["duration_from_rows"] > MIN_DURATION_HOURS)
    & (nox_df["FuelMassFlow"].notnull())
    & (nox_df["FuelMassFlow"] > MIN_FUEL_MASS_FLOW)
    & (nox_df["NOxMAF_motorbelasting"] >= MIN_MOTORBELASTING)
    & (nox_df["NOxMAF_motorbelasting"] <= MAX_MOTORBELASTING)
)
nox_df = nox_df[filter_mask].copy()
n_after = nox_df["MachineId"].nunique()
machines_after_filter1 = set(nox_df["MachineId"].unique())
machines_lost_filter1 = machines_before_filter1 - machines_after_filter1
print(f"✓ Duration > {MIN_DURATION_HOURS} & FuelMassFlow > {MIN_FUEL_MASS_FLOW} & motorbelasting [0,1]: {n_after} machines (lost {n_before - n_after})")

# Create DataFrame of machines lost in Filter 1
lost_filter1_df = pd.read_csv(DATA_PATH)
lost_filter1_df = lost_filter1_df[lost_filter1_df["MachineId"].isin(machines_lost_filter1)].copy()
print(f"  → Saved {lost_filter1_df['MachineId'].nunique()} lost machines to lost_filter1_df ({len(lost_filter1_df):,} rows)")

# Filter 2: Minimum number of unique days per machine
n_before = nox_df["MachineId"].nunique()
datekey_counts = nox_df.groupby("MachineId")["datekey"].nunique()
machines_with_enough_days = datekey_counts[datekey_counts >= MIN_UNIQUE_DAYS].index
nox_df = nox_df[nox_df["MachineId"].isin(machines_with_enough_days)].copy()
n_after = nox_df["MachineId"].nunique()
print(f"✓ At least {MIN_UNIQUE_DAYS} unique datekeys: {n_after} machines (lost {n_before - n_after})")

# Filter 3: Stage IV and V only
n_before = nox_df["MachineId"].nunique()
stage_mask = nox_df["EngineClassificationLabel"].str.lower().str.contains("stage-iv|stage-v|stage iv|stage v", regex=True, na=False)
nox_df = nox_df[stage_mask].copy()
n_after = nox_df["MachineId"].nunique()
print(f"✓ Stage IV/V only: {n_after} machines (lost {n_before - n_after})")

# Filter 4: Remove extremely large FF_FPH values
n_before = nox_df["MachineId"].nunique()
nox_df = nox_df[(nox_df["FF_FPH"] <= MAX_FF_FPH) | (nox_df["FF_FPH"].isna())].copy()
n_after = nox_df["MachineId"].nunique()
print(f"✓ FF_FPH <= {MAX_FF_FPH} (or null): {n_after} machines (lost {n_before - n_after})")

# Filter 5: Remove extremely large NOxMAF_FPH values
n_before = nox_df["MachineId"].nunique()
nox_df = nox_df[(nox_df["NOxMAF_FPH"] <= MAX_NOXMAF_FPH) | (nox_df["NOxMAF_FPH"].isna())].copy()
n_after = nox_df["MachineId"].nunique()
print(f"✓ NOxMAF_FPH <= {MAX_NOXMAF_FPH} (or null): {n_after} machines (lost {n_before - n_after})")

# Compute NOxTotal fraction per machine
machine_nox_totals = nox_df.groupby("MachineId")["NOxTotal"].transform("sum")
nox_df["NOxTotal_fraction_of_machine_total"] = nox_df["NOxTotal"] / machine_nox_totals

# Filter 6: Fuel sensor consistency (NOxMAF_FPH within ±X% of FF_FPH)
# Filter out individual machine-days with large deviation, then check if any machines lost all days
n_rows_before = len(nox_df)
n_machines_before = nox_df["MachineId"].nunique()

# Calculate deviation per row (machine-day)
has_both_sensors = nox_df["FF_FPH"].notna() & nox_df["NOxMAF_FPH"].notna()
nox_df["fph_deviation_pct"] = np.nan
nox_df.loc[has_both_sensors, "fph_deviation_pct"] = (
    abs((nox_df.loc[has_both_sensors, "NOxMAF_FPH"] - nox_df.loc[has_both_sensors, "FF_FPH"])
        / nox_df.loc[has_both_sensors, "FF_FPH"]) * 100
)

display(nox_df)

# Keep rows that: pass the check, OR don't have both sensor values (not subject to filter)
rows_to_keep = (nox_df["fph_deviation_pct"] <= FPH_CONSISTENCY_PCT) | (~has_both_sensors)
nox_df = nox_df[rows_to_keep].copy()
nox_df = nox_df.drop(columns=["fph_deviation_pct"])

n_rows_after = len(nox_df)
n_machines_after = nox_df["MachineId"].nunique()
print(f"✓ Fuel sensor consistency (±{FPH_CONSISTENCY_PCT}%): {n_rows_after:,} rows (lost {n_rows_before - n_rows_after:,} machine-days, {n_machines_before - n_machines_after} machines)")

# Filter 7: Duration consistency (FPH duration within ±X% of duration_from_rows_valid)
# Filter out individual machine-days with large deviation, then check if any machines lost all days
n_rows_before = len(nox_df)
n_machines_before = nox_df["MachineId"].nunique()

# Calculate deviation per row (machine-day)
has_both_durations = nox_df["FF_validated_duration"].notna() & nox_df["duration_from_rows_valid"].notna()
nox_df["duration_deviation_pct"] = np.nan
nox_df.loc[has_both_durations, "duration_deviation_pct"] = (
    abs((nox_df.loc[has_both_durations, "FF_validated_duration"] - nox_df.loc[has_both_durations, "duration_from_rows_valid"])
        / nox_df.loc[has_both_durations, "duration_from_rows_valid"]) * 100
)

# Count machines without FF_validated_duration (not subject to this filter)
machines_without_ff_duration = nox_df.groupby("MachineId")["FF_validated_duration"].apply(lambda x: x.isna().all())
n_machines_no_ff_duration = machines_without_ff_duration.sum()

# Keep rows that: pass the check, OR don't have both duration values (not subject to filter)
rows_to_keep = (nox_df["duration_deviation_pct"] <= DURATION_CONSISTENCY_PCT) | (~has_both_durations)
nox_df = nox_df[rows_to_keep].copy()
nox_df = nox_df.drop(columns=["duration_deviation_pct"])

n_rows_after = len(nox_df)
n_machines_after = nox_df["MachineId"].nunique()
print(f"✓ Duration consistency (±{DURATION_CONSISTENCY_PCT}%): {n_rows_after:,} rows (lost {n_rows_before - n_rows_after:,} machine-days, {n_machines_before - n_machines_after} machines, {n_machines_no_ff_duration} without FF duration)")

# Filter 8: Date range (optional)
if MIN_DATEKEY is not None or MAX_DATEKEY is not None:
    n_rows_before = len(nox_df)
    n_machines_before = nox_df["MachineId"].nunique()

    if MIN_DATEKEY is not None and MAX_DATEKEY is not None:
        nox_df = nox_df[(nox_df["datekey"] >= MIN_DATEKEY) & (nox_df["datekey"] <= MAX_DATEKEY)].copy()
        date_range_str = f"{MIN_DATEKEY} to {MAX_DATEKEY}"
    elif MIN_DATEKEY is not None:
        nox_df = nox_df[nox_df["datekey"] >= MIN_DATEKEY].copy()
        date_range_str = f">= {MIN_DATEKEY}"
    else:
        nox_df = nox_df[nox_df["datekey"] <= MAX_DATEKEY].copy()
        date_range_str = f"<= {MAX_DATEKEY}"

    n_rows_after = len(nox_df)
    n_machines_after = nox_df["MachineId"].nunique()
    print(f"✓ Date range ({date_range_str}): {n_rows_after:,} rows (lost {n_rows_before - n_rows_after:,} machine-days, {n_machines_before - n_machines_after} machines)")

n_final_machines = nox_df["MachineId"].nunique()
print(f"\n{'=' * 80}")
print(f"FINAL DATASET: {n_final_machines} machines, {len(nox_df):,} machine-days")
print(f"{'=' * 80}\n")


# %% ============================================================================
# SECTION 1: MACHINE-LEVEL OVERVIEW
# ==============================================================================
# One point per machine showing average motorbelasting and average NOx per liter
# Useful for: Fleet overview, identifying outlier machines, comparing stage groups

print("\n" + "=" * 80)
print("SECTION 1: MACHINE-LEVEL OVERVIEW")
print("=" * 80)

# Prepare data for machine-level scatter plot
# Filter for clean data and specific machine groups
scatter_exclusions = ['', '+', 'nan+nan']
machine_scatter_df = nox_df[
    (~nox_df["Stage+Groep"].isin(scatter_exclusions))
    # & (nox_df["duration_from_rows"] >= 0.5)
    # & (nox_df["FuelMassFlow"] >= 0.5)
    # & (nox_df["Machinegroep"].isin(["C", "D"]))
].copy()

machine_scatter_df = machine_scatter_df.dropna(subset=["NOxMAF_motorbelasting", "NOxMAF_NOxPerLiter", "Stage+Groep"])

# Aggregate to machine level
machine_avg_df = machine_scatter_df.groupby(["MachineId", "Stage+Groep"]).agg({
    "NOxMAF_motorbelasting": "mean",
    "FF_motorbelasting": "mean",
    "NOxMAF_NOxPerLiter": "mean",
    "MainGroupLabel": "first",
    "MerkType": "first",
    "SensorSupplier": "first",
    "Pilot": "first"
}).reset_index()

machine_avg_df.columns = ["MachineId", "Stage+Groep", "avg_motorbelasting", "avg_motorbelasting_ff", "avg_nox_per_liter",
                           "MainGroupLabel", "MerkType", "SensorSupplier", "Pilot"]

# Fill missing SensorSupplier and Pilot values
machine_avg_df["SensorSupplier"] = machine_avg_df["SensorSupplier"].fillna("Unknown")
machine_avg_df["Pilot"] = machine_avg_df["Pilot"].fillna("Unknown")

print(f"Machine-level data: {len(machine_avg_df)} machines")
print(f"\nMachines per Stage+Groep:")
for stage_group in sorted(machine_avg_df["Stage+Groep"].unique()):
    count = len(machine_avg_df[machine_avg_df["Stage+Groep"] == stage_group])
    print(f"  {stage_group}: {count} machines")

# Create machine-level scatter plot
unique_stage_groups = sorted(machine_avg_df["Stage+Groep"].unique())

# Color palette - use standardised colors from helper module
default_colors = px.colors.qualitative.Plotly
palette = {}
for i, stage_group in enumerate(unique_stage_groups):
    palette[stage_group] = STAGE_GROUP_COLORS.get(stage_group, default_colors[i % len(default_colors)])

fig_machine = go.Figure()

# Marker shapes for different sensor suppliers
sensor_symbols = {
    "FleetsonlineV2": "circle",
    "GPS Buddy": "square"
}
unique_sensors_s1 = sorted(machine_avg_df["SensorSupplier"].unique())
unique_pilots_s1 = sorted(machine_avg_df["Pilot"].unique())

# Track which stage_groups and sensors we've added to legend
stage_group_in_legend = set()
sensor_in_legend = set()

# Track trace indices by sensor and pilot for interactive filtering
trace_sensors_s1 = []  # List of sensor names for each trace in fig_machine
trace_pilots_s1 = []  # List of pilot names for each trace in fig_machine

# Add scatter points for each Stage+Groep, SensorSupplier, and Pilot combination
for stage_group in unique_stage_groups:
    for sensor in unique_sensors_s1:
        for pilot in unique_pilots_s1:
            group_data = machine_avg_df[
                (machine_avg_df["Stage+Groep"] == stage_group) &
                (machine_avg_df["SensorSupplier"] == sensor) &
                (machine_avg_df["Pilot"] == pilot)
            ]

            if len(group_data) == 0:
                continue

            # Get marker symbol for this sensor
            symbol = sensor_symbols.get(sensor, "circle")

            # Determine legend visibility: show once per stage_group
            show_in_legend = stage_group not in stage_group_in_legend
            if show_in_legend:
                stage_group_in_legend.add(stage_group)

            fig_machine.add_trace(go.Scatter(
                x=group_data["avg_motorbelasting"],
                y=group_data["avg_nox_per_liter"],
                mode='markers+text',
                name=stage_group,
                text=group_data["MachineId"],
                textposition="top center",
                textfont=dict(size=9),
                marker=dict(
                    color=palette[stage_group],
                    size=10,
                    opacity=0.7,
                    symbol=symbol,
                    line=dict(width=1, color='white')
                ),
                legendgroup=stage_group,
                showlegend=show_in_legend,
                hovertemplate=(
                    "<b>Machine %{text}</b><br>" +
                    "Stage+Groep: " + stage_group + "<br>" +
                    "Sensor: " + sensor + "<br>" +
                    "Pilot: " + pilot + "<br>" +
                    "Avg Motorbelasting: %{x:.2%}<br>" +
                    "Avg NOx/L: %{y:.2f} g/L<br>" +
                    "<extra></extra>"
                )
            ))
            trace_sensors_s1.append(sensor)
            trace_pilots_s1.append(pilot)

# Add dummy traces for sensor supplier shape legend (in a separate legend group)
for sensor in unique_sensors_s1:
    symbol = sensor_symbols.get(sensor, "circle")
    fig_machine.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        name=f"Sensor: {sensor}",
        marker=dict(
            color='gray',
            size=10,
            symbol=symbol,
            line=dict(width=1, color='white')
        ),
        legendgroup="sensors",
        showlegend=True
    ))
    trace_sensors_s1.append("_legend_")  # Mark as legend trace (always visible)
    trace_pilots_s1.append("_legend_")  # Mark as legend trace (always visible)

# Add certification limits and reference curves
for stage_group in unique_stage_groups:
    parts = [part.strip() for part in stage_group.split("+")]
    if len(parts) != 2:
        continue
    stage_part, machine_group = parts

    # Add certification limit line
    try:
        limit_value = certification_limit(stage_part, machine_group=machine_group)
        fig_machine.add_trace(go.Scatter(
            x=[X_MIN, 1.0],
            y=[limit_value, limit_value],
            mode='lines',
            name=f"{stage_group} limit",
            line=dict(color=palette[stage_group], width=2, dash='dash'),
            legendgroup=stage_group,
            showlegend=True
        ))
        trace_sensors_s1.append("_reference_")  # Reference lines always visible
        trace_pilots_s1.append("_reference_")  # Reference lines always visible
    except ValueError:
        pass

    # Add AdBlue reference curves (multiple percentages per group)
    adblue_percentages = get_adblue_percentages(machine_group)
    if not adblue_percentages:
        continue

    for adblue_pct in adblue_percentages:
        try:
            model_func = nox_per_liter_polynomial(stage_part, machine_group, adblue_pct=adblue_pct)
            x_grid = np.linspace(0.03, 1.0, 200)
            y_grid = [model_func(x) * 1000 for x in x_grid]  # Convert kg/L to g/L

            # Label: "no AdBlue" for 0%, otherwise show percentage
            adblue_label = "no AdBlue" if adblue_pct == 0 else f"{adblue_pct:.0f}% AdBlue"
            fig_machine.add_trace(go.Scatter(
                x=x_grid,
                y=y_grid,
                mode='lines',
                name=f"{stage_group} model ({adblue_label})",
                line=dict(color=palette[stage_group], width=1, dash='dot'),
                legendgroup=stage_group,
                showlegend=True
            ))
            trace_sensors_s1.append("_reference_")  # Reference lines always visible
            trace_pilots_s1.append("_reference_")  # Reference lines always visible
        except Exception as e:
            print(f"Warning: Could not add AdBlue model for {stage_group} at {adblue_pct}%: {str(e)}")

# Create dropdown buttons for filtering (Section 1)
def create_filter_buttons_s1(trace_values, all_values, label_prefix="All"):
    """Create visibility lists for filter dropdown."""
    buttons = []

    # "All" button - show everything
    buttons.append(dict(
        label=f"All {label_prefix}",
        method="update",
        args=[{"visible": [True] * len(trace_values)}]
    ))

    # Individual value buttons
    for value in all_values:
        visibility = [
            (v == value or v in ["_legend_", "_reference_"])
            for v in trace_values
        ]
        buttons.append(dict(
            label=value,
            method="update",
            args=[{"visible": visibility}]
        ))

    return buttons

sensor_buttons_s1 = create_filter_buttons_s1(trace_sensors_s1, unique_sensors_s1, "Sensors")
pilot_buttons_s1 = create_filter_buttons_s1(trace_pilots_s1, unique_pilots_s1, "Pilots")

# Update layout
fig_machine.update_layout(
    title=dict(
        text="<b>Machine-Level Overview</b><br><sub>Average Motorbelasting vs NOx per Liter</sub>",
        font=dict(size=18)
    ),
    xaxis=dict(
        title="Average Motorbelasting",
        tickformat=".0%",
        range=[0, 0.7],
        gridcolor='lightgray'
    ),
    yaxis=dict(
        title="Average NOx per Liter (g/L)",
        range=[0, 30],
        gridcolor='lightgray'
    ),
    hovermode='closest',
    plot_bgcolor='white',
    legend=dict(
        x=1.02,
        y=1,
        xanchor='left',
        yanchor='top',
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='gray',
        borderwidth=1
    ),
    updatemenus=[
        dict(
            buttons=sensor_buttons_s1,
            direction="down",
            showactive=True,
            x=0.70,
            xanchor="left",
            y=1.15,
            yanchor="top",
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=11)
        ),
        dict(
            buttons=pilot_buttons_s1,
            direction="down",
            showactive=True,
            x=0.85,
            xanchor="left",
            y=1.15,
            yanchor="top",
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=11)
        )
    ],
    height=700,
    width=1200
)

fig_machine.show()

print("\n✓ Machine-level overview plot created")
print(f"  • {len(machine_avg_df)} machines plotted")
print(f"  • Certification limits shown as dashed lines")
print(f"  • AdBlue reference models shown as dotted lines\n")


# %% ============================================================================
# SECTION 2: MACHINE-DAY LEVEL ANALYSIS WITH FITTED CURVES
# ==============================================================================
# All daily records with fitted curves per Stage+Groep
# Useful for: Understanding daily variation, validating models, seeing data density

print("\n" + "=" * 80)
print("SECTION 2: MACHINE-DAY LEVEL ANALYSIS")
print("=" * 80)

# Prepare data for machine-day scatter plot
scatter_df = nox_df[
    (~nox_df["Stage+Groep"].isin(scatter_exclusions))
    # & (nox_df["duration_from_rows"] >= 0.5)
    # & (nox_df["FuelMassFlow"] >= 0.5)
    # & (nox_df["Machinegroep"].isin(["C", "D"]))
].copy()

scatter_df = scatter_df.dropna(subset=["NOxMAF_motorbelasting", "NOxMAF_NOxPerLiter", "Stage+Groep"])

print(f"Machine-day data: {len(scatter_df):,} daily records")
print(f"\nRecords per Stage+Groep:")
for stage_group in sorted(scatter_df["Stage+Groep"].unique()):
    count = len(scatter_df[scatter_df["Stage+Groep"] == stage_group])
    print(f"  {stage_group}: {count:,} days")

# Create figures
fig_daily_stage = go.Figure()
fig_daily_machine = go.Figure()

# Fill missing SensorSupplier and Pilot values in scatter_df
scatter_df["SensorSupplier"] = scatter_df["SensorSupplier"].fillna("Unknown")
scatter_df["Pilot"] = scatter_df["Pilot"].fillna("Unknown")
unique_sensors_s2 = sorted(scatter_df["SensorSupplier"].unique())
unique_pilots_s2 = sorted(scatter_df["Pilot"].unique())

# Track which stage_groups have been added to legend
stage_group_in_legend_s2 = set()

# Track trace indices by sensor and pilot for interactive filtering
trace_sensors_stage = []  # List of sensor names for each trace in fig_daily_stage
trace_sensors_machine = []  # List of sensor names for each trace in fig_daily_machine
trace_pilots_stage = []  # List of pilot names for each trace in fig_daily_stage
trace_pilots_machine = []  # List of pilot names for each trace in fig_daily_machine

# Add scatter points for each Stage+Groep, SensorSupplier, and Pilot combination
for stage_group in unique_stage_groups:
    for sensor in unique_sensors_s2:
        for pilot in unique_pilots_s2:
            group_data = scatter_df[
                (scatter_df["Stage+Groep"] == stage_group) &
                (scatter_df["SensorSupplier"] == sensor) &
                (scatter_df["Pilot"] == pilot)
            ]

            if len(group_data) == 0:
                continue

            # Get marker symbol for this sensor
            symbol = sensor_symbols.get(sensor, "circle")

            # Determine legend visibility: show once per stage_group
            show_in_legend = stage_group not in stage_group_in_legend_s2
            if show_in_legend:
                stage_group_in_legend_s2.add(stage_group)

            fig_daily_stage.add_trace(go.Scatter(
                x=group_data["NOxMAF_motorbelasting"],
                y=group_data["NOxMAF_NOxPerLiter"],
                mode='markers',
                name=stage_group,
                text=group_data["MachineId"],
                marker=dict(
                    color=palette[stage_group],
                    size=5,
                    opacity=0.5,
                    symbol=symbol,
                    line=dict(width=0)
                ),
                legendgroup=stage_group,
                showlegend=show_in_legend,
                hovertemplate=(
                    "<b>Machine %{text}</b><br>" +
                    stage_group + "<br>" +
                    "Sensor: " + sensor + "<br>" +
                    "Pilot: " + pilot + "<br>" +
                    "Motorbelasting: %{x:.2%}<br>" +
                    "NOx/L: %{y:.2f} g/L<br>" +
                    "<extra></extra>"
                )
            ))
            trace_sensors_stage.append(sensor)
            trace_pilots_stage.append(pilot)

# Add dummy traces for sensor supplier shape legend
for sensor in unique_sensors_s2:
    symbol = sensor_symbols.get(sensor, "circle")
    fig_daily_stage.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        name=f"Sensor: {sensor}",
        marker=dict(
            color='gray',
            size=8,
            symbol=symbol,
            line=dict(width=1, color='white')
        ),
        legendgroup="sensors",
        showlegend=True
    ))
    trace_sensors_stage.append("_legend_")  # Mark as legend trace (always visible)
    trace_pilots_stage.append("_legend_")  # Mark as legend trace (always visible)

# Add scatter points per machine (colors + legend for interaction)
unique_machines = sorted(scatter_df["MachineId"].unique())
machine_colors = px.colors.qualitative.Bold
machine_palette = {}
for i, machine_id in enumerate(unique_machines):
    machine_palette[machine_id] = machine_colors[i % len(machine_colors)]

# Track which machines have been added to legend
machine_in_legend = set()

for machine_id in unique_machines:
    for sensor in unique_sensors_s2:
        for pilot in unique_pilots_s2:
            group_data = scatter_df[
                (scatter_df["MachineId"] == machine_id) &
                (scatter_df["SensorSupplier"] == sensor) &
                (scatter_df["Pilot"] == pilot)
            ]

            if len(group_data) == 0:
                continue

            stage_group = group_data["Stage+Groep"].iloc[0]

            # Get marker symbol for this sensor
            symbol = sensor_symbols.get(sensor, "circle")

            # Determine legend visibility: show once per machine
            show_in_legend = machine_id not in machine_in_legend
            if show_in_legend:
                machine_in_legend.add(machine_id)

            fig_daily_machine.add_trace(go.Scatter(
                x=group_data["NOxMAF_motorbelasting"],
                y=group_data["NOxMAF_NOxPerLiter"],
                mode='markers',
                name=f"Machine {machine_id} ({stage_group})",
                marker=dict(
                    color=machine_palette[machine_id],
                    size=6,
                    opacity=0.7,
                    symbol=symbol,
                    line=dict(width=0)
                ),
                legendgroup=str(machine_id),
                showlegend=show_in_legend,
                customdata=group_data["Stage+Groep"],
                hovertemplate=(
                    "Machine: %{text}<br>" +
                    "Stage+Groep: %{customdata}<br>" +
                    "Sensor: " + sensor + "<br>" +
                    "Pilot: " + pilot + "<br>" +
                    "Motorbelasting: %{x:.2%}<br>" +
                    "NOx/L: %{y:.2f} g/L<br>" +
                    "<extra></extra>"
                ),
                text=group_data["MachineId"]
            ))
            trace_sensors_machine.append(sensor)
            trace_pilots_machine.append(pilot)

# Add dummy traces for sensor supplier shape legend (fig_daily_machine)
for sensor in unique_sensors_s2:
    symbol = sensor_symbols.get(sensor, "circle")
    fig_daily_machine.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        name=f"Sensor: {sensor}",
        marker=dict(
            color='gray',
            size=8,
            symbol=symbol,
            line=dict(width=1, color='white')
        ),
        legendgroup="sensors",
        showlegend=True
    ))
    trace_sensors_machine.append("_legend_")  # Mark as legend trace (always visible)
    trace_pilots_machine.append("_legend_")  # Mark as legend trace (always visible)

# Calculate grid for model curves
x_max = scatter_df["NOxMAF_motorbelasting"].max()
grid_min = 0.03
grid_max = max(grid_min + 1e-3, x_max)
motorbelasting_grid = np.linspace(grid_min, grid_max, 250)

# Track annotations to prevent overlap
annotations_data = []

# Add certification limits, reference models, and fitted curves
for stage_group in unique_stage_groups:
    parts = [part.strip() for part in stage_group.split("+")]
    if len(parts) != 2:
        continue
    stage_part, machine_group = parts

    # 1. Add certification limit line
    try:
        limit_value = certification_limit(stage_part, machine_group=machine_group)
        fig_daily_stage.add_trace(go.Scatter(
            x=[X_MIN, x_max],
            y=[limit_value, limit_value],
            mode='lines',
            name=f"{stage_group} limiet",
            line=dict(color=palette[stage_group], width=1, dash='dash'),
            legendgroup=stage_group,
            showlegend=True
        ))
        trace_sensors_stage.append("_reference_")  # Reference lines always visible
        trace_pilots_stage.append("_reference_")  # Reference lines always visible
        fig_daily_machine.add_trace(go.Scatter(
            x=[X_MIN, x_max],
            y=[limit_value, limit_value],
            mode='lines',
            name=f"{stage_group} limiet",
            line=dict(color=palette[stage_group], width=1, dash='dash'),
            legendgroup=stage_group,
            showlegend=True
        ))
        trace_sensors_machine.append("_reference_")  # Reference lines always visible
        trace_pilots_machine.append("_reference_")  # Reference lines always visible
    except ValueError:
        limit_value = None

    # 2. Add AdBlue reference models
    group_key = machine_group.strip().upper()
    adblue_samples = get_adblue_percentages(group_key)

    for adblue_pct in adblue_samples:
        model_func = nox_per_liter_polynomial(stage_part, machine_group, adblue_pct=adblue_pct)
        y_model = [model_func(x) * 1000 for x in motorbelasting_grid]  # Convert kg/L to g/L

        fig_daily_stage.add_trace(go.Scatter(
            x=motorbelasting_grid,
            y=y_model,
            mode='lines',
            name=f"{stage_group} | AdBlue {adblue_pct:.0f}%",
            line=dict(color=palette[stage_group], width=1, dash='dot'),
            legendgroup=stage_group,
            showlegend=True
        ))
        trace_sensors_stage.append("_reference_")  # Reference lines always visible
        trace_pilots_stage.append("_reference_")  # Reference lines always visible
        fig_daily_machine.add_trace(go.Scatter(
            x=motorbelasting_grid,
            y=y_model,
            mode='lines',
            name=f"{stage_group} | AdBlue {adblue_pct:.0f}%",
            line=dict(color=palette[stage_group], width=1, dash='dot'),
            legendgroup=stage_group,
            showlegend=True
        ))
        trace_sensors_machine.append("_reference_")  # Reference lines always visible
        trace_pilots_machine.append("_reference_")  # Reference lines always visible

    # 3. Fit exponential decay curve to actual data
    group_data = scatter_df[scatter_df["Stage+Groep"] == stage_group]

    # Filter for motorbelasting >= lower bound
    group_data = group_data[group_data["NOxMAF_motorbelasting"] >= MIN_MOTORBELASTING_FOR_FIT]

    if len(group_data) < 10:  # Need minimum data points for fitting
        continue

    x_data = group_data["NOxMAF_motorbelasting"].values
    y_data = group_data["NOxMAF_NOxPerLiter"].values

    try:
        # Estimate initial parameters from data
        y_high = y_data[x_data > 0.4].mean() if (x_data > 0.4).any() else y_data.mean()
        y_max = y_data.max()

        # Initial guesses: a=asymptote, b=extra at x=0, c=decay rate
        p0 = [max(0.1, y_high), max(0.1, y_max - y_high), 5.0]

        # Parameter bounds: all must be positive
        bounds = ([0.01, 0.01, 0.1], [50, 100, 50])

        # Fit the exponential decay model: y = a + b * exp(-c * x)
        popt, _ = curve_fit(
            model_fit_function,
            x_data,
            y_data,
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )

        a, b, c = popt

        # Generate fitted curve (starting from lower bound)
        fit_grid = motorbelasting_grid[motorbelasting_grid >= MIN_MOTORBELASTING_FOR_FIT]
        y_fit = model_fit_function(fit_grid, a, b, c)

        fig_daily_stage.add_trace(go.Scatter(
            x=fit_grid,
            y=y_fit,
            mode='lines',
            name=f"{stage_group} fit",
            line=dict(color=palette[stage_group], width=2),
            legendgroup=stage_group,
            showlegend=True
        ))
        trace_sensors_stage.append("_reference_")  # Fitted curves always visible
        trace_pilots_stage.append("_reference_")  # Fitted curves always visible

        # Add annotation with fit parameters
        # Position at right side of plot
        x_pos = x_max
        y_pos = model_fit_function(x_pos, a, b, c)

        annotation_text = (
            f"<b>{stage_group}</b><br>"
            f"y = {a:.2f} + {b:.2f}·e<sup>-{c:.2f}x</sup>"
        )

        annotations_data.append({
            "x": x_pos,
            "y": y_pos,
            "text": annotation_text,
            "stage_group": stage_group
        })

        print(f"\n✓ Fitted curve for {stage_group}:")
        print(f"  y = {a:.2f} + {b:.2f} * exp(-{c:.2f} * x)")
        print(f"  Asymptote: {a:.2f} g/L, Max at x=0: {a+b:.2f} g/L")
        print(f"  n = {len(group_data):,} days (motorbelasting >= 5%)")

    except Exception as e:
        print(f"\n✗ Could not fit curve for {stage_group}: {str(e)}")

# Add annotations (with simple vertical spacing to avoid overlap)
annotations = []
y_offset = 0.3
for i, ann_data in enumerate(annotations_data):
    annotations.append(dict(
        x=ann_data["x"],
        y=ann_data["y"] + i * y_offset,
        text=ann_data["text"],
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor=palette[ann_data["stage_group"]],
        ax=40,
        ay=0,
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor=palette[ann_data["stage_group"]],
        borderwidth=1,
        font=dict(size=9)
    ))

# Create dropdown buttons for filtering
def create_filter_buttons(trace_values, all_values, label_prefix="All"):
    """Create visibility lists for filter dropdown."""
    buttons = []

    # "All" button - show everything
    buttons.append(dict(
        label=f"All {label_prefix}",
        method="update",
        args=[{"visible": [True] * len(trace_values)}]
    ))

    # Individual value buttons
    for value in all_values:
        visibility = [
            (v == value or v in ["_legend_", "_reference_"])
            for v in trace_values
        ]
        buttons.append(dict(
            label=value,
            method="update",
            args=[{"visible": visibility}]
        ))

    return buttons

sensor_buttons_stage = create_filter_buttons(trace_sensors_stage, unique_sensors_s2, "Sensors")
sensor_buttons_machine = create_filter_buttons(trace_sensors_machine, unique_sensors_s2, "Sensors")
pilot_buttons_stage = create_filter_buttons(trace_pilots_stage, unique_pilots_s2, "Pilots")
pilot_buttons_machine = create_filter_buttons(trace_pilots_machine, unique_pilots_s2, "Pilots")

# Update layout
fig_daily_stage.update_layout(
    title=dict(
        text="<b>Machine-Day Level Analysis (Stage+Groep)</b><br><sub>Daily Motorbelasting vs NOx per Liter with Fitted Curves</sub>",
        font=dict(size=18)
    ),
    xaxis=dict(
        title="Motorbelasting (daily)",
        tickformat=".0%",
        range=[0, x_max * 1.05],
        gridcolor='lightgray'
    ),
    yaxis=dict(
        title="NOx per Liter (g/L)",
        range=[0, 30],
        gridcolor='lightgray'
    ),
    annotations=annotations,
    hovermode='closest',
    plot_bgcolor='white',
    legend=dict(
        x=1.02,
        y=1,
        xanchor='left',
        yanchor='top',
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='gray',
        borderwidth=1
    ),
    updatemenus=[
        dict(
            buttons=sensor_buttons_stage,
            direction="down",
            showactive=True,
            x=0.70,
            xanchor="left",
            y=1.15,
            yanchor="top",
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=11)
        ),
        dict(
            buttons=pilot_buttons_stage,
            direction="down",
            showactive=True,
            x=0.85,
            xanchor="left",
            y=1.15,
            yanchor="top",
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=11)
        )
    ],
    height=700,
    width=1400
)

fig_daily_machine.update_layout(
    title=dict(
        text="<b>Machine-Day Level Analysis (Machine)</b><br><sub>Daily Motorbelasting vs NOx per Liter</sub>",
        font=dict(size=18)
    ),
    xaxis=dict(
        title="Motorbelasting (daily)",
        tickformat=".0%",
        range=[0, x_max * 1.05],
        gridcolor='lightgray'
    ),
    yaxis=dict(
        title="NOx per Liter (g/L)",
        range=[0, 30],
        gridcolor='lightgray'
    ),
    annotations=annotations,
    hovermode='closest',
    plot_bgcolor='white',
    legend=dict(
        x=1.02,
        y=1,
        xanchor='left',
        yanchor='top',
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='gray',
        borderwidth=1
    ),
    updatemenus=[
        dict(
            buttons=sensor_buttons_machine,
            direction="down",
            showactive=True,
            x=0.70,
            xanchor="left",
            y=1.15,
            yanchor="top",
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=11)
        ),
        dict(
            buttons=pilot_buttons_machine,
            direction="down",
            showactive=True,
            x=0.85,
            xanchor="left",
            y=1.15,
            yanchor="top",
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=11)
        )
    ],
    height=700,
    width=1400
)

fig_daily_stage.show()
fig_daily_machine.show()

print(f"\n✓ Machine-day analysis plots created")
print(f"  • {len(scatter_df):,} daily records plotted")
print(f"  • Stage+Groep view includes fitted curves and reference models")
print(f"  • Machine view shows per-machine scatter points\n")


# %% ============================================================================
# SECTION 3: SINGLE-MACHINE DEEP DIVE
# ==============================================================================
# Detailed analysis for a specific machine including temporal patterns
# Useful for: Diagnosing issues, understanding machine behavior, validating sensors

print("\n" + "=" * 80)
print("SECTION 3: SINGLE-MACHINE DEEP DIVE")
print("=" * 80)

# SELECT MACHINE TO ANALYZE
MACHINE_ID = 2485  # Change this to analyze different machines

print(f"\nAnalyzing Machine ID: {MACHINE_ID}\n")

# Get machine data
machine_data = nox_df[nox_df["MachineId"] == MACHINE_ID].copy()

if machine_data.empty:
    print(f"ERROR: No data found for Machine {MACHINE_ID}")
else:
    # Sort by date
    machine_data = machine_data.sort_values("datekey")

    # Convert datekey to proper datetime format
    machine_data["date"] = pd.to_datetime(machine_data["datekey"].astype(str), format='%Y%m%d')

    # Get machine specs
    specs = machine_data.iloc[0]

    print("MACHINE SPECIFICATIONS")
    print("-" * 80)
    print(f"Machine ID:        {specs['MachineId']}")

    # Handle potentially missing columns
    brand_type = specs.get('MerkType', 'N/A')
    if pd.isna(brand_type) or brand_type == 'Unknown Unknown':
        brand = specs.get('BrandLabel', 'N/A')
        equipment_type = specs.get('TypeOfEquipment', 'N/A')
        brand_type = f"{brand} {equipment_type}" if brand != 'N/A' and equipment_type != 'N/A' else 'N/A'

    print(f"Brand/Type:        {brand_type}")
    print(f"Main Group:        {specs.get('MainGroupLabel', 'N/A')}")
    print(f"Stage+Group:       {specs.get('Stage+Groep', 'N/A')}")
    print(f"Engine Power:      {specs.get('Power', 'N/A')} kW")
    print(f"Construction Year: {specs.get('ConstructionYear', 'N/A')}")
    print(f"Machine Group:     {specs.get('Machinegroep', 'N/A')}")

    print("\nOPERATIONAL SUMMARY")
    print("-" * 80)
    print(f"Total days:        {len(machine_data)} days")
    print(f"Date range:        {machine_data['datekey'].min()} to {machine_data['datekey'].max()}")
    print(f"Avg hours/day:     {machine_data['duration_from_rows'].mean():.2f} hrs")
    print(f"Avg fuel/day:      {machine_data['FuelMassFlow'].mean():.2f} L")
    print(f"Avg motorbelasting: {machine_data['NOxMAF_motorbelasting'].mean():.2%}")
    print(f"Idle time (<10%):  {(machine_data['NOxMAF_motorbelasting'] < 0.10).sum() / len(machine_data):.1%} of days")

    print("\nNOx EMISSIONS")
    print("-" * 80)
    print(f"Avg NOx/L:         {machine_data['NOxMAF_NOxPerLiter'].mean():.2f} g/L")
    print(f"Avg NOx/hr:        {machine_data['NOxMAF_FPH'].mean():.2f} g/hr")
    print(f"Total NOx:         {machine_data['NOxTotal'].sum():.0f} g")

    # Compare to certification limit
    stage_group = specs.get('Stage+Groep', '')
    limit = certification_limit_for_stage_group(stage_group)
    if limit:
        avg_nox = machine_data['NOxMAF_NOxPerLiter'].mean()
        pct_of_limit = (avg_nox / limit) * 100
        print(f"Certification limit: {limit:.2f} g/L")
        print(f"% of limit:        {pct_of_limit:.1f}%")
        if pct_of_limit > 100:
            print(f"  ⚠️  EXCEEDS LIMIT by {pct_of_limit - 100:.1f}%")

    # -------------------------------------------------------------------------
    # 3A: Timeline Plot - NOx and Load over Time
    # -------------------------------------------------------------------------

    fig_timeline = go.Figure()

    # Add NOx per liter over time
    fig_timeline.add_trace(go.Scatter(
        x=machine_data["date"],
        y=machine_data["NOxMAF_NOxPerLiter"],
        mode='lines+markers',
        name='NOx per Liter',
        line=dict(color='#e74c3c', width=2),
        marker=dict(size=6),
        yaxis='y1'
    ))

    # Add motorbelasting over time
    fig_timeline.add_trace(go.Scatter(
        x=machine_data["date"],
        y=machine_data["NOxMAF_motorbelasting"],
        mode='lines+markers',
        name='Motorbelasting',
        line=dict(color='#3498db', width=2),
        marker=dict(size=6),
        yaxis='y2'
    ))

    # Add certification limit line if available
    if limit:
        fig_timeline.add_trace(go.Scatter(
            x=machine_data["date"],
            y=[limit] * len(machine_data),
            mode='lines',
            name='Certification Limit',
            line=dict(color='red', width=1, dash='dash'),
            yaxis='y1'
        ))

    fig_timeline.update_layout(
        title=f"<b>Machine {MACHINE_ID} - Timeline Analysis</b>",
        xaxis=dict(title="Date"),
        yaxis=dict(
            title="NOx per Liter (g/L)",
            side='left',
            color='#e74c3c'
        ),
        yaxis2=dict(
            title="Motorbelasting",
            side='right',
            overlaying='y',
            tickformat='.0%',
            color='#3498db'
        ),
        hovermode='x unified',
        plot_bgcolor='white',
        legend=dict(x=0, y=1.1, orientation='h'),
        height=500,
        width=1400
    )

    fig_timeline.show()
    print("\n✓ Timeline plot created")

    # -------------------------------------------------------------------------
    # 3B: Load vs NOx Scatter
    # -------------------------------------------------------------------------

    fig_load_nox = go.Figure()

    # Scatter plot
    fig_load_nox.add_trace(go.Scatter(
        x=machine_data["NOxMAF_motorbelasting"],
        y=machine_data["NOxMAF_NOxPerLiter"],
        mode='markers',
        name='Daily records',
        marker=dict(
            color='#3498db',
            size=8,
            opacity=0.6
        ),
        hovertemplate=(
            "Date: %{text}<br>" +
            "Motorbelasting: %{x:.2%}<br>" +
            "NOx/L: %{y:.2f} g/L<br>" +
            "<extra></extra>"
        ),
        text=machine_data["date"].dt.strftime('%Y-%m-%d')
    ))

    # Add certification limit
    if limit:
        fig_load_nox.add_trace(go.Scatter(
            x=[0, 0.7],
            y=[limit, limit],
            mode='lines',
            name='Certification Limit',
            line=dict(color='red', width=2, dash='dash')
        ))

    # Add AdBlue reference curves using TNO AUB formula
    # Formula: NOx [kg] = Qb * liter_brandstof + Qu * draaiuren + Qa * liter_AdBlue
    # Source: TNO 2021 R12305, Section 2.4 and Table 3
    stage_group = specs.get('Stage+Groep', '')
    machine_group = stage_group.split('+')[1].strip() if '+' in stage_group else None

    coeffs = get_tno_coefficients(machine_group)
    if machine_group in ["C", "D"] and coeffs:
        # Get average fuel consumption rate for this machine (for context)
        avg_fph = machine_data['NOxMAF_FPH'].mean()  # L/h
        power_kw = specs.get('Power', np.nan)

        # Determine AdBlue percentages based on machine group
        adblue_percentages = get_adblue_percentages(machine_group)

        print(f"  • Average fuel consumption: {avg_fph:.2f} L/h")
        print(f"  • Engine power: {power_kw if not pd.isna(power_kw) else 'N/A'} kW")
        print(f"  • TNO coefficients (Group {machine_group}): Qb={coeffs['Qb']:.3f}, Qu={coeffs['Qu']:.3f}, Qa={coeffs['Qa']:.2f}")

        if pd.isna(power_kw) or power_kw <= 0:
            print("  • Missing engine power; skipping load-based TNO AUB curves")
        else:
            load_values = np.linspace(X_MIN, 0.7, 60)
            for adblue_pct in adblue_percentages:
                # TNO AUB formula for 1 hour:
                # NOx [kg] = Qb * liter_brandstof + Qu * uren + Qa * liter_AdBlue
                fuel_per_hour = load_values * (power_kw / 4.0)
                uren = 1.0  # hour
                liter_adblue = fuel_per_hour * (adblue_pct / 100)
                nox_kg = (coeffs['Qb'] * fuel_per_hour) + (coeffs['Qu'] * uren) + (coeffs['Qa'] * liter_adblue)
                nox_per_liter = (nox_kg / fuel_per_hour) * 1000

                fig_load_nox.add_trace(go.Scatter(
                    x=load_values,
                    y=nox_per_liter,
                    mode='lines',
                    name=f'{adblue_pct:.0f}% AdBlue (TNO AUB)',
                    line=dict(width=1.5, dash='dot'),
                    showlegend=True,
                    hovertemplate=(
                        "Motorbelasting: %{x:.2%}<br>"
                        f"TNO AUB with {adblue_pct:.0f}% AdBlue: %{{y:.2f}} g/L<extra></extra>"
                    )
                ))

        print(f"  • TNO AUB reference lines added")

    fig_load_nox.update_layout(
        title=f"<b>Machine {MACHINE_ID} - Load vs NOx</b>",
        xaxis=dict(
            title="Motorbelasting",
            tickformat='.0%',
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title="NOx per Liter (g/L)",
            gridcolor='lightgray'
        ),
        plot_bgcolor='white',
        legend=dict(x=0.7, y=0.95),
        height=600,
        width=900
    )

    fig_load_nox.show()
    print("✓ Load vs NOx scatter plot created")

    # -------------------------------------------------------------------------
    # 3C: Load Distribution Table
    # -------------------------------------------------------------------------

    print("\nLOAD DISTRIBUTION")
    print("-" * 80)

    load_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    load_labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-100%']
    machine_data['load_bin'] = pd.cut(machine_data['NOxMAF_motorbelasting'], bins=load_bins, labels=load_labels)

    load_dist = machine_data['load_bin'].value_counts(normalize=True).sort_index() * 100

    for load_range, pct in load_dist.items():
        bar = '█' * int(pct / 2)
        print(f"{load_range:>10}: {bar:<50} {pct:5.1f}%")

    # -------------------------------------------------------------------------
    # 3D: Fuel Source Comparison
    # -------------------------------------------------------------------------

    print("\nFUEL CONSUMPTION COMPARISON")
    print("-" * 80)

    if 'CANBUS_FPH' in machine_data.columns and 'FF_FPH' in machine_data.columns:
        canbus_avg = machine_data['CANBUS_FPH'].mean()
        ff_avg = machine_data['FF_FPH'].mean()
        noxmaf_avg = machine_data['NOxMAF_FPH'].mean()

        print(f"CANBUS_FPH avg:  {canbus_avg:.2f} L/hr")
        print(f"FF_FPH avg:      {ff_avg:.2f} L/hr")
        print(f"NOxMAF_FPH avg:  {noxmaf_avg:.2f} L/hr")

        if not pd.isna(canbus_avg) and not pd.isna(ff_avg):
            diff_pct = abs((ff_avg - canbus_avg) / canbus_avg) * 100
            print(f"FF vs CANBUS:    {diff_pct:.1f}% difference")

        if not pd.isna(ff_avg) and not pd.isna(noxmaf_avg):
            diff_pct = abs((noxmaf_avg - ff_avg) / ff_avg) * 100
            print(f"NOxMAF vs FF:    {diff_pct:.1f}% difference")

    print("\n" + "=" * 80)
    print(f"SINGLE-MACHINE ANALYSIS COMPLETE FOR MACHINE {MACHINE_ID}")
    print("=" * 80)


# %% ============================================================================
# SUMMARY EXPORT
# ==============================================================================

print("\n" + "=" * 80)
print("EXPORTING SUMMARY DATA")
print("=" * 80)

# Create comprehensive machine summary table
# Build aggregation dict dynamically to handle missing columns
agg_dict = {
    "duration_from_rows": "mean",
    # "FF_validated_duration": "mean",
    "NOxMAF_motorbelasting": "mean",
    "NOxMAF_NOxPerLiter": "mean",
    "NOxMAF_FPH": "mean",
    # "FF_FPH": "mean",
    "NOxTotal": "mean",
    "NOxMAF_AUB_NoxEmission": "mean",
    "datekey": "nunique",
    "Stage+Groep": "first",
}

# Add optional columns if they exist
if "MainGroupLabel" in nox_df.columns:
    agg_dict["MainGroupLabel"] = "first"
if "MerkType" in nox_df.columns:
    agg_dict["MerkType"] = "first"
if "Power" in nox_df.columns:
    agg_dict["Power"] = "first"
if "ConstructionYear" in nox_df.columns:
    agg_dict["ConstructionYear"] = "first"
if "name" in nox_df.columns:
    agg_dict["name"] = "first"

summary_df = nox_df.groupby("MachineId").agg(agg_dict).reset_index()

# Rename core columns
rename_dict = {
    "duration_from_rows": "avg_hours_per_day",
    "NOxMAF_motorbelasting": "avg_motorbelasting",
    "NOxMAF_NOxPerLiter": "avg_nox_per_liter",
    "NOxMAF_FPH": "avg_fuel_per_hour",
    "NOxTotal": "nox_per_day_kg_sensor",
    "NOxMAF_AUB_NoxEmission": "nox_per_day_kg_aub",
    "datekey": "num_days"
}
summary_df = summary_df.rename(columns=rename_dict)

# Round numeric columns
numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
summary_df[numeric_cols] = summary_df[numeric_cols].round(2)

# Reorder columns
column_order = [
    "MachineId",
    "name",
    "MainGroupLabel",
    "MerkType",
    "Power",
    "ConstructionYear",
    "Stage+Groep",
    "num_days",
    "avg_hours_per_day",
    "avg_fuel_per_hour",
    "avg_motorbelasting",
    "avg_nox_per_liter",
    "nox_per_day_kg_sensor",
    "nox_per_day_kg_aub",
]
# Only include columns that exist
column_order = [c for c in column_order if c in summary_df.columns]
summary_df = summary_df[column_order]

# Display and save
print(f"\nMachine summary table: {len(summary_df)} machines")
display(summary_df)

output_file = "boskalis_machine_summary.csv"
summary_df.to_csv(output_file, index=False)
print(f"\n✓ Summary exported to: {output_file}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nThree-level analysis:")
print("  1. MACHINE-LEVEL: Overview of fleet performance")
print("  2. MACHINE-DAY: Detailed patterns with fitted curves")
print("  3. SINGLE-MACHINE: Deep dive into specific machine behavior")
print("\nChange MACHINE_ID variable in Section 3 to analyze different machines")
print("=" * 80)

# %%
