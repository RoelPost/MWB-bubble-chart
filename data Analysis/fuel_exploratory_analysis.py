# %% [markdown]
# # Exploratory Analysis - Machine Day Sensortype Data

# %%
# Import required libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import warnings
# warnings.filterwarnings('ignore')

# %%
# Load the machine-day-sensortype level data
df_motorbelasting = pd.read_csv('/Users/roelpost/DeveloperTools/MWB bubble chart/Exploratory/Motorbelasting.csv')

df_motorbelasting.head()

# %%
# Analyze fuel-related sensors
print("=" * 60)
print("FUEL SENSOR ANALYSIS")
print("=" * 60)

# Get all unique sensor types
print("\nAll sensor types in the dataset:")
print(df_motorbelasting['SensorType'].unique())

# Filter for fuel-related sensors
fuel_sensors = df_motorbelasting[df_motorbelasting['SensorType'].str.contains('fuel', case=False, na=False)]

print(f"\n\nFuel-related sensor types and row counts:")
print("-" * 60)
fuel_sensor_counts = fuel_sensors['SensorType'].value_counts()
print(fuel_sensor_counts)

print(f"\n\nTotal rows with fuel sensors: {len(fuel_sensors):,}")
print(f"Total rows in dataset: {len(df_motorbelasting):,}")
print(f"Percentage of fuel sensor rows: {len(fuel_sensors)/len(df_motorbelasting)*100:.1f}%")

# Show summary with additional details
print("\n\nDetailed breakdown by sensor type:")
print("-" * 60)
fuel_summary = fuel_sensors.groupby('SensorType').agg({
    'MachineId': 'nunique',
    'SensorSupplier': lambda x: x.unique().tolist(),
    'totalFuelPerDay': ['mean', 'sum', 'count']
}).round(2)

for sensor_type in fuel_sensor_counts.index:
    sensor_data = fuel_sensors[fuel_sensors['SensorType'] == sensor_type]
    print(f"\n{sensor_type}:")
    print(f"  • Total rows: {len(sensor_data):,}")
    print(f"  • Unique machines: {sensor_data['MachineId'].nunique()}")
    print(f"  • Sensor suppliers: {sensor_data['SensorSupplier'].unique().tolist()}")
    print(f"  • Avg fuel per day: {sensor_data['totalFuelPerDay'].mean():.2f}")
    print(f"  • Total fuel: {sensor_data['totalFuelPerDay'].sum():.2f}")

# %%
# Create machine-day level consolidated dataframe
print("=" * 60)
print("CREATING MACHINE-DAY CONSOLIDATED DATAFRAME")
print("=" * 60)

# First, let's check what sensor types we have
print("\nAvailable sensor types:")
print(df_motorbelasting['SensorType'].value_counts())

# Pivot the data to have one column per sensor type
df_pivoted = df_motorbelasting.pivot_table(
    index=['MachineId', 'datekey', 'source_origin_id', 'MainGroupLabel', 'SubGroupLabel',
           'BrandLabel', 'TypeOfEquipment', 'displayname_unique', 'Machinegroep',
           'EngineClassificationLabel', 'ConstructionYear', 'Power'],
    columns='SensorType',
    values=['totalFuelPerDay', 'totalHoursPerDay', 'FPH', 'motorbelasting', 'countRows'],
    aggfunc='first'  # Use first value if there are duplicates
).reset_index()

# Flatten column names
df_pivoted.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in df_pivoted.columns.values]

# Create consolidated fuel column with priority: Fuel flow > CANBUS > Fuel_consumption
df_pivoted['consolidated_fuel'] = np.nan
df_pivoted['fuel_source'] = None

# Check which fuel columns exist and apply hierarchy
fuel_consumption_col = 'totalFuelPerDay_Fuel_consumption'
fuel_flow_col = 'totalFuelPerDay_Fuel flow'
canbus_col = 'totalFuelPerDay_CANBUS'

# Priority 1: Fuel flow
if fuel_flow_col in df_pivoted.columns:
    mask = df_pivoted[fuel_flow_col].notna()
    df_pivoted.loc[mask, 'consolidated_fuel'] = df_pivoted.loc[mask, fuel_flow_col]
    df_pivoted.loc[mask, 'fuel_source'] = 'Fuel flow'
    print(f"\n✓ Applied Fuel flow for {mask.sum():,} rows")

# Priority 2: CANBUS (only if Fuel flow is not available)
if canbus_col in df_pivoted.columns:
    mask = (df_pivoted['consolidated_fuel'].isna()) & (df_pivoted[canbus_col].notna())
    df_pivoted.loc[mask, 'consolidated_fuel'] = df_pivoted.loc[mask, canbus_col]
    df_pivoted.loc[mask, 'fuel_source'] = 'CANBUS'
    print(f"✓ Applied CANBUS for {mask.sum():,} rows")

# Priority 3: Fuel_consumption (only if both above are not available)
if fuel_consumption_col in df_pivoted.columns:
    mask = (df_pivoted['consolidated_fuel'].isna()) & (df_pivoted[fuel_consumption_col].notna())
    df_pivoted.loc[mask, 'consolidated_fuel'] = df_pivoted.loc[mask, fuel_consumption_col]
    df_pivoted.loc[mask, 'fuel_source'] = 'Fuel_consumption'
    print(f"✓ Applied Fuel_consumption for {mask.sum():,} rows")

# Similarly consolidate hours and motorbelasting
# Hours - use same priority
df_pivoted['consolidated_hours'] = np.nan
for sensor_type in ['Fuel flow', 'CANBUS', 'Fuel_consumption']:
    hours_col = f'totalHoursPerDay_{sensor_type}'
    if hours_col in df_pivoted.columns:
        mask = df_pivoted['consolidated_hours'].isna() & df_pivoted[hours_col].notna()
        df_pivoted.loc[mask, 'consolidated_hours'] = df_pivoted.loc[mask, hours_col]

# Motorbelasting - use same priority
df_pivoted['consolidated_motorbelasting'] = np.nan
for sensor_type in ['Fuel flow', 'CANBUS', 'Fuel_consumption']:
    mb_col = f'motorbelasting_{sensor_type}'
    if mb_col in df_pivoted.columns:
        mask = df_pivoted['consolidated_motorbelasting'].isna() & df_pivoted[mb_col].notna()
        df_pivoted.loc[mask, 'consolidated_motorbelasting'] = df_pivoted.loc[mask, mb_col]

# FPH - use same priority
df_pivoted['consolidated_FPH'] = np.nan
for sensor_type in ['Fuel flow', 'CANBUS', 'Fuel_consumption']:
    fph_col = f'FPH_{sensor_type}'
    if fph_col in df_pivoted.columns:
        mask = df_pivoted['consolidated_FPH'].isna() & df_pivoted[fph_col].notna()
        df_pivoted.loc[mask, 'consolidated_FPH'] = df_pivoted.loc[mask, fph_col]

print(f"\n{'='*60}")
print("CONSOLIDATION SUMMARY")
print(f"{'='*60}")
print(f"Total machine-day records: {len(df_pivoted):,}")
print(f"\nFuel source breakdown:")
print(df_pivoted['fuel_source'].value_counts())
print(f"\nRows with fuel data: {df_pivoted['consolidated_fuel'].notna().sum():,}")
print(f"Rows without fuel data: {df_pivoted['consolidated_fuel'].isna().sum():,}")

# Create final clean dataframe with selected columns
df_machine_day = df_pivoted[[
    'MachineId', 'datekey', 'source_origin_id', 'displayname_unique',
    'MainGroupLabel', 'SubGroupLabel', 'BrandLabel', 'TypeOfEquipment',
    'Machinegroep', 'EngineClassificationLabel', 'ConstructionYear', 'Power',
    'consolidated_fuel', 'consolidated_hours', 'consolidated_motorbelasting',
    'consolidated_FPH', 'fuel_source'
]].copy()

# Rename consolidated columns for clarity
df_machine_day.rename(columns={
    'consolidated_fuel': 'fuel',
    'consolidated_hours': 'hours',
    'consolidated_motorbelasting': 'motorbelasting',
    'consolidated_FPH': 'FPH'
}, inplace=True)

print(f"\n{'='*60}")
print(f"Final dataframe shape: {df_machine_day.shape}")
print(f"{'='*60}")

df_machine_day.head(10)

# %%
# Aggregate machine-day metrics to MainGroupLabel level
print("=" * 60)
print("MACHINE GROUP LEVEL METRICS")
print("=" * 60)

metrics_of_interest = ['hours', 'motorbelasting', 'FPH']
df_machine_day_valid = df_machine_day[df_machine_day['MainGroupLabel'].notna()].copy()

# Filter out machine-days with less than one recorded hour
low_hour_mask = df_machine_day_valid['hours'].notna() & (df_machine_day_valid['hours'] < 1)
if low_hour_mask.any():
    print(f"\nFiltering {low_hour_mask.sum():,} rows with < 1 hour recorded.")
df_machine_day_valid = df_machine_day_valid[~low_hour_mask].copy()

# Filter out invalid motorbelasting values that fall outside the 0-1 range
invalid_motorbelasting = (
    df_machine_day_valid['motorbelasting'].notna()
    & ~df_machine_day_valid['motorbelasting'].between(0, 1)
)
if invalid_motorbelasting.any():
    print(f"\nFiltering {invalid_motorbelasting.sum():,} rows with motorbelasting outside [0, 1].")
df_machine_day_valid = df_machine_day_valid[~invalid_motorbelasting].copy()

group_summary = (
    df_machine_day_valid.groupby('MainGroupLabel')
    .agg(
        machine_days=('MachineId', 'count'),
        unique_machines=('MachineId', 'nunique'),
        hours_mean=('hours', 'mean'),
        hours_std=('hours', 'std'),
        motorbelasting_mean=('motorbelasting', 'mean'),
        motorbelasting_std=('motorbelasting', 'std'),
        FPH_mean=('FPH', 'mean'),
        FPH_std=('FPH', 'std')
    )
    .reset_index()
    .sort_values('machine_days', ascending=False)
)

print("\nGroup summary (first 10 rows):")
print(group_summary.head(10).round(2))

# %%
# Calculate per-machine deviations from the MainGroupLabel averages
group_metric_stats = (
    df_machine_day_valid
    .groupby('MainGroupLabel')[metrics_of_interest]
    .agg(['mean', 'std'])
    .reset_index()
)
group_metric_stats.columns = [
    '_'.join(col).strip('_') if col[0] != 'MainGroupLabel' else col[0]
    for col in group_metric_stats.columns
]

df_machine_day_enriched = df_machine_day_valid.merge(
    group_metric_stats,
    on='MainGroupLabel',
    how='left'
)

for metric in metrics_of_interest:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    deviation_col = f"{metric}_deviation"
    zscore_col = f"{metric}_zscore"

    df_machine_day_enriched[deviation_col] = (
        df_machine_day_enriched[metric] - df_machine_day_enriched[mean_col]
    )
    df_machine_day_enriched[zscore_col] = df_machine_day_enriched[deviation_col] / (
        df_machine_day_enriched[std_col].replace({0: np.nan})
    )

print("\nPer-machine sample with deviation metrics:")
print(
    df_machine_day_enriched[
        ['MachineId', 'MainGroupLabel'] +
        metrics_of_interest +
        [f"{metric}_mean" for metric in metrics_of_interest] +
        [f"{metric}_deviation" for metric in metrics_of_interest]
    ].head(10).round(2)
)

# %%
# Variance explained by MainGroupLabel for each metric
variance_rows = []
for metric in metrics_of_interest:
    mean_col = f"{metric}_mean"
    valid = df_machine_day_enriched[[metric, mean_col]].dropna()
    if valid.empty:
        continue

    sst = ((valid[metric] - valid[metric].mean()) ** 2).sum()
    sse = ((valid[metric] - valid[mean_col]) ** 2).sum()
    explained_ratio = 1 - (sse / sst) if sst > 0 else np.nan
    residual_ratio = 1 - explained_ratio if explained_ratio is not np.nan else np.nan

    variance_rows.append({
        'metric': metric,
        'explained_ratio': explained_ratio,
        'residual_ratio': residual_ratio
    })

variance_summary = pd.DataFrame(variance_rows)
print("\nVariance explained by MainGroupLabel:")
print((variance_summary * 100).round(1))

# %%
# Visualization: group means with within-group variation (std as error bars)
summary_long_records = []
for metric in metrics_of_interest:
    summary_long_records.append(pd.DataFrame({
        'MainGroupLabel': group_summary['MainGroupLabel'],
        'metric': metric,
        'mean_value': group_summary[f"{metric}_mean"],
        'std_value': group_summary[f"{metric}_std"]
    }))
summary_long = pd.concat(summary_long_records, ignore_index=True)

fig_group_means = px.bar(
    summary_long,
    x='MainGroupLabel',
    y='mean_value',
    color='metric',
    facet_col='metric',
    facet_col_spacing=0.05,
    error_y='std_value',
    title='Average hours, motorbelasting, and FPH per MainGroupLabel\n(error bars show within-group std dev)'
)
fig_group_means.update_layout(showlegend=False, height=500)
fig_group_means.update_yaxes(matches=None)
fig_group_means.show()

# %%
# Visualization: deviation distributions (how much variation remains within groups)
deviation_long_records = []
for metric in metrics_of_interest:
    deviation_long_records.append(pd.DataFrame({
        'MainGroupLabel': df_machine_day_enriched['MainGroupLabel'],
        'metric': metric,
        'deviation': df_machine_day_enriched[f"{metric}_deviation"]
    }))
deviation_long = pd.concat(deviation_long_records, ignore_index=True).dropna(subset=['deviation'])

fig_deviation = px.box(
    deviation_long,
    x='MainGroupLabel',
    y='deviation',
    color='MainGroupLabel',
    facet_col='metric',
    facet_col_spacing=0.05,
    points='outliers',
    title='Distribution of deviations from MainGroupLabel averages'
)
fig_deviation.update_layout(showlegend=False, height=500)
fig_deviation.update_yaxes(matches=None, title='Deviation from group mean')
fig_deviation.show()

# %%
# Visualization: proportion of variance explained vs residual within MainGroupLabel
variance_long = variance_summary.melt(
    id_vars='metric',
    value_vars=['explained_ratio', 'residual_ratio'],
    var_name='component',
    value_name='ratio'
).dropna(subset=['ratio'])

component_mapping = {
    'explained_ratio': 'Explained by MainGroupLabel',
    'residual_ratio': 'Residual (within groups)'
}
variance_long['component'] = variance_long['component'].map(component_mapping)

fig_variance = px.bar(
    variance_long,
    x='metric',
    y='ratio',
    color='component',
    barmode='stack',
    text=variance_long['ratio'].map(lambda x: f"{x*100:.1f}%"),
    title='Share of variance explained by MainGroupLabel'
)
fig_variance.update_layout(yaxis_tickformat='.0%', height=400)
fig_variance.show()

# %%
# Machine type A: absolute yearly fuel use averaged per MainGroupLabel
if {'Machinegroep', 'fuel', 'datekey', 'MainGroupLabel', 'MachineId'}.issubset(df_machine_day.columns):
    df_a = df_machine_day[df_machine_day['Machinegroep'].astype(str).str.upper() == 'A'].copy()
    df_a['date'] = pd.to_datetime(df_a['datekey'], errors='coerce')
    df_a = df_a.dropna(subset=['date', 'fuel'])
    df_a['year'] = df_a['date'].dt.year

    if not df_a.empty:
        machines_per_group = (
            df_a.groupby('MainGroupLabel')['MachineId'].nunique()
        )

        per_machine_year = (
            df_a.groupby(['MainGroupLabel', 'MachineId', 'year'])['fuel']
            .sum()
            .reset_index(name='fuel_per_year')
        )
        per_group_year = (
            per_machine_year.groupby(['MainGroupLabel', 'year'])['fuel_per_year']
            .mean()
            .reset_index(name='avg_fuel_per_machine_year')
        )
        per_group_year['MainGroupLabel_with_count'] = per_group_year['MainGroupLabel'].apply(
            lambda lbl: f"{lbl} ({machines_per_group.get(lbl, 0)})"
        )

        print("\nAverage fuel per machine per year (Machinegroep A) by MainGroupLabel:")
        print(per_group_year.head())

        fig_fuel_a = px.bar(
            per_group_year,
            x='MainGroupLabel_with_count',
            y='avg_fuel_per_machine_year',
            barmode='group',
            title='Gemiddelde jaarlijkse brandstof per machine (Machinegroep A) per MainGroupLabel',
            labels={
                'MainGroupLabel_with_count': 'MainGroupLabel (machines)',
                'avg_fuel_per_machine_year': 'Gemiddelde brandstof per machine per jaar',
            },
            hover_data={'year': False}
        )
        fig_fuel_a.update_traces(marker_color='#4c78a8', showlegend=False)
        fig_fuel_a.update_layout(legend_title_text=None)
        fig_fuel_a.show()
    else:
        print("\nNo Machinegroep A records with valid fuel/date found for yearly fuel plot.")
else:
    print("\nRequired columns for Machinegroep A yearly fuel plot are missing.")

# %%
# Correlation of motorbelasting with machine properties and utilization metrics
corr_features = ['motorbelasting', 'hours', 'fuel', 'FPH', 'Power', 'ConstructionYear']
available_features = [col for col in corr_features if col in df_machine_day_valid.columns]

if len(available_features) >= 2:
    corr_matrix = df_machine_day_valid[available_features].corr().round(3)
    print("\nCorrelation matrix (subset of numeric features):")
    print(corr_matrix)

    fig_corr_heatmap = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title='Correlation heatmap: motorbelasting vs. machine properties & utilization metrics'
    )
    fig_corr_heatmap.update_layout(height=500)
    fig_corr_heatmap.show()

    if 'motorbelasting' in corr_matrix.index:
        motor_corr = (
            corr_matrix.loc['motorbelasting']
            .drop('motorbelasting')
            .reset_index()
            .rename(columns={'index': 'feature', 'motorbelasting': 'correlation'})
        )
        fig_motor_corr = px.bar(
            motor_corr,
            x='correlation',
            y='feature',
            orientation='h',
            color='correlation',
            color_continuous_scale='RdBu_r',
            range_color=(-1, 1),
            title='Correlation of motorbelasting with other features'
        )
        fig_motor_corr.update_layout(height=400)
        fig_motor_corr.show()
else:
    print("Not enough numeric features available to calculate correlations.")

# %%
# Visualization: per-machine motorbelasting boxplot with MainGroupLabel coloring
machine_day_box_data = df_machine_day_valid[
    df_machine_day_valid['motorbelasting'].notna()
    & df_machine_day_valid['motorbelasting'].between(0, 1)
].copy()

if not machine_day_box_data.empty:
    machine_stats = (
        machine_day_box_data
        .groupby('MachineId')['motorbelasting']
        .agg(['mean', 'var', 'count'])
        .reset_index()
        .rename(columns={
            'mean': 'motorbelasting_mean',
            'var': 'motorbelasting_variance',
            'count': 'machine_days'
        })
    )
    machine_day_box_data = machine_day_box_data.merge(machine_stats, on='MachineId', how='left')
    machine_day_box_data = machine_day_box_data.sort_values(['MainGroupLabel', 'MachineId'])
    ordered_machine_ids = machine_day_box_data['MachineId'].astype(str).unique().tolist()
    maingroup_order = machine_day_box_data['MainGroupLabel'].dropna().unique().tolist()
    machine_day_box_data['MachineId_str'] = pd.Categorical(
        machine_day_box_data['MachineId'].astype(str),
        categories=ordered_machine_ids,
        ordered=True
    )
    machine_day_box_data['MainGroupLabel_cat'] = pd.Categorical(
        machine_day_box_data['MainGroupLabel'],
        categories=maingroup_order,
        ordered=True
    )

    fig_motorbelasting_main_group = px.box(
        machine_day_box_data,
        x='MachineId_str',
        y='motorbelasting',
        color='MainGroupLabel_cat',
        points='outliers',
        hover_data={
            'motorbelasting_mean': ':.3f',
            'motorbelasting_variance': ':.4f',
            'machine_days': True
        },
        title='Motorbelasting distribution per machine-day (MainGroupLabel coloring)',
        category_orders={
            'MachineId_str': ordered_machine_ids,
            'MainGroupLabel_cat': maingroup_order
        }
    )
    fig_motorbelasting_main_group.update_layout(
        xaxis_title='MachineId',
        yaxis_title='Motorbelasting',
        height=600,
        showlegend=True
    )
    fig_motorbelasting_main_group.show()
else:
    print("Insufficient data to render MainGroupLabel-colored motorbelasting boxplot.")

# %%
# Visualization: per-machine hours boxplot with MainGroupLabel coloring
machine_day_hours_data = df_machine_day_valid[df_machine_day_valid['hours'].notna()].copy()

if not machine_day_hours_data.empty:
    hours_stats = (
        machine_day_hours_data
        .groupby('MachineId')['hours']
        .agg(['mean', 'var', 'count'])
        .reset_index()
        .rename(columns={
            'mean': 'hours_mean',
            'var': 'hours_variance',
            'count': 'machine_days'
        })
    )
    machine_day_hours_data = machine_day_hours_data.merge(hours_stats, on='MachineId', how='left')
    machine_day_hours_data = machine_day_hours_data.sort_values(['MainGroupLabel', 'MachineId'])
    ordered_machine_ids_hours = machine_day_hours_data['MachineId'].astype(str).unique().tolist()
    maingroup_order_hours = machine_day_hours_data['MainGroupLabel'].dropna().unique().tolist()
    machine_day_hours_data['MachineId_str'] = pd.Categorical(
        machine_day_hours_data['MachineId'].astype(str),
        categories=ordered_machine_ids_hours,
        ordered=True
    )
    machine_day_hours_data['MainGroupLabel_cat'] = pd.Categorical(
        machine_day_hours_data['MainGroupLabel'],
        categories=maingroup_order_hours,
        ordered=True
    )

    fig_hours_main_group = px.box(
        machine_day_hours_data,
        x='MachineId_str',
        y='hours',
        color='MainGroupLabel_cat',
        points='outliers',
        hover_data={
            'hours_mean': ':.2f',
            'hours_variance': ':.2f',
            'machine_days': True
        },
        title='Hours distribution per machine-day (MainGroupLabel coloring)',
        category_orders={
            'MachineId_str': ordered_machine_ids_hours,
            'MainGroupLabel_cat': maingroup_order_hours
        }
    )
    fig_hours_main_group.update_layout(
        xaxis_title='MachineId',
        yaxis_title='Hours',
        height=600,
        showlegend=True
    )
    fig_hours_main_group.show()
else:
    print("Insufficient data to render MainGroupLabel-colored hours boxplot.")

# %%
# Variance decomposition: day-to-day within machines vs. between MainGroupLabel
motorbelasting_valid = df_machine_day_valid[df_machine_day_valid['motorbelasting'].notna()].copy()

if motorbelasting_valid.empty:
    print("No valid motorbelasting data available for variance decomposition.")
else:
    global_motor_mean = motorbelasting_valid['motorbelasting'].mean()
    total_ss = ((motorbelasting_valid['motorbelasting'] - global_motor_mean) ** 2).sum()

    # Within-machine (day-to-day) variation
    machine_motor_stats = (
        motorbelasting_valid
        .groupby(['MachineId', 'MainGroupLabel'])
        .agg(
            machine_motor_mean=('motorbelasting', 'mean'),
            day_count=('motorbelasting', 'count')
        )
        .reset_index()
    )
    motor_with_machine_mean = motorbelasting_valid.merge(
        machine_motor_stats[['MachineId', 'machine_motor_mean']],
        on='MachineId',
        how='left'
    )
    within_machine_ss = (
        (motor_with_machine_mean['motorbelasting'] - motor_with_machine_mean['machine_motor_mean']) ** 2
    ).sum()

    # Between-machine variation overall
    machine_motor_stats['between_machine_component'] = (
        machine_motor_stats['day_count']
        * (machine_motor_stats['machine_motor_mean'] - global_motor_mean) ** 2
    )
    between_machine_ss = machine_motor_stats['between_machine_component'].sum()

    # Between MainGroupLabel variation
    group_motor_stats = (
        motorbelasting_valid
        .groupby('MainGroupLabel')
        .agg(
            group_motor_mean=('motorbelasting', 'mean'),
            day_count=('motorbelasting', 'count')
        )
        .reset_index()
    )
    group_motor_stats['between_group_component'] = (
        group_motor_stats['day_count']
        * (group_motor_stats['group_motor_mean'] - global_motor_mean) ** 2
    )
    between_group_ss = group_motor_stats['between_group_component'].sum()
    between_machine_within_group_ss = max(between_machine_ss - between_group_ss, 0)

    variance_components = pd.DataFrame([
        {
            'component': 'Within machine (day-to-day variation)',
            'sum_of_squares': within_machine_ss,
            'share_pct': within_machine_ss / total_ss * 100 if total_ss else np.nan
        },
        {
            'component': 'Between MainGroupLabel (machine type differences)',
            'sum_of_squares': between_group_ss,
            'share_pct': between_group_ss / total_ss * 100 if total_ss else np.nan
        },
        {
            'component': 'Between machines within MainGroupLabel',
            'sum_of_squares': between_machine_within_group_ss,
            'share_pct': between_machine_within_group_ss / total_ss * 100 if total_ss else np.nan
        }
    ])

    print("\nMotorbelasting variance decomposition:")
    print(
        variance_components[['component', 'share_pct']]
        .round({'share_pct': 1})
        .to_string(index=False)
    )

    print(
        f"\nWithin-machine day-to-day variation explains "
        f"{variance_components.loc[0, 'share_pct']:.1f}% of total variance, "
        f"while MainGroupLabel differences account for "
        f"{variance_components.loc[1, 'share_pct']:.1f}%."
    )

# %%
# Lagged motorbelasting correlations (prior day and prior week)
lagged_motorbelasting = df_machine_day_valid[
    df_machine_day_valid['motorbelasting'].notna()
    & df_machine_day_valid['datekey'].notna()
].copy()

if lagged_motorbelasting.empty:
    print("No valid motorbelasting data with date available for lagged correlation analysis.")
else:
    lagged_motorbelasting['datekey'] = pd.to_datetime(
        lagged_motorbelasting['datekey'].astype(str),
        format='%Y%m%d',
        errors='coerce'
    )
    lagged_motorbelasting.sort_values(['MachineId', 'datekey'], inplace=True)

    lagged_motorbelasting['motor_lag1'] = (
        lagged_motorbelasting
        .groupby('MachineId')['motorbelasting']
        .shift(1)
    )
    lagged_motorbelasting['motor_lag7_avg'] = (
        lagged_motorbelasting
        .groupby('MachineId')['motorbelasting']
        .transform(lambda s: s.shift(1).rolling(window=7, min_periods=3).mean())
    )

    def compute_correlations(group: pd.DataFrame) -> pd.Series:
        def corr_with(col: str) -> float:
            valid = group[['motorbelasting', col]].dropna()
            if len(valid) < 3:
                return np.nan
            return valid['motorbelasting'].corr(valid[col])

        corr_lag1 = corr_with('motor_lag1')
        corr_lag7 = corr_with('motor_lag7_avg')

        return pd.Series({
            'MainGroupLabel': group['MainGroupLabel'].iloc[0],
            'TypeOfEquipment': group['TypeOfEquipment'].iloc[0] if 'TypeOfEquipment' in group else None,
            'corr_lag1': corr_lag1,
            'corr_lag7_avg': corr_lag7,
            'var_share_lag1': corr_lag1 ** 2 if pd.notna(corr_lag1) else np.nan,
            'var_share_lag7_avg': corr_lag7 ** 2 if pd.notna(corr_lag7) else np.nan,
            'pairs_lag1': group[['motorbelasting', 'motor_lag1']].dropna().shape[0],
            'pairs_lag7_avg': group[['motorbelasting', 'motor_lag7_avg']].dropna().shape[0],
            'observed_days': group['datekey'].nunique()
        })

    machine_lag_corr = (
        lagged_motorbelasting
        .groupby('MachineId')
        .apply(compute_correlations)
        .reset_index()
    )

    overall_corr_summary = (
        machine_lag_corr[['corr_lag1', 'corr_lag7_avg', 'var_share_lag1', 'var_share_lag7_avg']]
        .describe(percentiles=[0.5, 0.9])
        .round(3)
    )

    print("\nMotorbelasting vs. lagged values (overall machine-level correlations and variance shares):")
    print(overall_corr_summary.loc[['count', 'mean', '50%', '90%']])

    maingroup_corr_summary = (
        machine_lag_corr
        .groupby('MainGroupLabel')[['corr_lag1', 'corr_lag7_avg', 'var_share_lag1', 'var_share_lag7_avg']]
        .agg(['count', 'mean', 'median'])
    )
    maingroup_corr_summary.columns = [
        f"{col[0]}_{col[1]}" for col in maingroup_corr_summary.columns
    ]
    print("\nMotorbelasting vs. lagged values by MainGroupLabel (count/mean/median of per-machine correlations and variance shares):")
    print(maingroup_corr_summary.round(3).to_string())

    # Identify representative machines with strong vs weak persistence
    eligible_machines = machine_lag_corr.dropna(subset=['corr_lag1'])
    eligible_machines = eligible_machines[eligible_machines['observed_days'] >= 10]
    if eligible_machines.empty:
        eligible_machines = machine_lag_corr.dropna(subset=['corr_lag1'])

    representative_high = (
        eligible_machines
        .sort_values('corr_lag1', ascending=False)
        .head(3)['MachineId']
        .tolist()
    )
    representative_low = (
        eligible_machines
        .sort_values('corr_lag1', ascending=True)
        .head(3)['MachineId']
        .tolist()
    )

    example_selection = representative_high + representative_low
    example_series = lagged_motorbelasting[
        lagged_motorbelasting['MachineId'].isin(example_selection)
    ].copy()
    example_series.sort_values(['MachineId', 'datekey'], inplace=True)
    example_series['MachineId_str'] = example_series['MachineId'].astype(str)
    example_series['corr_group'] = example_series['MachineId'].map(
        {mid: 'High autocorrelation' for mid in representative_high}
    ).fillna('Low autocorrelation')

    fig_example_autocorr = px.line(
        example_series,
        x='datekey',
        y='motorbelasting',
        color='corr_group',
        facet_col='MachineId_str',
        facet_col_wrap=3,
        title='Motorbelasting time series for machines with high vs low autocorrelation',
        labels={'datekey': 'Date', 'motorbelasting': 'Motorbelasting'}
    )
    fig_example_autocorr.update_traces(mode='lines+markers', opacity=0.8)
    type_lookup = machine_lag_corr.set_index('MachineId')['TypeOfEquipment'].to_dict()
    fig_example_autocorr.for_each_annotation(
        lambda a: a.update(
            text=f"Machine {machine_id} – {type_lookup.get(int(machine_id), 'Unknown')}"
            if (machine_id := a.text.split('=')[1]) else a.text
        )
    )
    fig_example_autocorr.update_xaxes(matches=None, tickformat='%b %d, %Y', tickangle=45)
    fig_example_autocorr.update_layout(height=800)
    fig_example_autocorr.show()

# %%
# Motorbelasting timelines per MainGroupLabel (per-machine lines)
if lagged_motorbelasting.empty:
    print("No lagged motorbelasting data available to plot timelines.")
else:
    timeline_data = lagged_motorbelasting.copy()
    timeline_data = timeline_data.dropna(subset=['datekey'])
    if 'SubGroupLabel' not in timeline_data.columns:
        timeline_data['SubGroupLabel'] = None
    timeline_data['MachineId_str'] = timeline_data['MachineId'].astype(str)

    group_order = (
        timeline_data.groupby('MainGroupLabel')['MachineId']
        .nunique()
        .sort_values(ascending=False)
        .index.tolist()
    )

    max_machines_display = 20

    for group in group_order:
        group_data = timeline_data[timeline_data['MainGroupLabel'] == group].copy()
        total_machines = group_data['MachineId'].nunique()

        if total_machines == 0:
            continue

        if total_machines > max_machines_display:
            top_machine_ids = (
                group_data.groupby('MachineId')['datekey']
                .nunique()
                .sort_values(ascending=False)
                .head(max_machines_display)
                .index
            )
            plot_data = group_data[group_data['MachineId'].isin(top_machine_ids)].copy()
            subtitle_note = f" (showing top {max_machines_display} of {total_machines} machines)"
        else:
            plot_data = group_data
            subtitle_note = ""

        if plot_data.empty:
            continue

        fig_group_timeline = px.line(
            plot_data,
            x='datekey',
            y='motorbelasting',
            color='MachineId_str',
            title=f"Motorbelasting over time – {group}{subtitle_note}",
            labels={'datekey': 'Date', 'motorbelasting': 'Motorbelasting', 'MachineId_str': 'MachineId'},
            hover_data={
                'MachineId': True,
                'motorbelasting': ':.3f',
                'TypeOfEquipment': True,
                'SubGroupLabel': True
            }
        )
        fig_group_timeline.update_layout(
            legend_title='MachineId',
            xaxis_title='Date',
            yaxis_title='Motorbelasting',
            height=500
        )
        fig_group_timeline.update_xaxes(tickformat='%b %d, %Y')
        fig_group_timeline.update_traces(mode='lines+markers', marker=dict(size=4, opacity=0.8))
        fig_group_timeline.show()

# %%
