"""
Data Preparation for MWB Bubble Chart
=====================================

This script prepares the NOx interval data for the bubble chart visualization.
All data transformations happen here - the JavaScript only visualizes.

Input:  data/processed/aggregated_device_intervallen_20260122_143706.csv
Output: data/processed/aggregated_device_intervallen_20260122_143706.csv (updated in place)

Transformations Applied:
------------------------

1. power_state - Combines machine_staat and motorbelasting into 4 categories:
   - Uit: machine staat uit
   - Stationair: machine staat stationair
   - Lage belasting: werkend met motorbelasting < 25%
   - Hoge belasting: werkend met motorbelasting >= 25%

2. machine_type - Groups 17 detailed machine types into 11 categories:
   - Graafmachines (Rupsgraafmachine, Mobiele graafmachine)
   - Lader
   - Asfaltmachines (Asfaltverwerking, Asfaltverdichting)
   - Hijskraan (Mobiele/Rups/Vaste)
   - Generator
   - Grondverzet (Bulldozer, Grondwals)
   - Heistelling (Heistelling, Heischip)
   - Tractor (Tractor, Werktuigdrager, Maaier)
   - Overig (Dumper, Betonverwerking, Markeeringsmachine, Testopstelling)

3. nox_gram_per_hour - NOx emissions rate in grams per hour:
   Formula: NOx_mass_flow_kg * 1000 * 6
   - Multiply by 1000 to convert kg to grams
   - Multiply by 6 because data is per 10-minute interval (60/10 = 6)

4. nox_gram_per_liter - NOx emissions per liter of fuel consumed:
   Formula: (NOx_mass_flow_kg * 1000) / fuel_mass_flow_liter
   - Returns 0 when fuel_mass_flow_liter is 0 to avoid division by zero
"""

import pandas as pd

# Configuration
INPUT_FILE = 'data/processed/aggregated_device_intervallen_20260122_143706.csv'
OUTPUT_FILE = 'data/processed/aggregated_device_intervallen_20260122_143706.csv'

# Threshold for high/low load classification (25%)
LOAD_THRESHOLD = 0.25

# Machine type mapping: detailed MainGroupLabel -> aggregated category
MACHINE_TYPE_MAPPING = {
    # Graafmachines - separate categories
    'Hydraulische rupsgraafmachine': 'Rupsgraafmachine',
    'Mobiele graafmachine': 'Mobiele graafmachine',

    # Lader
    'Lader': 'Lader',

    # Asfaltmachines - separate categories
    'Asfaltverwerking': 'Asfaltverwerking',
    'Asfaltverdichting': 'Asfaltverdichting',

    # Hijskranen - combined
    'Mobiele hijskraan': 'Hijskraan',
    'Rupshijskraan': 'Hijskraan',
    'Vaste hijskraan': 'Hijskraan',

    # Generator
    'Generatoren': 'Generator',

    # Grondverzet - combined
    'Bulldozer': 'Grondverzet',
    'Grondwals': 'Grondverzet',

    # Heistelling - combined
    'Heistelling': 'Heistelling',
    'Heischip': 'Heistelling',

    # Tractor - combined
    'Tractor': 'Tractor',
    'Werktuigdrager': 'Tractor',
    'Maaier': 'Tractor',

    # Overig - catch-all
    'Dumper': 'Overig',
    'Betonverwerking': 'Overig',
    'Markeeringsmachine': 'Overig',
    'Testopstelling': 'Overig',
}


def get_power_state(row):
    """
    Determine power state from machine_staat and motorbelasting.

    Returns one of: 'Uit', 'Stationair', 'Lage belasting', 'Hoge belasting'

    Threshold: 25% motorbelasting for high vs low load
    """
    if row['machine_staat'] == 'Uit':
        return 'Uit'
    elif row['machine_staat'] == 'Stationair':
        return 'Stationair'
    else:  # Werkend
        if row['motorbelasting'] >= LOAD_THRESHOLD:
            return 'Hoge belasting'
        else:
            return 'Lage belasting'


def get_machine_type(main_group_label):
    """
    Map detailed MainGroupLabel to aggregated machine type category.

    Unknown labels are mapped to 'Overig'.
    """
    return MACHINE_TYPE_MAPPING.get(main_group_label, 'Overig')


def calculate_nox_gram_per_hour(nox_kg):
    """
    Calculate NOx emissions in grams per hour.

    Formula: nox_kg * 1000 * 6
    - * 1000: convert kg to grams
    - * 6: convert 10-minute interval to hourly rate (60/10)
    """
    return nox_kg * 1000 * 6


def calculate_nox_gram_per_liter(nox_kg, fuel_liter):
    """
    Calculate NOx emissions per liter of fuel consumed.

    Formula: (nox_kg * 1000) / fuel_liter
    Returns 0 when fuel_liter is 0 to avoid division by zero.
    """
    if fuel_liter > 0:
        return (nox_kg * 1000) / fuel_liter
    return 0


def main():
    print(f"Loading data from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows")

    # Apply transformations
    print("\nApplying transformations...")

    # 1. Power state classification
    print(f"  - power_state (threshold: {LOAD_THRESHOLD*100:.0f}%)")
    df['power_state'] = df.apply(get_power_state, axis=1)

    # 2. Machine type aggregation
    print("  - machine_type (11 categories)")
    df['machine_type'] = df['MainGroupLabel'].apply(get_machine_type)

    # 3. NOx gram per hour
    print("  - nox_gram_per_hour")
    df['nox_gram_per_hour'] = df['NOx_mass_flow_kg'].apply(calculate_nox_gram_per_hour)

    # 4. NOx gram per liter
    print("  - nox_gram_per_liter")
    df['nox_gram_per_liter'] = df.apply(
        lambda row: calculate_nox_gram_per_liter(row['NOx_mass_flow_kg'], row['fuel_mass_flow_liter']),
        axis=1
    )

    # Print summary statistics
    print("\n" + "="*50)
    print("Summary Statistics")
    print("="*50)

    print("\nPower State Distribution:")
    for state, count in df['power_state'].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {state}: {count:,} ({pct:.1f}%)")

    print("\nMachine Type Distribution:")
    type_counts = df.groupby('machine_type')['device_id'].nunique().sort_values(ascending=False)
    for mtype, count in type_counts.items():
        print(f"  {mtype}: {count} machines")
    print(f"\nTotal: {df['device_id'].nunique()} unique machines in {len(type_counts)} categories")

    print("\nNOx Statistics:")
    print(f"  nox_gram_per_hour  - mean: {df['nox_gram_per_hour'].mean():.2f}, max: {df['nox_gram_per_hour'].max():.2f}")
    print(f"  nox_gram_per_liter - mean: {df['nox_gram_per_liter'].mean():.2f}, max: {df['nox_gram_per_liter'].max():.2f}")

    # Save to output file
    print(f"\nSaving to: {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Done! Output has {len(df.columns)} columns")

    # Show new columns added
    new_cols = ['power_state', 'machine_type', 'nox_gram_per_hour', 'nox_gram_per_liter']
    print(f"\nNew/updated columns: {', '.join(new_cols)}")


if __name__ == '__main__':
    main()
