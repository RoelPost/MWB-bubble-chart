"""
Data Preparation for MWB Bubble Chart
=====================================

Dit script bereidt de NOx interval data voor voor de bubble chart visualisatie.

Transformaties:
1. Belasting categorie: Combineert machine_staat en motorbelasting tot 4 categorieën:
   - Uit: machine staat uit
   - Stationair: machine staat stationair
   - Lage belasting: werkend met motorbelasting < 25%
   - Hoge belasting: werkend met motorbelasting >= 25%

2. Machine categorieën: Groepeert 19 machine types naar 11 categorieën:
   - Graafmachines blijven apart (Rupsgraafmachine, Mobiele graafmachine)
   - Asfaltmachines blijven apart (Asfaltverwerking, Asfaltverdichting)
   - Hijskranen gecombineerd (Mobiele/Rups/Vaste hijskraan)
   - Grondverzet gecombineerd (Bulldozer, Dumper, Grondwals)
   - Heistelling gecombineerd (Heistelling, Heischip)
   - Tractor gecombineerd (Tractor, Werktuigdrager, Maaier)
   - Overig (Betonverwerking, Markeeringsmachine, Testopstelling)

3. Representatieve dag: De brondata bevat machines verspreid over meerdere dagen.
   Om alle 84 machines tegelijk te tonen, worden alle timestamps omgezet naar
   één referentiedag (14 Jan). Zo toont de animatie een "typische werkdag".

Input:  data/NOx_intervals - 2026-01-22T101313.540.csv
Output: data/NOx_intervals_with_belasting.csv
"""

import pandas as pd

# Load the data
df = pd.read_csv('data/NOx_intervals - 2026-01-22T101313.540.csv')

# Convert time_interval to datetime
df['time_interval'] = pd.to_datetime(df['time_interval'])

# Create a "representative day" by extracting just the time component
# All machines will be mapped to the same reference date (2026-01-14)
# This combines data from multiple days into one animated day view
df['time_only'] = df['time_interval'].dt.strftime('%H:%M:%S')
df['time_interval'] = pd.to_datetime('2026-01-14 ' + df['time_only'])
df = df.drop(columns=['time_only'])

# Create the new combined column based on machine_staat and motorbelasting
def get_belasting_category(row):
    if row['machine_staat'] == 'Uit':
        return 'Uit'
    elif row['machine_staat'] == 'Stationair':
        return 'Stationair'
    else:  # Werkend
        if row['motorbelasting'] < 0.25:
            return 'Lage belasting'
        else:
            return 'Hoge belasting'

df['belasting_categorie'] = df.apply(get_belasting_category, axis=1)

# Combine machine types into broader categories
machine_type_mapping = {
    # Graafmachines - blijven los
    'Hydraulische rupsgraafmachine': 'Rupsgraafmachine',
    'Mobiele graafmachine': 'Mobiele graafmachine',
    # Laders
    'Lader': 'Lader',
    # Asfaltmachines - blijven los
    'Asfaltverwerking': 'Asfaltverwerking',
    'Asfaltverdichting': 'Asfaltverdichting',
    # Hijskranen - combineren
    'Mobiele hijskraan': 'Hijskraan',
    'Rupshijskraan': 'Hijskraan',
    'Vaste hijskraan': 'Hijskraan',
    # Generatoren
    'Generatoren': 'Generator',
    # Grondverzet - combineren
    'Bulldozer': 'Grondverzet',
    'Dumper': 'Grondverzet',
    'Grondwals': 'Grondverzet',
    # Heistelling + Heischip - combineren
    'Heistelling': 'Heistelling',
    'Heischip': 'Heistelling',
    # Tractor + Werktuigdrager + Maaier - combineren
    'Tractor': 'Tractor',
    'Werktuigdrager': 'Tractor',
    'Maaier': 'Tractor',
    # Overig - combineren
    'Betonverwerking': 'Overig',
    'Markeeringsmachine': 'Overig',
    'Testopstelling': 'Overig',
}

df['MachineCategorie'] = df['MainGroupLabel'].map(machine_type_mapping)

# Show distribution of new categories
print("Distribution of belasting_categorie:")
print(df['belasting_categorie'].value_counts())

print("\nDistribution of MachineCategorie:")
machine_counts = df.groupby('MachineCategorie')['device_id'].nunique().sort_values(ascending=False)
for cat, count in machine_counts.items():
    print(f"  {cat}: {count} machines")
print(f"\nTotal: {df['device_id'].nunique()} machines in {len(machine_counts)} categories")

# Save to new CSV
output_path = 'data/NOx_intervals_with_belasting.csv'
df.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")
print(f"Total rows: {len(df)}")
