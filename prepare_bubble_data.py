import pandas as pd

# Load the data
df = pd.read_csv('data/NOx_intervals - 2026-01-20T164805.872.csv')

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

# Show distribution of new categories
print("Distribution of new belasting_categorie:")
print(df['belasting_categorie'].value_counts())

# Save to new CSV
output_path = 'data/NOx_intervals_with_belasting.csv'
df.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")
print(f"Total rows: {len(df)}")
