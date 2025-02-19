import pandas as pd 
import numpy as np
import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from util.create_pillar import create_pillar_diagram 

# Indlæs data
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'Data', 'Police', 'police.csv')
df = pd.read_csv(csv_path)

# TRIN 1: DATA CLEANING
# Fjern rækker med manglende værdier
df_cleaned = df.dropna(subset=['driver_gender', 'violation', 'stop_outcome']).copy()

# Konverter datoer
df_cleaned['stop_date'] = pd.to_datetime(df_cleaned['stop_date'])
df_cleaned['stop_year'] = df_cleaned['stop_date'].dt.year
df_cleaned['stop_month'] = df_cleaned['stop_date'].dt.month

# TRIN 2: ANALYSE
if 'driver_age_raw' in df_cleaned:
    # Beregn alder
    df_cleaned['driver_age'] = 2021 - df_cleaned['driver_age_raw']
    
    # Opret aldersgrupper
    bins = np.array([0, 20, 30, 40, 50, 60, 70, 80, np.inf])
    labels = ['<20', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
    df_cleaned['age_group'] = pd.cut(df_cleaned['driver_age'], bins=bins, labels=labels, right=False)
    
    # Analyser aldersgrupper
    age_stats = df_cleaned.groupby('age_group').agg({
        'stop_date': 'count',
        'driver_age': 'mean'
    }).rename(columns={'stop_date': 'antal_stops', 'driver_age': 'gns_alder'})
    
    # Beregn procenter
    total_stops = len(df_cleaned)
    age_stats['procent'] = (age_stats['antal_stops'] / total_stops * 100)

    # Analyser køn og overtrædelser
    gender_violation_stats = df_cleaned.groupby(['driver_gender', 'violation']).size().unstack(fill_value=0)
    gender_violation_pct = gender_violation_stats.div(gender_violation_stats.sum(axis=1), axis=0) * 100

    # Tæl stops per race og køn
    stops_per_race = df_cleaned['driver_race'].value_counts()
    stops_per_gender = df_cleaned['driver_gender'].value_counts()

    # TRIN 3: VISUALISERING
    # Søjlediagram for aldersfordeling
    create_pillar_diagram(
        {'Aldersfordeling': age_stats['antal_stops']},
        title='Fordeling af trafikstops i aldersgrupper',
        xlabel='Alder',
        ylabel='Antal stops'
    )

    # Stablet søjlediagram for køn og race
    create_pillar_diagram(
        {'Race': stops_per_race, 'Køn': stops_per_gender},
        title='Fordeling af køn og race',
        xlabel='Kategori',
        ylabel='Antal stops',
        stacked=True
    )

    # Søjlediagram for overtrædelser per køn
    
    create_pillar_diagram(
        {'Overtrædelser': gender_violation_stats},
        title='Overtrædelser fordelt på køn',
        xlabel='Køn',
        ylabel='Antal',
        stacked=True
    )

    # Vis statistik pænt formateret
    # flot formatering  https://www.geeksforgeeks.org/python-output-formatting/ 
    print("\nStatistik for aldersgrupper:")
    print(f"{'Gruppe':6} {'Antal':>8} {'Procent':>10} {'Gns. alder':>12}")
    print("-" * 45)
    
    for group in labels:
        stats = age_stats.loc[group]
        print(f"{group:6} {int(stats['antal_stops']):8d} {stats['procent']:9.1f}% {stats['gns_alder']:11.1f}")

# Gem renset datasæt
output_path = os.path.join(os.path.dirname(csv_path), 'police_cleaned.csv')
df_cleaned.to_csv(output_path, index=False)