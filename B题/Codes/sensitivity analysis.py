import pandas as pd
import numpy as np
import copy, csv, json, math, random, time
from functools import wraps
from typing import Any, Dict, List, Tuple, Optional
import seaborn as sns
import matplotlib.pyplot as plt

df_data = {}
df_data['ALPHA'] = (7381.63-7390.17)/0.000842105262156
df_data['NF'] = (7381.20-7400.88)/0.2
df_data['taskflow'] = (8112.32-6101.60)/0.2
df_data['Noise Density'] = (8273.14-7999.25)/0.2
df_data['Ruili Attenuation'] = 0

df = pd.DataFrame.from_dict(df_data, orient='index', columns=['sensitivity'])
print("Original DataFrame:")
print(df)
from sklearn.preprocessing import StandardScaler
import seaborn as sns
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
df_scaled['sensitivity'] = df_scaled['sensitivity'].abs()

# Sort by sensitivity for better visualization
df_scaled = df_scaled.sort_values('sensitivity', ascending=False)

print("\nSorted Scaled DataFrame:")
print(df_scaled)


# Beautify the plot
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 7))
barplot = sns.barplot(x=df_scaled.index, y=df_scaled['sensitivity'], palette='viridis')

plt.xticks(rotation=45, ha='right')
plt.xlabel('Parameters', fontsize=12, fontweight='bold')
plt.ylabel('Sensitivity (Absolute Scaled Value)', fontsize=12, fontweight='bold')
plt.title('Sensitivity Analysis of Parameters', fontsize=16, fontweight='bold')

# Add data labels on top of the bars
for p in barplot.patches:
    barplot.annotate(format(p.get_height(), '.2f'),
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha = 'center', va = 'center',
                     xytext = (0, 9),
                     textcoords = 'offset points')

plt.tight_layout()
plt.show()