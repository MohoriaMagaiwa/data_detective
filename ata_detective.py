# 📊 Data Detective: Unmasking Insights with Pandas & Matplotlib

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# -------------------- Task 1: Load and Explore Dataset --------------------
print("🔍 Loading the Iris dataset...")

# Load dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Display first rows
print("\n🪷 First 5 rows of the dataset:")
print(df.head())

# Explore structure
print("\n📋 Dataset Info:")
print(df.info())

# Check for missing values
print("\n⚠️ Missing values count:")
print(df.isnull().sum())

# Clean data (if any missing values)
df = df.dropna()

# -------------------- Task 2: Basic Data Analysis --------------------
print("\n📊 Basic Statistics:")
print(df.describe())

# Group by species and compute mean
group_means = df.groupby('target').mean()
print("\n📈 Average values by species (target):")
print(group_means)

# Observation: Let's rename target column values for clarity
species_names = {i: name for i, name in enumerate(iris.target_names)}
df['species'] = df['target'].map(species_names)

# -------------------- Task 3: Data Visualization --------------------
sns.set(style="whitegrid")

# 1. Line chart (trend over index to simulate time series for sepal length)
plt.figure(figsize=(8,5))
plt.plot(df.index, df['sepal length (cm)'], color='green', label='Sepal Length')
plt.title("📈 Sepal Length Trend Over Samples")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()

# 2. Bar chart (average petal length per species)
plt.figure(figsize=(6,4))
df.groupby('species')['petal length (cm)'].mean().plot(kind='bar', color=['#FFB6C1','#ADD8E6','#98FB98'])
plt.title("🌸 Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

# 3. Histogram (distribution of sepal width)
plt.figure(figsize=(6,4))
plt.hist(df['sepal width (cm)'], bins=15, color='#87CEEB', edgecolor='black')
plt.title("📊 Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 4. Scatter plot (relationship between sepal and petal length)
plt.figure(figsize=(6,4))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df, palette='Set2')
plt.title("🧬 Sepal vs Petal Length by Species")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

print("\n✅ Analysis complete — Data detective mission accomplished!")
