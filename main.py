import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

# PROJECT: Multi-Platform Movie Metrics Statistical Analysis
# Group Members: [Names]

# 0. LOADING AND CLEANING THE DATA
df = pd.read_csv('Movie_Analytics.csv')

# Giving columns better names so the code is easier to read during the demo
df.rename(columns={'Success_Metric': 'Box_Office', 'Unnamed: 10': 'Netflix_Hours'}, inplace=True)

# Cleaning logic: convert to string -> remove commas -> convert back to numeric
df['Box_Office'] = df['Box_Office'].astype(str).str.replace(',', '').str.replace('"', '').replace('NULL', np.nan)
df['Box_Office'] = pd.to_numeric(df['Box_Office'], errors='coerce')
df['Netflix_Hours'] = pd.to_numeric(df['Netflix_Hours'], errors='coerce')


# PRESENTER 1: DATA NORMALIZATION (The Fairness Step)
# Since we have Rupees (Crores) and Watch Hours (Millions), we can't compare them.
# We use Min-Max scaling to put every movie on a scale from 0 to 1.


def min_max_scale(series):
    return (series - series.min()) / (series.max() - series.min())

df['Box_Office_Norm'] = min_max_scale(df['Box_Office'])
df['Netflix_Hours_Norm'] = min_max_scale(df['Netflix_Hours'])

# We create a single 'Success Index'. This is our main Y-variable for the project.
df['Success_Index'] = df[['Box_Office_Norm', 'Netflix_Hours_Norm']].max(axis=1)

print("--- P1: Success Index Created ---")
print(df[['Movie_Title', 'Success_Index']].head())


# PRESENTER 2: CENTRAL TENDENCY (The "Typical" Movie)
# We calculate Mean and Median to see if the average is being "pulled" by hits.
imdb_mean = df['IMDB'].mean()
imdb_median = df['IMDB'].median()
imdb_mode = df['IMDB'].mode()[0]



print(f"\n--- P2: Central Tendency ---\nMean: {imdb_mean:.2f}, Median: {imdb_median:.2f}, Mode: {imdb_mode:.2f}")


# PRESENTER 3: DISPERSION & SKEWNESS (The Flop Effect)
# Standard Deviation shows how polarized the ratings are. 
# We use the Pearson Coefficient for skewness to prove the "long tail" of flops.
imdb_std = df['IMDB'].std()

# This is the formula we will solve on the board: 3 * (Mean - Median) / StdDev
imdb_skew_manual = 3 * (imdb_mean - imdb_median) / imdb_std



print(f"\n--- P3: Spread & Skew ---\nStandard Deviation: {imdb_std:.2f}")
print(f"Manual Skewness: {imdb_skew_manual:.2f}")

# Plotting the distribution to show the class the "lean" in the data
plt.figure(figsize=(8, 5))
sns.histplot(df['IMDB'], kde=True, color='skyblue')
plt.title('IMDb Rating Distribution')
plt.savefig('distribution.png')


# PRESENTER 4: BIVARIATE CORRELATION (Quality vs. Money)
# We use Pearson's 'r' to see if there's a real link between score and success.
corr_val = df['IMDB'].corr(df['Success_Index'])



print(f"\n--- P4: Correlation Result ---\nPearson r: {corr_val:.4f}")

# Heatmap: This helps visualize which platforms correlate best with revenue.
plt.figure(figsize=(8, 6))
sns.heatmap(df[['IMDB', 'Letterboxd', 'RT_Critic', 'RT_Audience', 'Success_Index']].corr(), annot=True, cmap='YlGnBu')
plt.title('Correlation Heatmap')
plt.savefig('heatmap.png')


# PRESENTER 5: LINEAR REGRESSION (Predicting Success)
# We define a line (Y = mX + C) to predict success based on an IMDb score.
reg_data = df.dropna(subset=['Success_Index'])
X = reg_data[['IMDB']].values
y = reg_data['Success_Index'].values

model = LinearRegression().fit(X, y)
r_squared = model.score(X, y)



print(f"\n--- P5: Regression Model ---\nEquation: Y = {model.intercept_:.4f} + {model.coef_[0]:.4f}*X")
print(f"R-Squared: {r_squared:.4f}")

# Plotting the regression line over our data points
plt.figure(figsize=(8, 5))
plt.scatter(X, y, alpha=0.3, label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Prediction Line')
plt.xlabel('IMDb Rating')
plt.ylabel('Success Index')
plt.legend()
plt.savefig('regression.png')

print("\nAll stats calculated and plots saved for the presentation!")