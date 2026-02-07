import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from fpdf import FPDF

# --- INITIAL SETUP & CLEANING ---
def run_analysis():
    df = pd.read_csv('Movie_Analytics.csv')
    df = df.rename(columns={'Success_Metric': 'Box_Office_INR_Crores', 'Unnamed: 10': 'Watch_Hours_Millions'})
    
    numeric_cols = ['IMDB', 'Letterboxd', 'RT_Critic', 'RT_Audience', 'Box_Office_INR_Crores', 'Watch_Hours_Millions']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').str.replace('NULL', '').str.replace('nan', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Normalize Success
    scaler = MinMaxScaler()
    df['Normalized_Success'] = np.nan
    mask_bo = df['Box_Office_INR_Crores'].notnull()
    mask_wh = df['Watch_Hours_Millions'].notnull()
    df.loc[mask_bo, 'Normalized_Success'] = scaler.fit_transform(df.loc[mask_bo, ['Box_Office_INR_Crores']])
    df.loc[mask_wh, 'Normalized_Success'] = scaler.fit_transform(df.loc[mask_wh, ['Watch_Hours_Millions']])
    
    return df

df = run_analysis()

# --- GENERATE PLOTS ---
# Plot 1: Distribution
plt.figure(figsize=(10, 5))
sns.kdeplot(df['IMDB'], label='IMDb', fill=True, color='blue')
sns.kdeplot(df['Letterboxd']*2, label='Letterboxd (Scaled)', fill=True, color='orange')
plt.title('Rating Probability Distributions')
plt.savefig('distribution.png')
plt.close()

# Plot 2: Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[['IMDB', 'Letterboxd', 'RT_Critic', 'RT_Audience', 'Normalized_Success']].corr(), annot=True, cmap='coolwarm')
plt.title('Platform Correlation Heatmap')
plt.savefig('correlation.png')
plt.close()

# Plot 3: Regression
reg_data = df.dropna(subset=['IMDB', 'Normalized_Success'])
X = reg_data[['IMDB']].values
y = reg_data['Normalized_Success'].values
model = LinearRegression().fit(X, y)
plt.figure(figsize=(8, 6))
plt.scatter(X, y, alpha=0.5)
plt.plot(X, model.predict(X), color='red')
plt.title(f'Linear Regression (R²={model.score(X, y):.2f})')
plt.savefig('regression.png')
plt.close()

# --- THE PDF COMPILER ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'CineAnalytics: Multi-Platform Movie Success Study', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, text):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, text)
        self.ln()

pdf = PDF()
pdf.add_page()

# Section 1: Central Tendency
pdf.chapter_title('1. Central Tendency Analysis')
stats_text = f"Mean Ratings:\nIMDb: {df['IMDB'].mean():.2f} | Letterboxd: {df['Letterboxd'].mean():.2f}\n" \
             f"RT Critic: {df['RT_Critic'].mean():.2f}% | RT Audience: {df['RT_Audience'].mean():.2f}%"
pdf.chapter_body(stats_text)
pdf.image('distribution.png', x=10, w=150)
pdf.ln(10)

# Section 2: Dispersion & Correlation
pdf.add_page()
pdf.chapter_title('2. Dispersion & Platform Correlation')
df['RT_Gap'] = df['RT_Audience'] - df['RT_Critic']
disp_text = f"Standard Deviation of Audience-Critic Gap: {df['RT_Gap'].std():.2f}\n" \
            "This high variance indicates significant polarization between professional critics and general fans."
pdf.chapter_body(disp_text)
pdf.image('correlation.png', x=10, w=130)

# Section 3: Regression & Conclusion
pdf.add_page()
pdf.chapter_title('3. Predictive Success Model')
pdf.chapter_body('The following regression analysis determines if IMDb scores can predict commercial success.')
pdf.image('regression.png', x=10, w=140)

pdf.output('Final_Movie_Report.pdf')
print("PDF Generated: Final_Movie_Report.pdf")