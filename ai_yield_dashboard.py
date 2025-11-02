# AI-Driven Process Anomaly Detection Dashboard
# Author: Hridesh Singh Chauhan (Portfolio Project)

"""
Robust runnable script for the AI-Driven Process Anomaly Detection Dashboard.

This version fixes a FileNotFoundError by automatically searching for a suitable
CSV dataset under /mnt/data (recursively). If no matching CSV is found, the
script will generate a small synthetic dataset so the pipeline can still run
for demonstration and debugging purposes.

Behavior summary:
 - Look for candidate CSV files under '/mnt/data' that contain keywords like
   'yield' or 'chemical'. If found, use the best match.
 - If no candidate file exists, fall back to creating a synthetic dataset and
   continue the pipeline (this avoids hard crashes during development).
 - Produce: anomaly flags (IsolationForest), feature importance (RandomForest),
   visualizations (scatter, heatmap, importances), and a text report saved to
   /mnt/data/ai_process_report_final.txt.

Notes for the user:
 - If you have uploaded the Kaggle CSV, place it anywhere under /mnt/data. The
   script will detect it automatically (preferred file names contain 'yield',
   'chemical', or 'reaction').
 - If you prefer to explicitly point to a file, change `explicit_path` below.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# Configuration
# -------------------------------
search_root = "/mnt/data"   # root folder to search for CSVs (uploaded files land here)
preferred_keywords = ["yield", "chemical", "reaction"]
explicit_path = "C:/Users/hride/Downloads/chemical_reaction_yield_dataset.csv"  # if you want to force a specific path, set this to a string
synthetic_mode_if_missing = False  # if True, create synthetic dataset when no CSV found

# -------------------------------
# Helper: find CSV dataset
# -------------------------------

def find_candidate_csv(root=search_root, keywords=None):
    """Recursively search `root` for CSV files and rank matches by keyword hits.

    Returns full path to selected CSV or None if none found.
    """
    candidates = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith('.csv'):
                full = os.path.join(dirpath, fn)
                score = 0
                name_lower = fn.lower()
                if keywords:
                    for k in keywords:
                        if k in name_lower:
                            score += 1
                # prefer files in nested 'chemical' folder slightly
                if 'chemical' in dirpath.lower():
                    score += 0.1
                candidates.append((score, full))

    if not candidates:
        return None
    # pick candidate with highest score, break ties by earliest file name
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1]


# -------------------------------
# Load or create dataset
# -------------------------------
if explicit_path:
    if os.path.exists(explicit_path):
        data_path = explicit_path
    else:
        raise FileNotFoundError(f"explicit_path set but file not found: {explicit_path}")
else:
    data_path = find_candidate_csv(search_root, preferred_keywords)

if data_path:
    print(f"Using dataset: {data_path}")
    data = pd.read_csv(data_path)
else:
    if synthetic_mode_if_missing:
        print("No suitable CSV found under '/mnt/data'. Generating synthetic dataset for demo.")
        # Generate a realistic synthetic dataset resembling a chemical process yield dataset
        np.random.seed(42)
        n = 1000
        temperature = np.random.normal(75, 8, n)
        pressure = np.random.normal(5, 1.2, n)
        concentration = np.random.normal(0.8, 0.1, n)
        catalyst_amt = np.random.normal(0.05, 0.01, n)
        reaction_time = np.random.normal(120, 20, n)
        energy_consumption = 0.8*temperature + 0.5*pressure + np.random.normal(0,5,n)

        yield_base = (
            80
            - 0.02*(temperature - 78)**2
            + 1.8*(pressure - 5)
            + 15*(concentration - 0.8)
            - 0.01*(reaction_time - 120)
        )
        noise = np.random.normal(0, 2.0, n)
        yield_percent = yield_base + noise

        # add some anomalies
        num_anomalies = 35
        anom_indices = np.random.choice(n, num_anomalies, replace=False)
        yield_percent[anom_indices] -= np.random.uniform(10, 25, size=num_anomalies)

        data = pd.DataFrame({
            "temperature": temperature,
            "pressure": pressure,
            "concentration": concentration,
            "catalyst_amt": catalyst_amt,
            "reaction_time": reaction_time,
            "energy_consumption": energy_consumption,
            "yield": yield_percent
        })
        print("Synthetic dataset created. Shape:", data.shape)
    else:
        # Provide a clear actionable error listing /mnt/data contents for debugging
        entries = []
        for p, d, f in os.walk(search_root):
            entries.append((p, len(f)))
        raise FileNotFoundError(
            "No CSV dataset found under '/mnt/data'.\n" +
            "Please upload your Kaggle CSV into /mnt/data or set `explicit_path` to the file path.\n" +
            "Directory summary (path, files_count):\n" + '\n'.join([f"{p}: {c}" for p, c in entries[:20]])
        )

# -------------------------------
# Quick cleaning / normalizing columns
# -------------------------------
# Normalize column names
orig_columns = list(data.columns)
data.columns = [c.strip().lower().replace(' ', '_') for c in data.columns]
print('\nColumns detected:')
for c in data.columns:
    print(' -', c)

# Drop exact duplicates and drop rows with all-NaN
data = data.drop_duplicates()
if data.dropna(how='all').shape[0] != data.shape[0]:
    print(f"Dropped {data.shape[0] - data.dropna(how='all').shape[0]} rows that were all-NaN")
data = data.dropna()

# -------------------------------
# Identify target and features
# -------------------------------
if 'yield' in data.columns:
    target_col = 'yield'
else:
    # default to last numeric column if 'yield' isn't present
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns found in dataset to predict. Please check the CSV.")
    target_col = 'yield' if 'yield' in data.columns else numeric_cols[-1]
    print(f"Target column not explicitly named 'yield'. Using detected numeric target: {target_col}")

# Ensure features are numeric and exclude target
features = [c for c in data.columns if c != target_col and pd.api.types.is_numeric_dtype(data[c])]
if not features:
    raise ValueError("No numeric feature columns found. Cannot run anomaly detection or model training.")

X = data[features].copy()
y = data[target_col].copy()

print(f"\nUsing target: {target_col}")
print(f"Using features: {features}")

# -------------------------------
# Anomaly detection (IsolationForest)
# -------------------------------
iso = IsolationForest(contamination=0.05, random_state=42)
try:
    data['anomaly'] = iso.fit_predict(X)
except Exception as e:
    raise RuntimeError(f"An error occurred running IsolationForest: {e}")

# Convert to boolean for clarity
data['anomaly'] = data['anomaly'] == -1
anomaly_count = int(data['anomaly'].sum())
print(f"\nDetected {anomaly_count} anomalies out of {len(data)} records.")

# -------------------------------
# Feature importance via RandomForest regression
# -------------------------------
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(random_state=42, n_estimators=200)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

importances = pd.DataFrame({'feature': features, 'importance': rf.feature_importances_}).sort_values(by='importance', ascending=False)

print(f"\nModel Performance:\n - MSE: {mse:.3f}\n - R2: {r2:.3f}")
print('\nTop features by importance:')
print(importances.head(10).to_string(index=False))

# -------------------------------
# Visualizations
# -------------------------------
# Scatter of top feature vs target with anomalies highlighted
top_feat = importances['feature'].iloc[0]
plt.figure(figsize=(8, 6))
colors = data['anomaly'].map({True: 'red', False: 'blue'})
plt.scatter(data[top_feat], data[target_col], c=colors, alpha=0.7, s=40, edgecolor='k', linewidth=0.2)
plt.xlabel(top_feat.capitalize())
plt.ylabel(target_col.capitalize())
plt.title(f'{top_feat.capitalize()} vs {target_col.capitalize()} (anomalies in red)')
plt.tight_layout()
plt.show()

# Correlation heatmap (features + target)
plt.figure(figsize=(10, 8))
cols_for_corr = features + [target_col]
sns.heatmap(data[cols_for_corr].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Process Variables')
plt.tight_layout()
plt.show()

# Feature importance plot
plt.figure(figsize=(8, 5))
plt.bar(importances['feature'], importances['importance'])
plt.title('Feature Importance - Random Forest')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# Save outputs and report
# -------------------------------

print("\n--- Automated Anomaly Analysis ---")

# Ensure numpy is imported (it should be already from the top of your script)
import numpy as np

# Separate the normal data from the anomalous data
anomalies_df = data[data['anomaly'] == True]
normal_df = data[data['anomaly'] == False]

if anomalies_df.empty:
    print("No anomalies were detected, so no comparison can be run.")
else:
    # Get the average values for each group
    anomaly_stats = anomalies_df[features + [target_col]].mean()
    normal_stats = normal_df[features + [target_col]].mean()

    # Combine them into a comparison DataFrame
    stats_comparison = pd.DataFrame({
        'Normal_Average': normal_stats,
        'Anomaly_Average': anomaly_stats
    })

    # Add Absolute Difference
    stats_comparison['Absolute_Difference'] = stats_comparison['Anomaly_Average'] - stats_comparison['Normal_Average']

    # Add Percentage Difference ---
    # Calculate (Difference / Normal_Average) * 100
    # This shows the % change relative to the normal baseline
    stats_comparison['Percent_Difference (%)'] = (stats_comparison['Absolute_Difference'] / stats_comparison[
        'Normal_Average']) * 100

    stats_comparison = stats_comparison.replace([np.inf, -np.inf], np.nan).fillna(0)

    print("Comparing average values of Normal vs. Anomaly runs:")
    # Format the output to 2 decimal places for easier reading
    print(stats_comparison.to_string(float_format="%.2f"))
    # --- NEW: Automated Insight Snippet ---

    # Set the significance threshold
    threshold = 3.0

    # Get the yield difference
    yield_percent_diff = stats_comparison.loc[target_col, 'Percent_Difference (%)']
    # Find all features (excluding yield/anomaly) that are above the threshold
    significant_features_df = stats_comparison[
        (stats_comparison['Percent_Difference (%)'].abs() > threshold) &
        (~stats_comparison.index.isin(['yield', 'anomaly']))
        ]

    feature_names = significant_features_df.index.tolist()

    if not feature_names:
        print("\nInsight: No single feature deviated more than 3% in the anomalous runs.")
    else:
        # Format the feature list for printing
        if len(feature_names) == 1:
            feature_list_str = f"**{feature_names[0]}**"
        elif len(feature_names) == 2:
            feature_list_str = f"**{feature_names[0]}** and **{feature_names[1]}**"
        else:
            # Joins all but the last with ", " and adds ", and " before the last one
            feature_list_str = f"**{', '.join(feature_names[:-1])}**, and **{feature_names[-1]}**"

        print("\n--- Automated Insight ---")
        print(f"Key Finding: Deviations over {threshold}% in {feature_list_str} "
              f"are correlated with a **{yield_percent_diff:.2f}%** difference in yield.")

os.makedirs("/mnt/data", exist_ok=True)
# -----------------------------------------------------------

out_csv = "/mnt/data/process_data_with_anomalies.csv"
data.to_csv(out_csv, index=False)

report_path = "/mnt/data/ai_process_report_final.txt"
summary_lines = [
    "AI-Driven Process Anomaly Detection Dashboard",
    "-------------------------------------------",
    f"Dataset used: {data_path if data_path else 'synthetic dataset was generated'}",
    f"Rows: {len(data)}",
    f"Anomalies detected: {anomaly_count}",
    "\nTop feature importances (RandomForest):",
]
for feat, val in zip(importances['feature'], importances['importance']):
    summary_lines.append(f" - {feat}: {val:.3f}")
summary_lines.extend([
    "\nModel performance (RandomForest):",
    f" - R2: {r2:.3f}",
    f" - MSE: {mse:.3f}",
    "\nInsights:",
    " - Top features are likely actionable levers for improving yield.",
    " - Anomalies highlight runs that require triage; the product could surface root-cause candidates.",
    "\nSuggested product features:",
    " 1. Real-time anomaly alerts with top correlated parameters.",
    " 2. What-if optimizer to simulate small parameter changes and expected yield.",
    " 3. Batch comparison views to surface common failure modes.",
])

with open(report_path, 'w') as f:
    f.write('\n'.join(summary_lines))

print(f"\nSaved processed dataset to: {out_csv}")
print(f"Saved report to: {report_path}")