# detector_core.py

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from scipy import stats
import logging
from typing import Dict, Tuple, Optional, List
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_or_generate_dataset(uploaded_file=None):
    """Load dataset from uploaded file or use default dataset from project folder."""
    if uploaded_file is not None:
        # User uploaded their own file
        data = pd.read_csv(uploaded_file)
        source = uploaded_file.name
    else:
        # Use the default dataset from the project folder
        default_dataset_path = Path(__file__).parent / "chemical_reaction_yield_dataset.csv"
        if default_dataset_path.exists():
            data = pd.read_csv(default_dataset_path)
            source = "chemical_reaction_yield_dataset.csv (default)"
        else:
            raise FileNotFoundError(
                f"Default dataset not found at {default_dataset_path}. "
                "Please upload a CSV file or ensure the dataset file is in the project folder."
            )
    return data, source


def calculate_optimal_contamination(data, target_col='yield'):
    """
    Automatically calculate optimal contamination rate using IQR method with balanced approach.
    Uses a combination of IQR outlier detection and Z-score method for robustness.
    """
    if target_col not in data.columns:
        return 0.05  # Default fallback
    
    y = data[target_col]
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    
    # IQR method: outliers outside 1.5*IQR
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_iqr = ((y < lower_bound) | (y > upper_bound)).sum()
    contamination_iqr = outliers_iqr / len(y)
    
    # Z-score method: points beyond 2.5 standard deviations
    mean_y = y.mean()
    std_y = y.std()
    outliers_z = ((y < mean_y - 2.5 * std_y) | (y > mean_y + 2.5 * std_y)).sum()
    contamination_z = outliers_z / len(y)
    
    # Take the average of both methods for more balanced detection
    contamination = (contamination_iqr + contamination_z) / 2
    
    # Use the calculated value directly (no floor or ceiling limits)
    # This allows for the best possible detection based on actual data characteristics
    
    logger.info(f"Auto-calculated contamination rate: {contamination:.3f} (IQR: {outliers_iqr}, Z-score: {outliers_z} outliers)")
    return contamination


def calculate_optimal_threshold(data, target_col='yield'):
    """
    Calculate optimal initial threshold based on coefficient of variation.
    Higher variability = higher threshold needed.
    """
    if target_col not in data.columns:
        return 3.0  # Default fallback
    
    y = data[target_col]
    cv = y.std() / y.mean() if y.mean() != 0 else 0
    
    # Scale threshold based on variability
    # Low CV (<0.1) -> lower threshold (2-3%)
    # Medium CV (0.1-0.2) -> medium threshold (3-5%)
    # High CV (>0.2) -> higher threshold (5-8%)
    
    if cv < 0.1:
        threshold = 2.0 + cv * 10  # 2-3%
    elif cv < 0.2:
        threshold = 3.0 + (cv - 0.1) * 20  # 3-5%
    else:
        threshold = 5.0 + min((cv - 0.2) * 30, 3)  # 5-8%
    
    threshold = max(1.0, min(10.0, threshold))  # Clamp between 1-10%
    
    logger.info(f"Auto-calculated threshold: {threshold:.2f}% (CV: {cv:.3f})")
    return threshold


def detect_anomalies_and_analyze(data, contamination=None, n_estimators=200, random_state=42):
    """
    Perform comprehensive anomaly detection and analysis with advanced metrics.
    
    Args:
        data: DataFrame with process variables
        contamination: Expected proportion of anomalies (auto-calculated if None)
        n_estimators: Number of trees for RandomForest (default: 200)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (data, importances, metrics, target_col, iso_model, rf_model)
    """
    try:
        # Clean and validate
        original_shape = data.shape
        data = data.copy()
        data.columns = [c.lower().replace(' ', '_') for c in data.columns]
        data = data.dropna().drop_duplicates()
        
        if len(data) < 10:
            raise ValueError(f"Insufficient data after cleaning: {len(data)} records (minimum 10 required)")
        
        logger.info(f"Data cleaned: {original_shape[0]} -> {len(data)} records")

        # Target + features
        target_col = 'yield' if 'yield' in data.columns else data.columns[-1]
        features = [c for c in data.columns if c != target_col and pd.api.types.is_numeric_dtype(data[c])]
        
        if not features:
            raise ValueError("No numeric feature columns found")
        
        X, y = data[features].copy(), data[target_col].copy()
        
        # Auto-calculate contamination if not provided
        if contamination is None:
            contamination = calculate_optimal_contamination(data, target_col)
        
        # Isolation Forest for anomaly detection
        iso = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=100)
        anomaly_scores = iso.fit_predict(X)
        data['anomaly'] = anomaly_scores == -1
        data['anomaly_score'] = iso.score_samples(X)
        
        anomaly_count = int(data['anomaly'].sum())
        anomaly_rate = anomaly_count / len(data)
        logger.info(f"Detected {anomaly_count} anomalies ({anomaly_rate:.2%})")

        # Random Forest for prediction and feature importance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=None
        )
        
        rf = RandomForestRegressor(
            n_estimators=n_estimators, 
            random_state=random_state,
            n_jobs=-1,
            max_depth=None,
            min_samples_split=5
        )
        rf.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = rf.predict(X_train)
        y_pred_test = rf.predict(X_test)
        
        # Feature importances
        importances = pd.DataFrame({
            'feature': features,
            'importance': rf.feature_importances_,
            'std': np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
        }).sort_values(by='importance', ascending=False)
        
        # Cross-validation for robust metrics
        cv_scores = cross_val_score(rf, X, y, cv=5, scoring='r2', n_jobs=-1)
        cv_mse = -cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

        # Comprehensive metrics
        metrics = {
            # Model performance
            'mse': mean_squared_error(y_test, y_pred_test),
            'mae': mean_absolute_error(y_test, y_pred_test),
            'mape': mean_absolute_percentage_error(y_test, y_pred_test),
            'r2': r2_score(y_test, y_pred_test),
            'r2_train': r2_score(y_train, y_pred_train),
            'r2_cv_mean': cv_scores.mean(),
            'r2_cv_std': cv_scores.std(),
            'mse_cv_mean': cv_mse.mean(),
            'mse_cv_std': cv_mse.std(),
            
            # Anomaly metrics
            'anomaly_count': anomaly_count,
            'anomaly_rate': anomaly_rate,
            'total_records': len(data),
            
            # Data quality
            'n_features': len(features),
            'train_size': len(X_train),
            'test_size': len(X_test),
            
            # Statistical metrics
            'yield_mean': float(y.mean()),
            'yield_std': float(y.std()),
            'yield_median': float(y.median()),
            'yield_min': float(y.min()),
            'yield_max': float(y.max()),
            
            # Anomaly yield stats
            'anomaly_yield_mean': float(data[data['anomaly']][target_col].mean()) if anomaly_count > 0 else None,
            'normal_yield_mean': float(data[~data['anomaly']][target_col].mean()),
            'yield_impact': float(data[~data['anomaly']][target_col].mean() - data[data['anomaly']][target_col].mean()) if anomaly_count > 0 else None,
        }
        
        # Statistical significance tests
        if anomaly_count > 5 and anomaly_count < len(data) - 5:
            normal_yields = data[~data['anomaly']][target_col].values
            anomaly_yields = data[data['anomaly']][target_col].values
            
            # T-test for yield difference
            t_stat, p_value = stats.ttest_ind(normal_yields, anomaly_yields)
            metrics['yield_t_statistic'] = float(t_stat)
            metrics['yield_p_value'] = float(p_value)
            metrics['yield_statistically_significant'] = p_value < 0.05
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_p_value = stats.mannwhitneyu(normal_yields, anomaly_yields, alternative='two-sided')
            metrics['yield_u_statistic'] = float(u_stat)
            metrics['yield_u_p_value'] = float(u_p_value)

        logger.info(f"Model trained: R² = {metrics['r2']:.3f}, CV R² = {metrics['r2_cv_mean']:.3f} ± {metrics['r2_cv_std']:.3f}")
        
        return data, importances, metrics, target_col, iso, rf
        
    except Exception as e:
        logger.error(f"Error in anomaly detection: {str(e)}")
        raise

def generate_analysis(data, features, target_col='yield', threshold=3.0):
    """
    Compare normal vs anomalous runs with comprehensive statistical analysis.
    Returns (stats_comparison, insight_text, root_cause_features).
    """
    anomalies_df = data[data['anomaly'] == True].copy()
    normal_df = data[data['anomaly'] == False].copy()

    if anomalies_df.empty:
        return None, "No anomalies detected — no comparison available.", []

    # Compute comprehensive statistics
    normal_stats = normal_df[features + [target_col]].agg(['mean', 'std', 'median', 'min', 'max'])
    anomaly_stats = anomalies_df[features + [target_col]].agg(['mean', 'std', 'median', 'min', 'max'])

    # Build detailed comparison table
    stats_comparison = pd.DataFrame({
        'Normal_Mean': normal_stats.loc['mean'],
        'Normal_Std': normal_stats.loc['std'],
        'Anomaly_Mean': anomaly_stats.loc['mean'],
        'Anomaly_Std': anomaly_stats.loc['std'],
        'Normal_Median': normal_stats.loc['median'],
        'Anomaly_Median': anomaly_stats.loc['median'],
    })
    
    stats_comparison['Absolute_Difference'] = (
        stats_comparison['Anomaly_Mean'] - stats_comparison['Normal_Mean']
    )
    stats_comparison['Percent_Difference (%)'] = (
        (stats_comparison['Absolute_Difference'] / stats_comparison['Normal_Mean'].abs()) * 100
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Effect size (Cohen's d)
    stats_comparison['Effect_Size'] = (
        stats_comparison['Absolute_Difference'] / 
        ((stats_comparison['Normal_Std'] + stats_comparison['Anomaly_Std']) / 2)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Statistical significance tests
    p_values = {}
    for feat in features + [target_col]:
        normal_vals = normal_df[feat].dropna()
        anomaly_vals = anomalies_df[feat].dropna()
        
        if len(normal_vals) > 3 and len(anomaly_vals) > 3:
            try:
                # Shapiro-Wilk test for normality
                _, p_normal = stats.shapiro(normal_vals) if len(normal_vals) <= 5000 else (None, 0.01)
                _, p_anomaly = stats.shapiro(anomaly_vals) if len(anomaly_vals) <= 5000 else (None, 0.01)
                
                # Choose test based on normality
                if p_normal > 0.05 and p_anomaly > 0.05:
                    # Both normal: use t-test
                    _, p_val = stats.ttest_ind(normal_vals, anomaly_vals)
                else:
                    # Non-normal: use Mann-Whitney U
                    _, p_val = stats.mannwhitneyu(normal_vals, anomaly_vals, alternative='two-sided')
                p_values[feat] = float(p_val)
            except:
                p_values[feat] = 1.0
        else:
            p_values[feat] = 1.0
    
    stats_comparison['P_Value'] = pd.Series(p_values)
    stats_comparison['Statistically_Significant'] = stats_comparison['P_Value'] < 0.05
    # Ensure Statistically_Significant is boolean and visible
    stats_comparison['Statistically_Significant'] = stats_comparison['Statistically_Significant'].astype(bool)

    # Root cause analysis: features with largest effect size and statistical significance
    # Relaxed thresholds: lower effect size requirement (0.3 instead of 0.5) and p-value threshold
    root_cause_df = stats_comparison[
        (stats_comparison['P_Value'] < 0.05) &
        (stats_comparison['Effect_Size'].abs() > 0.3) &
        (~stats_comparison.index.isin([target_col, 'anomaly', 'anomaly_score']))
    ].sort_values('Effect_Size', key=abs, ascending=False)
    
    root_cause_features = root_cause_df.index.tolist()[:5]  # Top 5 root causes

    # Generate comprehensive insights
    yield_diff = stats_comparison.loc[target_col, 'Percent_Difference (%)']
    if isinstance(yield_diff, pd.Series):
        yield_diff = float(yield_diff.iloc[0])
    else:
        yield_diff = float(yield_diff)

    # Identify significant features
    significant_features_df = stats_comparison[
        (stats_comparison['Percent_Difference (%)'].abs() > threshold) &
        (stats_comparison['Statistically_Significant']) &
        (~stats_comparison.index.isin([target_col, 'anomaly', 'anomaly_score']))
    ]
    feature_names = significant_features_df.index.tolist()

    # Build insight text
    insights = []
    insights.append(f"**Anomaly Impact:** Anomalous runs show a **{yield_diff:.2f}%** difference in {target_col}.")
    p_val = stats_comparison.loc[target_col, 'P_Value']
    # Format p-value consistently - use same format as table (.4f, no rounding)
    # Ensure exact same formatting as table display
    if isinstance(p_val, (int, float, np.number)):
        p_val_float = float(p_val)
        p_val_formatted = f"{p_val_float:.4f}" if p_val_float >= 0.0001 else "<0.0001"
    else:
        p_val_formatted = str(p_val)
    insights.append(f"**Statistical Significance:** {'Yes' if stats_comparison.loc[target_col, 'Statistically_Significant'] else 'No'} (p={p_val_formatted})")
    
    if root_cause_features:
        insights.append(f"**Root Cause Indicators:** Top features with significant deviations: {', '.join(root_cause_features[:3])}")
    
    if feature_names:
        if len(feature_names) == 1:
            feature_list_str = f"**{feature_names[0]}**"
        elif len(feature_names) == 2:
            feature_list_str = f"**{feature_names[0]}** and **{feature_names[1]}**"
        else:
            feature_list_str = f"**{', '.join(feature_names[:-1])}**, and **{feature_names[-1]}**"
        insights.append(f"**Key Finding:** Deviations over {threshold:.1f}% in {feature_list_str} are correlated with anomalous yield.")
    else:
        insights.append(f"No single feature deviated more than {threshold:.1f}% in the anomalous runs (statistically significant threshold).")

    insight_text = "\n\n".join(insights)

    return stats_comparison, insight_text, root_cause_features


def calculate_distribution_metrics(data: pd.DataFrame, target_col: str) -> Dict:
    """Calculate distribution metrics for yield data."""
    yield_data = data[target_col]
    
    return {
        'skewness': float(stats.skew(yield_data)),
        'kurtosis': float(stats.kurtosis(yield_data)),
        'q1': float(yield_data.quantile(0.25)),
        'q3': float(yield_data.quantile(0.75)),
        'iqr': float(yield_data.quantile(0.75) - yield_data.quantile(0.25)),
        'cv': float(yield_data.std() / yield_data.mean() if yield_data.mean() != 0 else 0),  # Coefficient of variation
    }


def predict_yield(rf_model, feature_values: Dict[str, float], feature_names: List[str]) -> Tuple[float, Dict]:
    """
    Predict yield given feature values. Returns prediction and feature contributions.
    
    Args:
        rf_model: Trained RandomForest model
        feature_values: Dictionary mapping feature names to values
        feature_names: List of feature names in correct order
        
    Returns:
        (predicted_yield, feature_contributions_dict)
    """
    # Create input array in correct order
    X_input = np.array([[feature_values.get(feat, 0) for feat in feature_names]])
    
    # Predict
    prediction = rf_model.predict(X_input)[0]
    
    # Feature contributions (simplified - using feature importance weighted by deviation from mean)
    contributions = {}
    for i, feat in enumerate(feature_names):
        contributions[feat] = rf_model.feature_importances_[i]
    
    return float(prediction), contributions
