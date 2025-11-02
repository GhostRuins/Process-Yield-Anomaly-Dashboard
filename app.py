"""
Advanced Process Yield Anomaly Detection Dashboard

A production-grade analytics platform for detecting anomalies in chemical process yields.
Built with enterprise-level error handling, performance optimization, and comprehensive analysis.

Author: Hridesh Singh Chauhan
Purpose: Portfolio project demonstrating full-stack engineering capabilities for FDSE internship applications.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
from datetime import datetime
from typing import Optional, Dict, Tuple
import json
from detector_core import (
    load_or_generate_dataset,
    detect_anomalies_and_analyze,
    generate_analysis,
    calculate_optimal_contamination,
    calculate_optimal_threshold,
    calculate_distribution_metrics,
    predict_yield
)

# ============================================================================
# Configuration & Constants
# ============================================================================

PAGE_TITLE = "Process Yield Anomaly Detection Dashboard"
APP_VERSION = "2.0"
MIN_DATA_RECORDS = 10
MAX_DATA_RECORDS = 100000  # Reasonable limit for web app

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title=PAGE_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': f"{PAGE_TITLE} v{APP_VERSION}\nDeveloped by Hridesh Singh Chauhan"
    }
)

# ============================================================================
# Custom CSS for Professional Styling
# ============================================================================

st.markdown("""
<style>
    /* Main styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #1a1a1a;
    }
    
    /* Professional color scheme with dark background */
    :root {
        --primary-color: #1e40af;
        --success-color: #059669;
        --warning-color: #d97706;
        --error-color: #dc2626;
        --text-primary: #ffffff;
        --text-secondary: #e5e7eb;
        --bg-primary: #1a1a1a;
        --bg-secondary: #2d2d2d;
    }
    
    /* Main content background */
    .main {
        background-color: #1a1a1a;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: #ffffff;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.95rem;
        color: #ffffff;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Distribution cards with dark background */
    .distribution-card {
        background: #2d2d2d;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        transition: all 0.2s;
    }
    
    .distribution-card:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
        border-left-width: 6px;
    }
    
    .distribution-label {
        font-size: 0.875rem;
        color: #ffffff;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .distribution-value {
        font-size: 1.25rem;
        color: #ffffff;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    
    /* Typography - all white text */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700;
    }
    
    .stMarkdown {
        color: #ffffff !important;
    }
    
    p, span, div {
        color: #ffffff !important;
    }
    
    /* Tables */
    .stDataFrame {
        font-size: 0.875rem;
        color: #ffffff;
        background-color: #2d2d2d;
    }
    
    /* Streamlit widgets */
    .stSelectbox label, .stSlider label, .stRadio label {
        color: #ffffff !important;
    }
    
    /* Info boxes */
    .info-box {
        background: #1e3a5f;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
        color: #ffffff;
    }
    
    /* Success boxes */
    .success-box {
        background: #1e3f3a;
        border-left: 4px solid #22c55e;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
        color: #ffffff;
    }
    
    /* Warning boxes */
    .warning-box {
        background: #3f3a1e;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
        color: #ffffff;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a1a1a;
    }
    
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Loading spinner customization */
    .stSpinner > div {
        border-top-color: #ffffff;
    }
    
    /* Captions and text */
    .stCaption {
        color: #e5e7eb !important;
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
    }
    
    .dataframe th {
        background-color: #3d3d3d !important;
        color: #ffffff !important;
    }
    
    .dataframe td {
        background-color: #2d2d2d !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Utility Functions
# ============================================================================

@st.cache_data(ttl=3600)
def validate_data(data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Validate dataset for analysis requirements.
    
    Returns:
        (is_valid, error_message)
    """
    if data is None or data.empty:
        return False, "Dataset is empty or None."
    
    if len(data) < MIN_DATA_RECORDS:
        return False, f"Insufficient data: {len(data)} records. Minimum {MIN_DATA_RECORDS} required."
    
    if len(data) > MAX_DATA_RECORDS:
        return False, f"Dataset too large: {len(data)} records. Maximum {MAX_DATA_RECORDS} supported."
    
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return False, "Dataset must contain at least 2 numeric columns."
    
    missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
    if missing_pct > 0.5:
        return False, f"Too many missing values: {missing_pct:.1%}. Maximum 50% allowed."
    
    return True, None

def format_number(value: float, decimals: int = 2) -> str:
    """Format number with appropriate precision."""
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format percentage."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}%"

# ============================================================================
# Header & Title
# ============================================================================

st.title("🔬 Process Yield Anomaly Detection Dashboard")
st.caption(f"Enterprise Analytics Platform | Version {APP_VERSION} | Developed by Hridesh Singh Chauhan")

# ============================================================================
# Sidebar Configuration
# ============================================================================

st.sidebar.header("📁 Data Configuration")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"],
    key="file_uploader_main",
    help="Upload a CSV file with process variables and yield data"
)

# Data Loading with Enhanced Error Handling
data = None
source = None
data_load_time = None

try:
    start_time = time.time()
    data, source = load_or_generate_dataset(uploaded_file)
    data_load_time = time.time() - start_time
    
    # Validate data
    is_valid, error_msg = validate_data(data)
    
    if not is_valid:
        st.sidebar.error(f"❌ {error_msg}")
        st.error(f"**Data Validation Error:** {error_msg}")
        st.stop()
    
    st.sidebar.success(f"✅ Data loaded: {source}")
    st.sidebar.write(f"**Records:** {len(data):,}")
    st.sidebar.write(f"**Columns:** {len(data.columns)}")
    st.sidebar.write(f"**Load Time:** {data_load_time:.2f}s")
    
    # Data quality metrics
    missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
    numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
    
    with st.sidebar.expander("📊 Data Quality Metrics", expanded=False):
        st.metric("Missing Data", f"{missing_pct:.1f}%")
        st.metric("Numeric Columns", numeric_cols)
        st.metric("Total Columns", len(data.columns))
        
    with st.sidebar.expander("📋 Data Preview", expanded=False):
        st.dataframe(data.head(10), use_container_width=True)
        
except Exception as e:
    st.sidebar.error(f"❌ Error loading data")
    st.error(f"**Error:** {str(e)}\n\nPlease check your data file and try again.")
    st.stop()

# ============================================================================
# Model Configuration
# ============================================================================

st.sidebar.header("⚙️ Model Configuration")

# Model parameters with tooltips
n_estimators = st.sidebar.slider(
    "Random Forest Estimators",
    50, 500, 200, 50,
    key="n_estimators_slider",
    help="Number of trees in the Random Forest model. More trees = better accuracy but slower training."
)

# Calculate optimal values before analysis (using best values)
target_col_for_calc = 'yield' if 'yield' in data.columns else data.select_dtypes(include=[np.number]).columns[-1] if len(data.select_dtypes(include=[np.number]).columns) > 0 else data.columns[-1]

calculated_threshold = calculate_optimal_threshold(data, target_col=target_col_for_calc)
calculated_contamination = calculate_optimal_contamination(data, target_col=target_col_for_calc)

# User-configurable analysis threshold (defaults to calculated value)
analysis_threshold = st.sidebar.slider(
    "Analysis Threshold (%)",
    0.5, 10.0, calculated_threshold, 0.1,
    help="Percent difference threshold for identifying statistically significant deviations between normal and anomalous runs",
    key="analysis_threshold_slider"
)

# Display calculated values
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Calculated Optimal Values")

col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Contamination", f"{calculated_contamination:.2%}")
with col2:
    st.metric("Threshold", f"{calculated_threshold:.2f}%")

st.sidebar.info(
    f"**Contamination Rate:** Auto-calculated using IQR and Z-score methods.\n\n"
    f"**Threshold:** Auto-calculated based on coefficient of variation.\n\n"
    f"You'll be asked which threshold to use when running analysis."
)

# ============================================================================
# Tab Navigation
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard",
    "🔍 Analysis",
    "📈 Visualizations",
    "🎯 Root Cause",
    "🔮 Predictions",
    "💾 Export"
])

# ============================================================================
# Session State Management
# ============================================================================

if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = None
if 'importances' not in st.session_state:
    st.session_state.importances = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'iso_model' not in st.session_state:
    st.session_state.iso_model = None
if 'rf_model' not in st.session_state:
    st.session_state.rf_model = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'analysis_timestamp' not in st.session_state:
    st.session_state.analysis_timestamp = None
if 'analysis_runtime' not in st.session_state:
    st.session_state.analysis_runtime = None
if 'show_threshold_choice' not in st.session_state:
    st.session_state.show_threshold_choice = False
if 'final_threshold' not in st.session_state:
    st.session_state.final_threshold = None

# ============================================================================
# Dashboard Tab
# ============================================================================

with tab1:
    st.header("📊 Executive Dashboard")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Show threshold choice if button was clicked
        if st.session_state.show_threshold_choice:
            st.info("⚙️ **Analysis Configuration**")
            
            threshold_choice = st.radio(
                "Select threshold for analysis:",
                [
                    f"Use calculated optimal threshold ({calculated_threshold:.2f}%)",
                    f"Use user-selected threshold ({analysis_threshold:.2f}%)"
                ],
                key="pre_analysis_threshold_choice",
                index=0
            )
            
            # Determine which threshold to use
            final_threshold = calculated_threshold if "calculated" in threshold_choice.lower() else analysis_threshold
            
            # Confirm and proceed
            col_confirm1, col_confirm2 = st.columns(2)
            with col_confirm1:
                if st.button("✅ Confirm & Run Analysis", type="primary", key="confirm_analysis_button"):
                    with st.spinner("🔄 Analyzing data... This may take a moment."):
                        analysis_start_time = time.time()
                        
                        try:
                            # Run analysis with calculated contamination (best value, no limits)
                            data_processed, importances, metrics, target_col, iso_model, rf_model = detect_anomalies_and_analyze(
        data,
                                contamination=calculated_contamination,
                                n_estimators=n_estimators
                            )
                            
                            # Extract features
                            features = [c for c in data_processed.columns
                                       if c != target_col and c not in ['anomaly', 'anomaly_score']
                                       and pd.api.types.is_numeric_dtype(data_processed[c])]
                            
                            analysis_runtime = time.time() - analysis_start_time
                            
                            # Update session state
                            st.session_state.analysis_run = True
                            st.session_state.data_processed = data_processed
                            st.session_state.importances = importances
                            st.session_state.metrics = metrics
                            st.session_state.target_col = target_col
                            st.session_state.iso_model = iso_model
                            st.session_state.rf_model = rf_model
                            st.session_state.features = features
                            st.session_state.analysis_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.session_state.analysis_runtime = analysis_runtime
                            st.session_state.final_threshold = final_threshold
                            st.session_state.show_threshold_choice = False
                            
                            st.success(f"✅ Analysis complete in {analysis_runtime:.2f}s! Navigate to other tabs to explore results.")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"**Analysis Error:** {str(e)}\n\nPlease check your data and configuration, then try again.")
                            st.exception(e)
            
            with col_confirm2:
                if st.button("❌ Cancel", key="cancel_analysis_button"):
                    st.session_state.show_threshold_choice = False
                    st.rerun()
        
        else:
            # Initial button to start analysis
            if st.button("🚀 Run Analysis", type="primary", key="run_analysis_button", use_container_width=True):
                st.session_state.show_threshold_choice = True
                st.rerun()
    
    # Display metrics if analysis has been run
    if st.session_state.analysis_run and st.session_state.metrics:
        metrics = st.session_state.metrics
        
        # Analysis metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"🕐 **Last Analysis:** {st.session_state.analysis_timestamp}")
        with col2:
            st.caption(f"⏱️ **Runtime:** {st.session_state.analysis_runtime:.2f}s")
        with col3:
            st.caption(f"📊 **Features Analyzed:** {len(st.session_state.features)}")
        
        st.divider()
        
        # Key Performance Indicators
        st.subheader("📈 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            anomaly_count = int(metrics.get('anomaly_count', 0))
            total_records = int(metrics.get('total_records', len(st.session_state.data_processed)))
            anomaly_rate = metrics.get('anomaly_rate', 0)
            st.metric(
                "Anomalies Detected",
                f"{anomaly_count:,}",
                f"{anomaly_rate:.1%} of total",
                delta_color="inverse"
            )
        
        with col2:
            r2 = metrics.get('r2', 0)
            r2_cv_mean = metrics.get('r2_cv_mean', 0)
            r2_cv_std = metrics.get('r2_cv_std', 0)
            st.metric(
                "Model R² Score",
                f"{r2:.3f}",
                f"CV: {r2_cv_mean:.3f} ± {r2_cv_std:.3f}",
                help="Coefficient of determination with 5-fold cross-validation"
            )
        
        with col3:
            yield_diff = metrics.get('yield_statistically_significant', False)
            p_val = metrics.get('yield_p_value', 1.0)
            if p_val >= 0.0001:
                p_val_str = f"{p_val:.4f}"
            else:
                p_val_str = "<0.0001"
            sig_text = "Yes" if yield_diff else "No"
            st.metric(
                "Statistical Significance",
                sig_text,
                f"p={p_val_str}",
                delta_color="normal" if yield_diff else "off",
                help="T-test for yield difference between normal and anomalous runs"
            )
        
        with col4:
            mse = metrics.get('mse', 0)
            mae = metrics.get('mae', 0)
            st.metric(
                "Mean Squared Error",
                f"{mse:.2f}",
                f"MAE: {mae:.2f}",
                help="Model prediction error metrics"
            )
        
        # Yield Impact Analysis
        st.divider()
        st.subheader("📊 Yield Impact Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            normal_mean = metrics.get('normal_yield_mean', 0)
            st.metric("Normal Yield Mean", f"{normal_mean:.2f}")
        
        with col2:
            anomaly_mean = metrics.get('anomaly_yield_mean', 0)
            if anomaly_mean:
                st.metric("Anomaly Yield Mean", f"{anomaly_mean:.2f}")
            else:
                st.metric("Anomaly Yield Mean", "N/A")
        
        with col3:
            yield_impact = metrics.get('yield_impact', 0)
            if yield_impact is not None:
                impact_pct = (yield_impact / normal_mean * 100) if normal_mean > 0 else 0
                st.metric(
                    "Yield Impact",
                    f"{yield_impact:+.2f}",
                    f"{impact_pct:+.1f}%",
                    delta_color="inverse"
                )
            else:
                st.metric("Yield Impact", "N/A")
        
        # Distribution Metrics
        st.divider()
        st.subheader("📊 Distribution Metrics")
        
        dist_metrics = calculate_distribution_metrics(st.session_state.data_processed, st.session_state.target_col)
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        metrics_to_show = [
            ('Skewness', dist_metrics['skewness'], "Measure of asymmetry"),
            ('Kurtosis', dist_metrics['kurtosis'], "Measure of tail heaviness"),
            ('Q1', dist_metrics['q1'], "First quartile"),
            ('Q3', dist_metrics['q3'], "Third quartile"),
            ('IQR', dist_metrics['iqr'], "Interquartile range"),
            ('CV', dist_metrics['cv'], "Coefficient of variation")
        ]
        
        for i, (label, value, tooltip) in enumerate(metrics_to_show):
            with [col1, col2, col3, col4, col5, col6][i]:
                st.markdown(f"""
                <div class="distribution-card">
                    <div class="distribution-label" title="{tooltip}">{label}</div>
                    <div class="distribution-value">{value:.3f}</div>
                </div>
                """, unsafe_allow_html=True)

# ============================================================================
# Analysis Tab (continues in next part due to length)
# ============================================================================

with tab2:
    st.header("🔍 Detailed Statistical Analysis")
    
    if not st.session_state.analysis_run:
        st.info("👆 Please run analysis from the Dashboard tab first.")
    else:
        data_processed = st.session_state.data_processed
        target_col = st.session_state.target_col
        features = st.session_state.features
        
        # Use the threshold that was selected when analysis was run
        if st.session_state.final_threshold is not None:
            actual_threshold = st.session_state.final_threshold
        else:
            # Fallback to slider value if not set
            actual_threshold = analysis_threshold
        
        # Generate analysis
        with st.spinner("🔄 Computing statistical analysis..."):
            stats_comparison, insight_text, root_cause_features = generate_analysis(
                data_processed,
                features,
                target_col,
                threshold=actual_threshold
            )
        
        if stats_comparison is not None:
            # Analysis Summary Metrics
            st.divider()
            st.subheader("📊 Analysis Summary")
            
            metrics = st.session_state.metrics
            anomaly_count = int(metrics.get('anomaly_count', 0)) if metrics else 0
            anomaly_rate = metrics.get('anomaly_rate', 0) if metrics else 0
            yield_sig = metrics.get('yield_statistically_significant', False) if metrics else False
            p_val = metrics.get('yield_p_value', 1.0) if metrics else 1.0
            p_val_str = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Anomalies Detected",
                    anomaly_count,
                    help=f"Total number of anomalous runs identified ({anomaly_rate:.1%} of data)"
                )
            
            with col2:
                st.metric(
                    "Anomaly Rate",
                    f"{anomaly_rate:.2%}",
                    help="Percentage of runs identified as anomalies"
                )
            
            with col3:
                st.metric(
                    "Statistical Significance",
                    "Yes" if yield_sig else "No",
                    delta=None,
                    help=f"P-Value: {p_val_str}"
                )
            
            with col4:
                normal_mean = metrics.get('normal_yield_mean', 0) if metrics else 0
                anomaly_mean = metrics.get('anomaly_yield_mean', 0) if metrics else 0
                yield_diff = normal_mean - anomaly_mean if normal_mean and anomaly_mean else 0
                st.metric(
                    "Yield Difference",
                    f"{yield_diff:.2f}",
                    help=f"Normal: {normal_mean:.2f}, Anomalous: {anomaly_mean:.2f}"
                )
            
            # Model Performance Metrics
            st.divider()
            st.subheader("🎯 Model Performance Metrics")
            
            col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
            
            with col_perf1:
                r2 = metrics.get('r2', 0) if metrics else 0
                st.metric(
                    "R² Score",
                    f"{r2:.4f}",
                    help="Coefficient of determination - higher is better"
                )
            
            with col_perf2:
                mse = metrics.get('mse', 0) if metrics else 0
                st.metric(
                    "MSE",
                    f"{mse:.4f}",
                    help="Mean Squared Error - lower is better"
                )
            
            with col_perf3:
                mae = metrics.get('mae', 0) if metrics else 0
                st.metric(
                    "MAE",
                    f"{mae:.4f}",
                    help="Mean Absolute Error - lower is better"
                )
            
            with col_perf4:
                mape = metrics.get('mape', 0) if metrics else 0
                st.metric(
                    "MAPE",
                    f"{mape:.2f}%",
                    help="Mean Absolute Percentage Error - lower is better"
                )
            
            # Cross-validation metrics
            if metrics and metrics.get('r2_cv_mean') is not None:
                col_cv1, col_cv2 = st.columns(2)
                with col_cv1:
                    r2_cv_mean = metrics.get('r2_cv_mean', 0)
                    r2_cv_std = metrics.get('r2_cv_std', 0)
                    st.metric(
                        "R² (5-Fold CV)",
                        f"{r2_cv_mean:.4f}",
                        delta=f"±{r2_cv_std:.4f}",
                        help="Cross-validated R² with standard deviation"
                    )
            
            # Distribution Metrics
            st.divider()
            st.subheader("📈 Distribution Metrics")
            
            dist_metrics = calculate_distribution_metrics(data_processed, target_col)
            
            col_dist1, col_dist2, col_dist3, col_dist4 = st.columns(4)
            
            with col_dist1:
                st.metric(
                    "Skewness",
                    f"{dist_metrics['skewness']:.3f}",
                    help="Measure of asymmetry (0 = symmetric, >0 = right-skewed, <0 = left-skewed)"
                )
            
            with col_dist2:
                st.metric(
                    "Kurtosis",
                    f"{dist_metrics['kurtosis']:.3f}",
                    help="Measure of tail heaviness (3 = normal distribution)"
                )
            
            with col_dist3:
                st.metric(
                    "IQR",
                    f"{dist_metrics['iqr']:.3f}",
                    help="Interquartile Range - spread of middle 50% of data"
                )
            
            with col_dist4:
                st.metric(
                    "Coefficient of Variation",
                    f"{dist_metrics['cv']:.3f}",
                    help="Relative variability (std/mean) - lower indicates more consistent data"
                )
            
            # Automated Insights
            st.divider()
            st.subheader("🤖 Automated Insights")
            st.markdown(insight_text)
            
            # Normal vs Anomalous Run Comparison
            st.divider()
            st.subheader("📊 Normal vs Anomalous Run Comparison")
            
            # Select key columns for display
            key_columns = ['Normal_Mean', 'Anomaly_Mean', 'Percent_Difference (%)',
                          'P_Value', 'Effect_Size', 'Statistically_Significant']
            
            # Filter to only include features and target
            display_df = stats_comparison[stats_comparison.index.isin(features + [target_col])].copy()
            
            # Create display dataframe
            summary_df = display_df[key_columns].copy()
            
            # Format columns - title case
            summary_df.columns = [col.replace('_', ' ').title() for col in summary_df.columns]
            
            # Store original numeric values for styling
            numeric_summary = summary_df.copy()
            
            # Format values for display
            formatted_df = summary_df.copy()
            for col in formatted_df.columns:
                if col == 'P Value':
                    formatted_df[col] = formatted_df[col].apply(
                        lambda x: f"{x:.4f}" if x >= 0.0001 else "<0.0001"
                    )
                elif col in ['Normal Mean', 'Anomaly Mean', 'Percent Difference (%)', 'Effect Size']:
                    formatted_df[col] = formatted_df[col].apply(
                        lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
                    )
                elif col == 'Statistically Significant':
                    formatted_df[col] = formatted_df[col].apply(lambda x: "Yes" if x else "No")
            
            # Style the table using numeric values for calculations
            # Create a mapping from formatted_df index to numeric_summary index
            def style_row(formatted_row):
                styles = pd.Series('', index=formatted_row.index)
                row_name = formatted_row.name
                numeric_row = numeric_summary.loc[row_name]
                
                is_significant = numeric_row.get('Statistically Significant') == True
                
                if is_significant:
                    percent_diff_col = 'Percent Difference (%)'
                    if percent_diff_col in numeric_row.index:
                        try:
                            percent_diff = float(numeric_row[percent_diff_col])
                            if percent_diff < 0:
                                styles[percent_diff_col] = 'background-color: #7f1d1d; color: #ffffff; font-weight: bold;'
                            else:
                                styles[percent_diff_col] = 'background-color: #14532d; color: #ffffff; font-weight: bold;'
                        except (ValueError, TypeError):
                            pass
                    
                    # Bold other significant columns (but preserve percent diff background)
                    for col in ['P Value', 'Effect Size', 'Statistically Significant']:
                        if col in styles.index:
                            styles[col] = 'font-weight: bold; color: #ffffff;'
                
                if 'Normal Mean' in styles.index:
                    styles['Normal Mean'] = 'background-color: #2d2d2d; color: #ffffff;'
                
                # Ensure all text is white
                for col in styles.index:
                    if not styles[col]:
                        styles[col] = 'color: #ffffff;'
                
                return styles
            
            styled_df = formatted_df.style.apply(style_row, axis=1).set_properties(**{
                'background-color': '#2d2d2d',
                'color': '#ffffff'
            })
            
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Feature Importance Analysis
            st.divider()
            st.subheader("🎯 Feature Importance Analysis")
            
            if st.session_state.importances is not None:
                importances = st.session_state.importances
                
                fig_imp = px.bar(
        importances,
                    x='importance',
                    y='feature',
                    orientation='h',
        title="Feature Importances (Random Forest)",
                    labels={'importance': 'Importance', 'feature': 'Feature'},
                    color='importance',
                    color_continuous_scale='Blues'
                )
                fig_imp.update_layout(
                    height=500,
                    xaxis_title="Importance Score",
                    yaxis=dict(
                        categoryorder='total ascending',
                        title_font=dict(color='#ffffff', size=12),
                        tickfont=dict(color='#ffffff', size=10)
                    ),
                    title_font=dict(color='#ffffff', size=16),
                    paper_bgcolor='#1a1a1a',
                    plot_bgcolor='#2d2d2d',
                    coloraxis_colorbar=dict(
                        title="Importance",
                        title_font=dict(color='#ffffff'),
                        tickfont=dict(color='#ffffff')
                    )
                )
                fig_imp.update_traces(
                    texttemplate='%{x:.3f}',
                    textposition='outside',
                    textfont=dict(color='#ffffff', size=10)
                )
                st.plotly_chart(fig_imp, use_container_width=True, key="feature_importance_chart_key")

# ============================================================================
# Visualizations Tab
# ============================================================================

with tab3:
    st.header("📈 Interactive Visualizations")
    
    if not st.session_state.analysis_run:
        st.info("👆 Please run analysis from the Dashboard tab first.")
    else:
        data_processed = st.session_state.data_processed
        target_col = st.session_state.target_col
        features = st.session_state.features
        
        # Visualization selection
        viz_option = st.selectbox(
            "Select Visualization Type",
            ["Scatter Plot", "Correlation Heatmap", "Distribution Comparison", "Box Plot Comparison", "Feature Importance"],
            key="viz_select_option",
            help="Choose a visualization type to explore your data"
        )
        
        if viz_option == "Scatter Plot":
            st.subheader("📊 Scatter Plot Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                x_feature = st.selectbox("X-axis Feature", features, key="x_feature_selector")
            with col2:
                y_feature = st.selectbox("Y-axis Feature", [target_col] + features, index=0, key="y_feature_selector")
            
            # Enhanced scatter plot
            fig_scatter = px.scatter(
                data_processed,
                x=x_feature,
                y=y_feature,
                color='anomaly',
                color_discrete_map={True: '#dc2626', False: '#2563eb'},
                title=f"{x_feature} vs {y_feature}",
                labels={'anomaly': 'Anomaly Status'},
                opacity=0.7,
                hover_data=[target_col] if target_col != y_feature else []
            )
            fig_scatter.update_layout(
                title_font=dict(color='#ffffff', size=16),
                xaxis_title_font=dict(color='#ffffff', size=12),
                yaxis_title_font=dict(color='#ffffff', size=12),
                legend_title_font=dict(color='#ffffff'),
                legend_font=dict(color='#ffffff'),
                paper_bgcolor='#1a1a1a',
                plot_bgcolor='#2d2d2d',
                height=500
            )
            fig_scatter.update_traces(marker=dict(line=dict(width=0.5, color='#ffffff'), size=8))
            st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_plot_chart_key")
        
        elif viz_option == "Correlation Heatmap":
            st.subheader("🔥 Correlation Heatmap")
            
            numeric_cols = [c for c in data_processed.columns
                           if pd.api.types.is_numeric_dtype(data_processed[c])
                           and c != 'anomaly']
            corr = data_processed[numeric_cols].corr()
            
            fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
                aspect="auto",
                title="Correlation Heatmap of Process Variables",
                labels=dict(color="Correlation")
            )
            fig_corr.update_layout(
                title_font=dict(color='#ffffff', size=16),
                xaxis_title_font=dict(color='#ffffff', size=12),
                yaxis_title_font=dict(color='#ffffff', size=12),
                coloraxis_colorbar=dict(
                    title="Correlation",
                    title_font=dict(color='#ffffff'),
                    tickfont=dict(color='#ffffff')
                ),
                paper_bgcolor='#1a1a1a',
                plot_bgcolor='#2d2d2d',
                height=600
            )
            try:
                fig_corr.update_traces(textfont=dict(size=10, color='#ffffff'))
            except:
                pass
            st.plotly_chart(fig_corr, use_container_width=True, key="correlation_heatmap_chart_key")
        
        elif viz_option == "Distribution Comparison":
            st.subheader("📊 Distribution Comparison")
            
            selected_feature = st.selectbox(
                "Select Feature to Compare",
                [target_col] + features,
                index=0,
                key="distribution_feature_selector"
            )
            
            normal_vals = data_processed[~data_processed['anomaly']][selected_feature]
            anomaly_vals = data_processed[data_processed['anomaly']][selected_feature]
            
            fig_dist = go.Figure()
            
            # Add histogram traces
            fig_dist.add_trace(go.Histogram(
                x=normal_vals,
                name='Normal',
                opacity=0.7,
                marker_color='#2563eb',
                nbinsx=30
            ))
            fig_dist.add_trace(go.Histogram(
                x=anomaly_vals,
                name='Anomalous',
                opacity=0.7,
                marker_color='#dc2626',
                nbinsx=30
            ))
            
            fig_dist.update_layout(
                title=f"{selected_feature} Distribution Comparison",
                xaxis_title=selected_feature,
                yaxis_title="Frequency",
                title_font=dict(color='#ffffff', size=16),
                xaxis_title_font=dict(color='#ffffff', size=12),
                yaxis_title_font=dict(color='#ffffff', size=12),
                legend_title_font=dict(color='#ffffff'),
                legend_font=dict(color='#ffffff'),
                paper_bgcolor='#1a1a1a',
                plot_bgcolor='#2d2d2d',
                height=500,
                barmode='overlay'
            )
            st.plotly_chart(fig_dist, use_container_width=True, key="distribution_comparison_chart_key")
        
        elif viz_option == "Box Plot Comparison":
            st.subheader("📦 Box Plot Comparison")
            
            selected_features = st.multiselect(
                "Select Features to Compare",
                features[:10],  # Limit to first 10 for performance
                default=features[:3] if len(features) >= 3 else features,
                key="boxplot_features_selector"
            )
            
            if selected_features:
                fig_box = go.Figure()
                
                for feat in selected_features:
                    normal_vals = data_processed[~data_processed['anomaly']][feat]
                    anomaly_vals = data_processed[data_processed['anomaly']][feat]
                    
                    fig_box.add_trace(go.Box(
                        y=normal_vals,
                        name=f'{feat} (Normal)',
                        marker_color='#2563eb',
                        boxmean='sd'
                    ))
                    fig_box.add_trace(go.Box(
                        y=anomaly_vals,
                        name=f'{feat} (Anomalous)',
                        marker_color='#dc2626',
                        boxmean='sd'
                    ))
                
                fig_box.update_layout(
                    title="Box Plot Comparison: Normal vs Anomalous",
                    yaxis_title="Feature Value",
                    xaxis_title="Feature & Status",
                    title_font=dict(color='#ffffff', size=16),
                    xaxis_title_font=dict(color='#ffffff', size=12),
                    yaxis_title_font=dict(color='#ffffff', size=12),
                    legend_title_font=dict(color='#ffffff'),
                    legend_font=dict(color='#ffffff'),
                    paper_bgcolor='#1a1a1a',
                    plot_bgcolor='#2d2d2d',
                    height=500
                )
                st.plotly_chart(fig_box, use_container_width=True, key="box_plot_comparison_chart_key")
            else:
                st.info("Please select at least one feature to compare.")
        
        elif viz_option == "Feature Importance":
            st.subheader("🎯 Feature Importance Visualization")
            
            if st.session_state.importances is not None:
                importances = st.session_state.importances
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart
                    fig_bar = px.bar(
                        importances.head(15),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Top 15 Feature Importances",
                        labels={'importance': 'Importance', 'feature': 'Feature'},
                        color='importance',
                        color_continuous_scale='Blues',
                        error_y='std'
                    )
                    fig_bar.update_layout(
                        height=500,
                        xaxis_title="Importance Score",
                        yaxis=dict(
                            categoryorder='total ascending',
                            title_font=dict(color='#ffffff', size=12),
                            tickfont=dict(color='#ffffff', size=10)
                        ),
                        title_font=dict(color='#ffffff', size=16),
                        paper_bgcolor='#1a1a1a',
                        plot_bgcolor='#2d2d2d',
                        coloraxis_colorbar=dict(
                            title="Importance",
                            title_font=dict(color='#ffffff'),
                            tickfont=dict(color='#ffffff')
                        )
                    )
                    fig_bar.update_traces(
                        texttemplate='%{x:.3f}',
                        textposition='outside',
                        textfont=dict(color='#ffffff', size=10)
                    )
                    st.plotly_chart(fig_bar, use_container_width=True, key="feature_importance_viz_chart_key")
                
                with col2:
                    # Pie chart for top features
                    top_n = st.slider("Top Features", 2, 15, 10, key="top_n_features_slider")
                    top_features = importances.head(top_n)
                    
                    fig_pie = px.pie(
                        top_features,
                        values='importance',
                        names='feature',
                        title="Top Features by Importance",
                        hole=0.4
                    )
                    fig_pie.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        textfont=dict(color='#ffffff', size=10)
                    )
                    fig_pie.update_layout(
                        height=500,
                        title_font=dict(color='#ffffff', size=16),
                        legend_title_font=dict(color='#ffffff'),
                        legend_font=dict(color='#ffffff'),
                        paper_bgcolor='#1a1a1a',
                        plot_bgcolor='#2d2d2d'
                    )
                    st.plotly_chart(fig_pie, use_container_width=True, key="feature_importance_pie_chart_key")
            else:
                st.info("Feature importances not available.")

# ============================================================================
# Root Cause Analysis Tab
# ============================================================================

with tab4:
    st.header("🎯 Root Cause Analysis")
    
    if not st.session_state.analysis_run:
        st.info("👆 Please run analysis from the Dashboard tab first.")
    else:
        data_processed = st.session_state.data_processed
        target_col = st.session_state.target_col
        features = st.session_state.features
        
        # Use the threshold that was selected during analysis run
        threshold_to_use = st.session_state.final_threshold if st.session_state.final_threshold is not None else analysis_threshold
        
        # Generate analysis
        stats_comparison, insight_text, root_cause_features = generate_analysis(
            data_processed,
            features,
            target_col,
            threshold=threshold_to_use
        )
        
        if stats_comparison is not None:
            if root_cause_features:
                st.subheader("🔍 Top Root Cause Indicators")
                
                root_cause_df = stats_comparison.loc[root_cause_features].copy()
                
                # Enhanced display with more information
                root_cause_display = pd.DataFrame({
                    'Feature': root_cause_df.index,
                    'Normal Mean': root_cause_df['Normal_Mean'].values,
                    'Anomaly Mean': root_cause_df['Anomaly_Mean'].values,
                    'Percent Difference (%)': root_cause_df['Percent_Difference (%)'].values,
                    'Effect Size': root_cause_df['Effect_Size'].values,
                    'P Value': root_cause_df['P_Value'].values
                })
                
                # Format values
                root_cause_display['Normal Mean'] = root_cause_display['Normal Mean'].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
                )
                root_cause_display['Anomaly Mean'] = root_cause_display['Anomaly Mean'].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
                )
                root_cause_display['P Value'] = root_cause_display['P Value'].apply(
                    lambda x: f"{x:.4f}" if x >= 0.0001 else f"{x:.5f}".rstrip('0').rstrip('.')
                )
                root_cause_display['Effect Size'] = root_cause_display['Effect Size'].apply(
                    lambda x: f"{x:.2f}"
                )
                root_cause_display['Percent Difference (%)'] = root_cause_display['Percent Difference (%)'].apply(
                    lambda x: f"{x:.2f}"
                )
                
                # Style the root cause table with red/green backgrounds
                def style_root_cause_row(row):
                    styles = pd.Series('', index=row.index)
                    percent_diff_col = 'Percent Difference (%)'
                    
                    if percent_diff_col in row.index:
                        try:
                            percent_diff = float(row[percent_diff_col])
                            # Red for negative, green for positive (both are significant root causes)
                            if percent_diff < 0:
                                styles[percent_diff_col] = 'background-color: #7f1d1d; color: #ffffff; font-weight: bold;'
                            else:
                                styles[percent_diff_col] = 'background-color: #14532d; color: #ffffff; font-weight: bold;'
                        except:
                            pass
                    
                    # Bold all important columns
                    for col in ['Feature', 'Percent Difference (%)', 'Effect Size', 'P Value']:
                        if col in styles.index:
                            styles[col] = (styles.get(col, '') + ' font-weight: bold;').strip()
                    
                    # Ensure all text is white
                    for col in styles.index:
                        if 'color' not in styles[col]:
                            styles[col] = (styles[col] + ' color: #ffffff;').strip()
                    
                    return styles
                
                styled_root_cause = root_cause_display.style.apply(style_root_cause_row, axis=1).set_properties(**{
                    'background-color': '#2d2d2d',
                    'color': '#ffffff'
                })
                
                st.dataframe(styled_root_cause, use_container_width=True)
                
                # Summary Statistics Section
                st.divider()
                st.subheader("📊 Root Cause Summary Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Root Causes Identified", len(root_cause_features))
                
                with col2:
                    avg_effect = root_cause_df['Effect_Size'].abs().mean()
                    st.metric("Avg Effect Size", f"{avg_effect:.2f}")
                
                with col3:
                    max_effect_feature = root_cause_df['Effect_Size'].abs().idxmax()
                    max_effect = root_cause_df.loc[max_effect_feature, 'Effect_Size']
                    st.metric("Max Effect Size", f"{abs(max_effect):.2f}", help=f"Feature: {max_effect_feature}")
                
                with col4:
                    avg_pct_diff = root_cause_df['Percent_Difference (%)'].abs().mean()
                    st.metric("Avg % Difference", f"{avg_pct_diff:.2f}%")
                
                # Actionable insights with more details
                st.divider()
                st.subheader("💡 Actionable Insights & Recommendations")
                
                top_feature = root_cause_features[0]
                top_stats = root_cause_df.loc[top_feature]
                
                col_insight1, col_insight2 = st.columns(2)
                
                with col_insight1:
                    p_val = top_stats['P_Value']
                    p_val_str = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
                    st.success(
                        f"**🔴 Primary Root Cause: {top_feature}**\n\n"
                        f"**Statistics:**\n"
                        f"- Normal Mean: {top_stats['Normal_Mean']:.2f}\n"
                        f"- Anomaly Mean: {top_stats['Anomaly_Mean']:.2f}\n"
                        f"- Percent Difference: {top_stats['Percent_Difference (%)']:.2f}%\n"
                        f"- Effect Size: {top_stats['Effect_Size']:.2f}\n"
                        f"- P-Value: {p_val_str}\n\n"
                        f"**Impact:** This feature shows the strongest deviation between normal and anomalous runs."
                    )
                
                with col_insight2:
                    if len(root_cause_features) > 1:
                        st.info(
                            f"**📋 Additional Root Causes:**\n\n"
                            + "\n".join([f"• **{feat}**: {root_cause_df.loc[feat, 'Percent_Difference (%)']:.2f}% difference, "
                                       f"Effect Size = {root_cause_df.loc[feat, 'Effect_Size']:.2f}" 
                                       for feat in root_cause_features[1:3]])
                        )
                
                # Detailed breakdown for each root cause
                st.divider()
                st.subheader("📈 Detailed Root Cause Breakdown")
                
                for idx, feat in enumerate(root_cause_features[:5], 1):
                    with st.expander(f"**{idx}. {feat}** - Detailed Analysis", expanded=(idx == 1)):
                        feat_stats = root_cause_df.loc[feat]
                        normal_data = data_processed[~data_processed['anomaly']][feat]
                        anomaly_data = data_processed[data_processed['anomaly']][feat]
                        
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        
                        with col_stat1:
                            st.write("**Normal Runs:**")
                            st.write(f"Mean: {normal_data.mean():.2f}")
                            st.write(f"Std: {normal_data.std():.2f}")
                            st.write(f"Median: {normal_data.median():.2f}")
                            st.write(f"Min: {normal_data.min():.2f}")
                            st.write(f"Max: {normal_data.max():.2f}")
                        
                        with col_stat2:
                            st.write("**Anomalous Runs:**")
                            st.write(f"Mean: {anomaly_data.mean():.2f}")
                            st.write(f"Std: {anomaly_data.std():.2f}")
                            st.write(f"Median: {anomaly_data.median():.2f}")
                            st.write(f"Min: {anomaly_data.min():.2f}")
                            st.write(f"Max: {anomaly_data.max():.2f}")
                        
                        with col_stat3:
                            st.write("**Impact Metrics:**")
                            st.write(f"Effect Size: {feat_stats['Effect_Size']:.2f}")
                            p_val = feat_stats['P_Value']
                            p_val_str = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
                            st.write(f"P-Value: {p_val_str}")
                            st.write(f"% Difference: {feat_stats['Percent_Difference (%)']:.2f}%")
                        
                        # Visual comparison
                        fig_detail = go.Figure()
                        fig_detail.add_trace(go.Box(
                            y=normal_data,
                            name='Normal',
                            marker_color='#2563eb',
                            boxmean='sd'
                        ))
                        fig_detail.add_trace(go.Box(
                            y=anomaly_data,
                            name='Anomalous',
                            marker_color='#dc2626',
                            boxmean='sd'
                        ))
                        fig_detail.update_layout(
                            title=f"{feat} Distribution Comparison",
                            yaxis_title=feat,
                            title_font=dict(color='#ffffff', size=14),
                            xaxis_title_font=dict(color='#ffffff', size=12),
                            yaxis_title_font=dict(color='#ffffff', size=12),
                            legend_title_font=dict(color='#ffffff'),
                            legend_font=dict(color='#ffffff'),
                            paper_bgcolor='#1a1a1a',
                            plot_bgcolor='#2d2d2d',
                            height=350
                        )
                        st.plotly_chart(fig_detail, use_container_width=True, key=f"root_cause_detail_{idx}_chart")
            else:
                st.warning(
                    "No root cause indicators identified. This may indicate:\n"
                    "- Anomalies are not statistically significant\n"
                    "- Multiple interacting factors\n"
                    "- Random variation\n\n"
                    "Try adjusting the analysis threshold or reviewing the data quality."
                )

# ============================================================================
# Predictions Tab
# ============================================================================

with tab5:
    st.header("🔮 Yield Predictions & What-If Analysis")
    
    if not st.session_state.analysis_run:
        st.info("👆 Please run analysis from the Dashboard tab first.")
    else:
        rf_model = st.session_state.rf_model
        features = st.session_state.features
        target_col = st.session_state.target_col
        data_processed = st.session_state.data_processed
        
        st.subheader("🎛️ Adjust Process Parameters")
        st.caption("Modify process parameters below to see predicted yield outcomes")
        
        # Get feature ranges with better organization
        feature_values = {}
        n_cols = 3
        cols = st.columns(n_cols)
        
        for i, feat in enumerate(features):
            col_idx = i % n_cols
            with cols[col_idx]:
                min_val = float(data_processed[feat].min())
                max_val = float(data_processed[feat].max())
                mean_val = float(data_processed[feat].mean())
                std_val = float(data_processed[feat].std())
                
                feature_values[feat] = st.slider(
                    feat,
                    min_val, max_val, mean_val,
                    step=(max_val - min_val) / 100,
                    key=f"slider_{feat}_prediction",
                    help=f"Range: [{min_val:.2f}, {max_val:.2f}], Mean: {mean_val:.2f}, Std: {std_val:.2f}"
                )
        
        if st.button("🔮 Predict Yield", type="primary", key="predict_button"):
            predicted_yield, contributions = predict_yield(rf_model, feature_values, features)
            
            st.subheader("📊 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Predicted Yield",
                    f"{predicted_yield:.2f}",
                    help=f"Predicted {target_col} based on current parameter settings"
                )
                
                historical_mean = float(data_processed[target_col].mean())
                diff = predicted_yield - historical_mean
                diff_pct = (diff / historical_mean) * 100 if historical_mean != 0 else 0
                
                st.metric(
                    "vs Historical Mean",
                    f"{diff:+.2f}",
                    f"{diff_pct:+.1f}%",
                    delta_color="normal" if diff > 0 else "inverse"
                )
            
            with col2:
                st.subheader("Top Feature Contributions")
                contributions_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': [contributions.get(f, 0) for f in features]
                }).sort_values('Importance', ascending=False)
                
                fig_contrib = px.bar(
                    contributions_df.head(10),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Feature Contributions",
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                fig_contrib.update_layout(
                    height=300,
                    title_font=dict(color='#ffffff'),
                    xaxis_title_font=dict(color='#ffffff'),
                    yaxis_title_font=dict(color='#ffffff'),
                    paper_bgcolor='#1a1a1a',
                    plot_bgcolor='#2d2d2d'
                )
                fig_contrib.update_traces(textfont=dict(color='#ffffff'))
                st.plotly_chart(fig_contrib, use_container_width=True, key="contributions_chart_key")

# ============================================================================
# Export Tab
# ============================================================================

with tab6:
    st.header("💾 Export & Reports")
    
    if not st.session_state.analysis_run:
        st.info("👆 Please run analysis from the Dashboard tab first.")
    else:
        data_processed = st.session_state.data_processed
        metrics = st.session_state.metrics
        importances = st.session_state.importances
        target_col = st.session_state.target_col
        features = st.session_state.features
        
        # Export processed dataset
        st.subheader("📥 Download Processed Dataset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = data_processed.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download CSV",
                csv,
                "processed_data.csv",
                "text/csv",
                key="download_csv_button",
                help="Download the processed dataset with anomaly labels"
            )
        
        with col2:
            # Export as JSON
            json_data = data_processed.to_json(orient='records', indent=2)
            st.download_button(
                "📥 Download JSON",
                json_data.encode("utf-8"),
                "processed_data.json",
                "application/json",
                key="download_json_button",
                help="Download the processed dataset in JSON format"
            )
        
        # Summary Report
        st.divider()
        st.subheader("📄 Comprehensive Summary Report")
        
        # Generate enhanced report (plain text format)
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("PROCESS YIELD ANOMALY DETECTION REPORT")
        report_lines.append("=" * 70)
        report_lines.append("")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Application Version: {APP_VERSION}")
        report_lines.append(f"Dataset: {len(data_processed)} records")
        report_lines.append(f"Target Variable: {target_col}")
        report_lines.append(f"Features Analyzed: {len(features)}")
        threshold_used = st.session_state.final_threshold if st.session_state.final_threshold is not None else analysis_threshold
        contamination_used = calculated_contamination
        report_lines.append(f"Analysis Threshold: {threshold_used:.2f}%")
        report_lines.append(f"Contamination Rate: {contamination_used:.2%}")
        report_lines.append("")
        report_lines.append("=" * 70)
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("=" * 70)
        anomaly_count = int(metrics.get('anomaly_count', 0))
        anomaly_rate = metrics.get('anomaly_rate', 0)
        report_lines.append(
            f"This analysis identified {anomaly_count} anomalies ({anomaly_rate:.1%}) "
            f"out of {len(data_processed)} total records."
        )
        
        yield_sig = metrics.get('yield_statistically_significant', False)
        if yield_sig:
            report_lines.append("The yield difference between normal and anomalous runs is STATISTICALLY SIGNIFICANT.")
        else:
            report_lines.append("The yield difference between normal and anomalous runs is NOT statistically significant.")
        report_lines.append("")
        
        report_lines.append("=" * 70)
        report_lines.append("MODEL PERFORMANCE METRICS")
        report_lines.append("=" * 70)
        report_lines.append(f"R² Score: {metrics.get('r2', 0):.4f}")
        report_lines.append(f"R² (CV Mean ± Std): {metrics.get('r2_cv_mean', 0):.4f} ± {metrics.get('r2_cv_std', 0):.4f}")
        report_lines.append(f"MSE: {metrics.get('mse', 0):.4f}")
        report_lines.append(f"MAE: {metrics.get('mae', 0):.4f}")
        report_lines.append(f"MAPE: {metrics.get('mape', 0):.4f}%")
        report_lines.append("")
        
        report_lines.append("=" * 70)
        report_lines.append("ANOMALY DETECTION RESULTS")
        report_lines.append("=" * 70)
        report_lines.append(f"Anomalies Detected: {anomaly_count}")
        report_lines.append(f"Anomaly Rate: {anomaly_rate:.2%}")
        report_lines.append("")
        
        report_lines.append("=" * 70)
        report_lines.append("STATISTICAL ANALYSIS")
        report_lines.append("=" * 70)
        p_val = metrics.get('yield_p_value', 1.0)
        p_val_str = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
        report_lines.append(f"Yield Difference Statistically Significant: {'Yes' if yield_sig else 'No'}")
        report_lines.append(f"P-Value: {p_val_str}")
        report_lines.append(f"Normal Yield Mean: {metrics.get('normal_yield_mean', 0):.2f}")
        report_lines.append(f"Anomaly Yield Mean: {metrics.get('anomaly_yield_mean', 0):.2f}")
        if metrics.get('yield_impact') is not None:
            report_lines.append(f"Yield Impact: {metrics.get('yield_impact', 0):.2f}")
        report_lines.append("")
        
        # Distribution metrics
        dist_metrics = calculate_distribution_metrics(data_processed, target_col)
        report_lines.append("=" * 70)
        report_lines.append("DISTRIBUTION METRICS")
        report_lines.append("=" * 70)
        report_lines.append(f"Skewness: {dist_metrics['skewness']:.3f}")
        report_lines.append(f"Kurtosis: {dist_metrics['kurtosis']:.3f}")
        report_lines.append(f"IQR: {dist_metrics['iqr']:.3f}")
        report_lines.append(f"Coefficient of Variation: {dist_metrics['cv']:.3f}")
        report_lines.append("")
        
        # Feature importance
        if importances is not None:
            report_lines.append("=" * 70)
            report_lines.append("TOP 10 MOST IMPORTANT FEATURES")
            report_lines.append("=" * 70)
            for i, row in importances.head(10).iterrows():
                report_lines.append(f"{i+1}. {row['feature']}: {row['importance']:.4f} (±{row['std']:.4f})")
            report_lines.append("")
        
        # Root cause analysis
        threshold_used = st.session_state.final_threshold if st.session_state.final_threshold is not None else analysis_threshold
        stats_comparison, insight_text, root_cause_features = generate_analysis(
            data_processed, features, target_col, threshold=threshold_used
        )
        if root_cause_features:
            report_lines.append("=" * 70)
            report_lines.append("ROOT CAUSE INDICATORS")
            report_lines.append("=" * 70)
            for feat in root_cause_features:
                effect_size = stats_comparison.loc[feat, 'Effect_Size']
                p_val = stats_comparison.loc[feat, 'P_Value']
                p_val_str = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
                percent_diff = stats_comparison.loc[feat, 'Percent_Difference (%)']
                normal_mean = stats_comparison.loc[feat, 'Normal_Mean']
                anomaly_mean = stats_comparison.loc[feat, 'Anomaly_Mean']
                report_lines.append(f"\nFeature: {feat}")
                report_lines.append(f"  Normal Mean: {normal_mean:.2f}")
                report_lines.append(f"  Anomaly Mean: {anomaly_mean:.2f}")
                report_lines.append(f"  Percent Difference: {percent_diff:.2f}%")
                report_lines.append(f"  Effect Size: {effect_size:.3f}")
                report_lines.append(f"  P-Value: {p_val_str}")
            report_lines.append("")
        
        report_lines.append("=" * 70)
        report_lines.append(f"Report generated by {PAGE_TITLE} v{APP_VERSION}")
        report_lines.append("=" * 70)
        
        report_text = "\n".join(report_lines)
        
        # Display report (render as markdown for viewing)
        with st.expander("📄 View Full Report", expanded=False):
            st.text(report_text)
        
        # Download report as TXT
        st.download_button(
            "📥 Download Report (TXT)",
            report_text.encode("utf-8"),
            f"anomaly_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "text/plain",
            key="download_report_button",
            help="Download a comprehensive text report of the analysis"
        )
