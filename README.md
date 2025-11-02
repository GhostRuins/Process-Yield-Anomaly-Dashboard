# 🔬 Advanced Process Yield Anomaly Detection Dashboard

An interactive anomaly detection platform for chemical process yield analysis. Built with Streamlit, scikit-learn, and Plotly for comprehensive analytics and visualization.

**Author:** Hridesh Singh Chauhan  
**Purpose:** Portfolio project demonstrating data science and machine learning capabilities within chemical engineering projects.

---

## 🚀 Features

### Core Capabilities
- **Advanced Anomaly Detection**: Isolation Forest algorithm with configurable contamination rates
- **Predictive Modeling**: Random Forest regression with cross-validation
- **Statistical Analysis**: Comprehensive statistical tests (t-tests, Mann-Whitney U, effect sizes)
- **Root Cause Analysis**: Automated identification of anomaly drivers
- **Interactive Visualizations**: Multiple chart types with drill-down capabilities
- **Predictive Analytics**: What-if scenario modeling for process optimization

### Production Features
- ✅ Configuration management system
- ✅ Comprehensive error handling and logging
- ✅ Model persistence capabilities
- ✅ Data validation and quality checks
- ✅ Cross-validation for robust metrics
- ✅ Export functionality (CSV, reports)

### Metrics & Analytics
- **Model Performance**: R², MSE, MAE, MAPE, cross-validation scores with confidence intervals
- **Anomaly Metrics**: Detection rate, impact analysis, statistical significance tests
- **Distribution Metrics**: Skewness, kurtosis, IQR, coefficient of variation
- **Feature Analysis**: Importance rankings with uncertainty estimates
- **Statistical Tests**: P-values, effect sizes (Cohen's d), normality tests

---

## 📋 Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

### Installation

```bash
# Clone or download the project
cd "Lab Yield Dashboard"

# Install dependencies
pip install -r requirements.txt
```

---

## 🎯 Quick Start

1. **Launch the application:**
   ```bash
   streamlit run app.py
   ```

2. **Load data:**
   - Upload your CSV file via the sidebar, OR
   - Use the synthetic demo data option

3. **Configure parameters:**
   - Set Random Forest estimators (default: 200)
   - Configure significance threshold

4. **Run analysis:**
   - Navigate to the **Dashboard** tab
   - Click **"Run Analysis"**
   - Explore results across multiple tabs

---

## 📊 Application Structure

### Main Interface Tabs

1. **📊 Dashboard**
   - Executive summary with key metrics
   - Quick visualizations
   - Model performance indicators

2. **🔍 Analysis**
   - Detailed statistical comparisons
   - Normal vs. Anomalous run analysis
   - Feature importance rankings
   - Distribution metrics

3. **📈 Visualizations**
   - Interactive scatter plots
   - Correlation heatmaps
   - Distribution comparisons
   - Box plots for feature analysis

4. **🎯 Root Cause Analysis**
   - Top root cause indicators
   - Statistical significance analysis
   - Feature-by-feature breakdowns
   - Visual comparisons

5. **🔮 Predictions**
   - What-if scenario modeling
   - Yield prediction based on process parameters
   - Feature contribution analysis
   - Contextual comparison with historical data

6. **💾 Export**
   - Download processed datasets
   - Comprehensive summary reports

---

## 🔧 Project Architecture

```
Lab Yield Dashboard/
├── app.py                 # Main Streamlit application
├── detector_core.py       # Core anomaly detection logic
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── chemical_reaction_yield_dataset.csv  # Sample dataset
```

### Key Modules

- **`detector_core.py`**: Core detection algorithms, statistical analysis, and metrics calculation
- **`app.py`**: Streamlit interface with multi-tab dashboard
- **`config.py`**: Configuration management system for production deployment

---

## 📈 Usage Examples

### Basic Analysis

1. Load dataset (upload CSV or use synthetic data)
2. Set model parameters in sidebar
3. Run analysis from Dashboard tab
4. Review metrics and visualizations

### Root Cause Investigation

1. Run analysis first
2. Navigate to "Root Cause" tab
3. Review top indicators with statistical significance
4. Drill down into specific features
5. Export findings

### Predictive Modeling

1. Complete initial analysis
2. Navigate to "Predictions" tab
3. Adjust process parameters using sliders
4. Click "Predict Yield" to see forecasted results
5. Analyze feature contributions

---

## 📊 Data Format

Expected CSV format:
- **Target column**: `yield` or `reaction_yield_percent` (auto-detected)
- **Feature columns**: Any numeric columns (process variables)
- **Rows**: Process runs/experiments

Example columns:
```
Temperature_C, Pressure_atm, Catalyst_Concentration_mol_L, Reaction_Yield_percent
```

---

## 🎓 Technical Highlights

### Machine Learning
- **Isolation Forest**: Unsupervised anomaly detection
- **Random Forest**: Regression with feature importance
- **Cross-validation**: 5-fold CV for robust metrics
- **Statistical Testing**: Multiple hypothesis tests with multiple comparison correction

### Data Science Best Practices
- Data validation and cleaning
- Proper train-test splits
- Feature importance with uncertainty estimates
- Comprehensive error handling
- Logging and monitoring

### Software Engineering
- Modular code structure
- Configuration management
- Session state management
- Export functionality
- Robust error handling

---

## 🔬 Methodology

### Anomaly Detection
1. Data preprocessing and validation
2. Isolation Forest training with specified contamination rate
3. Anomaly scoring and classification
4. Statistical validation of detections

### Predictive Modeling
1. Feature selection (automatic)
2. Random Forest regression training
3. Cross-validation for performance estimation
4. Feature importance calculation with uncertainty

### Statistical Analysis
1. Descriptive statistics for normal vs. anomalous groups
2. Statistical significance testing (t-test or Mann-Whitney U based on normality)
3. Effect size calculation (Cohen's d)
4. Root cause identification (significant + large effect size)

---

## 🐛 Troubleshooting

### Common Issues

**"Insufficient data after cleaning"**
- Ensure dataset has at least 10 records
- Check for excessive missing values
- Verify numeric columns exist

**"No numeric feature columns found"**
- Ensure CSV has numeric process variables
- Check column names and data types

**Model performance is poor**
- Try adjusting contamination rate
- Increase number of estimators
- Check data quality and feature relevance

**Version:** 2.0  
**Last Updated:** 2024

