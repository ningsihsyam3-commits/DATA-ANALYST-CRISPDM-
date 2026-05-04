import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Add project root to sys.path so `src` imports work from Streamlit
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import custom modules
try:
    from src.modeling import AnomalyDetector
    from src.evaluation import ModelEvaluator
    from src.preparation.data_cleaning import DataCleaner
except ImportError as e:
    st.error(f"Error importing custom modules: {e}")
    st.stop()

# ====================== CONFIGURATION ======================
class DashboardConfig:
    """Configuration untuk Dashboard."""
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    RESULTS_DIR = PROJECT_ROOT / "results"
    PLOTS_DIR = PROJECT_ROOT / "reports" / "plots"
    MODELS_DIR = PROJECT_ROOT / "models"

    # Default file paths
    PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "hospital_data_cleaned.csv"
    EVALUATION_REPORT = RESULTS_DIR / "evaluation" / "evaluation_report.json"
    MODEL_METRICS = RESULTS_DIR / "evaluation" / "model_metrics_isolation_forest.json"

    def required_files(self):
        return {
            "Processed Data": self.PROCESSED_DATA_FILE,
            "Evaluation Report": self.EVALUATION_REPORT,
            "Model Metrics": self.MODEL_METRICS,
        }

    def missing_required_files(self):
        return {name: path for name, path in self.required_files().items() if not path.exists()}

    def format_missing_files(self):
        missing = self.missing_required_files()
        if not missing:
            return None
        return "\n".join([f"- {name}: {path.relative_to(self.PROJECT_ROOT)}" for name, path in missing.items()])

# ====================== UTILITY FUNCTIONS ======================
def load_processed_data():
    """Load processed data dari file."""
    config = DashboardConfig()
    if config.PROCESSED_DATA_FILE.exists():
        return pd.read_csv(config.PROCESSED_DATA_FILE)
    return None

def load_evaluation_results():
    """Load evaluation results dari file."""
    config = DashboardConfig()
    if config.EVALUATION_REPORT.exists():
        with open(config.EVALUATION_REPORT, 'r') as f:
            return json.load(f)
    return None

def load_model_metrics():
    """Load model metrics dari file."""
    config = DashboardConfig()
    if config.MODEL_METRICS.exists():
        with open(config.MODEL_METRICS, 'r') as f:
            return json.load(f)
    return None

def create_anomaly_detector():
    """Create AnomalyDetector instance."""
    return AnomalyDetector()

def create_evaluator():
    """Create ModelEvaluator instance."""
    return ModelEvaluator()

# ====================== DASHBOARD SETUP ======================
st.set_page_config(
    page_title="CRISP-DM Anomaly Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .sidebar-content {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ====================== MAIN DASHBOARD ======================
def main():
    # Header
    st.markdown('<div class="main-header">🔍 CRISP-DM Anomaly Detection Dashboard</div>', unsafe_allow_html=True)

    # Introduction
    st.markdown("""
    ### Welcome to the CRISP-DM Anomaly Detection Dashboard!

    This comprehensive dashboard showcases our **CRISP-DM methodology** implementation for anomaly detection in healthcare data.
    Explore data insights, model performance, and interactive visualizations.

    **Key Features:**
    - 📊 **Data Overview**: Explore processed healthcare data
    - 📈 **Data Visualization**: Interactive charts and plots
    - 🤖 **Anomaly Detection**: Real-time anomaly detection with multiple algorithms
    - 📋 **Model Evaluation**: Comprehensive performance metrics
    - 🔧 **Model Configuration**: Tune parameters and compare methods
    """)

    # Sidebar navigation
    st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.sidebar.header('🚀 Dashboard Navigation')

    # Load data status
    config = DashboardConfig()
    missing_files = config.missing_required_files()
    data_loaded = config.PROCESSED_DATA_FILE.exists()
    eval_loaded = config.EVALUATION_REPORT.exists() or config.MODEL_METRICS.exists()

    if data_loaded:
        st.sidebar.success("✅ Processed data loaded")
    else:
        st.sidebar.warning(f"⚠️ No processed data found. Expected: {config.PROCESSED_DATA_FILE.relative_to(config.PROJECT_ROOT)}")

    if eval_loaded:
        st.sidebar.success("✅ Evaluation results loaded")
    else:
        st.sidebar.info("ℹ️ No evaluation results found")

    if missing_files:
        st.sidebar.markdown("---")
        st.sidebar.warning("Some required files are missing. See below for expected paths.")
        for name, path in missing_files.items():
            st.sidebar.write(f"- **{name}**: `{path.relative_to(config.PROJECT_ROOT)}`")

    # Navigation options
    options = st.sidebar.radio(
        'Select a section:',
        ('🏠 Home', '📊 Data Overview', '📈 Data Visualization',
         '🔍 Anomaly Detection', '📋 Model Evaluation', '🔧 Model Configuration')
    )

    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    # Route to selected section
    if options == '🏠 Home':
        show_home()
    elif options == '📊 Data Overview':
        show_data_overview()
    elif options == '📈 Data Visualization':
        show_data_visualization()
    elif options == '🔍 Anomaly Detection':
        show_anomaly_detection()
    elif options == '📋 Model Evaluation':
        show_model_evaluation()
    elif options == '🔧 Model Configuration':
        show_model_configuration()

# ====================== HOME SECTION ======================
def show_home():
    """Display home/dashboard overview."""
    st.markdown('<div class="section-header">🏠 Dashboard Overview</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Data Processing</h4>
            <p>Advanced data cleaning and preprocessing pipeline</p>
            <ul>
                <li>Missing value handling</li>
                <li>Outlier detection</li>
                <li>Feature scaling</li>
                <li>Data validation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🤖 Anomaly Detection</h4>
            <p>Multiple algorithms for robust detection</p>
            <ul>
                <li>Z-Score method</li>
                <li>Isolation Forest</li>
                <li>Local Outlier Factor</li>
                <li>Ensemble methods</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>📈 Visualization</h4>
            <p>Interactive charts and insights</p>
            <ul>
                <li>Real-time plotting</li>
                <li>Model comparison</li>
                <li>Performance metrics</li>
                <li>Data exploration</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Quick stats
    st.markdown("### 📈 Quick Statistics")

    df = load_processed_data()
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", f"{len(df):,}")

        with col2:
            st.metric("Features", len(df.columns))

        with col3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Features", len(numeric_cols))

        with col4:
            categorical_cols = df.select_dtypes(include=['object']).columns
            st.metric("Categorical Features", len(categorical_cols))
    else:
        st.warning("No processed data available. Please run data processing pipeline first.")

# ====================== DATA OVERVIEW SECTION ======================
def show_data_overview():
    """Display data overview section."""
    st.markdown('<div class="section-header">📊 Data Overview</div>', unsafe_allow_html=True)

    # Load data
    df = load_processed_data()

    if df is None:
        st.error(f"No processed data found. Expected file at: {DashboardConfig().PROCESSED_DATA_FILE.relative_to(DashboardConfig().PROJECT_ROOT)}")
        return

    # Data upload option
    st.markdown("### 📁 Data Source")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Load Processed Data"):
            st.rerun()

    with col2:
        uploaded_file = st.file_uploader("Upload Custom CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success("Custom data loaded successfully!")

    # Dataset overview
    st.markdown("### 📋 Dataset Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", f"{len(df):,}")

    with col2:
        st.metric("Total Features", len(df.columns))

    with col3:
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory Usage", f"{memory_usage:.2f} MB")

    # Data preview
    st.markdown("### 👀 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Data types
    st.markdown("### 🔢 Data Types")
    dtypes_df = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Non-Null Count': df.notna().sum(),
        'Null Count': df.isna().sum(),
        'Null %': (df.isna().sum() / len(df) * 100).round(2)
    })
    st.dataframe(dtypes_df, use_container_width=True)

    # Statistical summary
    st.markdown("### 📊 Statistical Summary")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    else:
        st.info("No numeric columns found for statistical summary.")

# ====================== DATA VISUALIZATION SECTION ======================
def show_data_visualization():
    """Display data visualization section."""
    st.markdown('<div class="section-header">📈 Data Visualization</div>', unsafe_allow_html=True)

    df = load_processed_data()
    if df is None:
        st.error("No data available for visualization.")
        return

    # Visualization controls
    st.markdown("### 🎛️ Visualization Controls")

    col1, col2, col3 = st.columns(3)

    with col1:
        viz_type = st.selectbox(
            "Visualization Type",
            ["Distribution", "Correlation", "Scatter Plot", "Box Plot", "Bar Chart"]
        )

    with col2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_col = st.selectbox("Select Column", numeric_cols if numeric_cols else df.columns)

    with col3:
        if viz_type in ["Scatter Plot", "Correlation"]:
            second_col = st.selectbox("Second Column", numeric_cols if numeric_cols else df.columns)
        else:
            second_col = None

    # Generate visualization
    st.markdown("### 📊 Visualization")

    try:
        if viz_type == "Distribution":
            fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}",
                             marginal="box", color_discrete_sequence=['#1f77b4'])
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Correlation":
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, title="Correlation Matrix",
                              color_continuous_scale='RdBu_r', aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns for correlation matrix.")

        elif viz_type == "Scatter Plot":
            fig = px.scatter(df, x=selected_col, y=second_col,
                           title=f"{selected_col} vs {second_col}",
                           color_discrete_sequence=['#1f77b4'])
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Box Plot":
            fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}",
                        color_discrete_sequence=['#1f77b4'])
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Bar Chart":
            if df[selected_col].nunique() <= 20:  # Limit categories for readability
                value_counts = df[selected_col].value_counts().head(20)
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f"Bar Chart of {selected_col}",
                           color_discrete_sequence=['#1f77b4'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Too many unique values for bar chart. Try a different column.")

    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}")

# ====================== ANOMALY DETECTION SECTION ======================
def show_anomaly_detection():
    """Display anomaly detection section."""
    st.markdown('<div class="section-header">🔍 Anomaly Detection</div>', unsafe_allow_html=True)

    df = load_processed_data()
    if df is None:
        st.error("No data available for anomaly detection.")
        return

    # Anomaly detection controls
    st.markdown("### 🎛️ Detection Controls")

    col1, col2, col3 = st.columns(3)

    with col1:
        method = st.selectbox(
            "Detection Method",
            ["Z-Score", "IQR", "Isolation Forest", "Local Outlier Factor", "Mahalanobis", "Ensemble"]
        )

    with col2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("Tidak ada kolom numerik yang tersedia untuk deteksi anomali.")
            return
        selected_col = st.selectbox("Target Column", numeric_cols)

    with col3:
        if method == "Z-Score":
            threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1)
            params = {"threshold": threshold}
        elif method == "IQR":
            k = st.slider("IQR Multiplier", 1.0, 3.0, 1.5, 0.1)
            params = {"k": k}
        elif method == "Isolation Forest":
            contamination = st.slider("Contamination", 0.01, 0.20, 0.10, 0.01)
            params = {"contamination": contamination}
        elif method == "Local Outlier Factor":
            n_neighbors = st.slider("Number of Neighbors", 5, 50, 20)
            params = {"n_neighbors": n_neighbors}
        elif method == "Mahalanobis":
            threshold = st.slider("Distance Threshold", 1.0, 5.0, 3.0, 0.1)
            params = {"threshold": threshold}
        else:  # Ensemble
            voting_threshold = st.slider("Voting Threshold", 0.3, 0.8, 0.5, 0.1)
            params = {"voting_threshold": voting_threshold}

    # Run detection
    if st.button("🚀 Run Anomaly Detection", type="primary"):
        with st.spinner("Detecting anomalies..."):
            try:
                detector = create_anomaly_detector()
                data = df[selected_col].values

                # Run detection based on method
                if method == "Z-Score":
                    anomalies, metrics = detector.detect_outliers_zscore(data, **params)
                elif method == "IQR":
                    anomalies, metrics = detector.detect_outliers_iqr(data, **params)
                elif method == "Isolation Forest":
                    anomalies, metrics = detector.detect_outliers_isolation_forest(data.reshape(-1, 1), **params)
                elif method == "Local Outlier Factor":
                    anomalies, metrics = detector.detect_outliers_lof(data.reshape(-1, 1), **params)
                elif method == "Mahalanobis":
                    anomalies, metrics = detector.detect_outliers_mahalanobis(data.reshape(-1, 1), **params)
                else:  # Ensemble
                    anomalies, metrics = detector.detect_outliers_ensemble(data, **params)

                # Display results
                st.markdown("### 📊 Detection Results")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Data Points", f"{len(data):,}")

                with col2:
                    st.metric("Anomalies Found", f"{len(anomalies):,}")

                with col3:
                    percentage = metrics.get('anomalies_percentage', metrics.get('percentage', 0.0))
                    st.metric("Anomaly %", f"{percentage:.2f}%")

                with col4:
                    normal_count = len(data) - len(anomalies)
                    st.metric("Normal Points", f"{normal_count:,}")

                # Visualization
                st.markdown("### 📈 Anomaly Visualization")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(data))),
                    y=data,
                    mode='lines',
                    name='Data',
                    line=dict(color='#1f77b4', width=1)
                ))

                if len(anomalies) > 0:
                    fig.add_trace(go.Scatter(
                        x=anomalies,
                        y=data[anomalies],
                        mode='markers',
                        name='Anomalies',
                        marker=dict(color='red', size=8, symbol='x')
                    ))

                fig.update_layout(
                    title=f"Anomaly Detection - {method}",
                    xaxis_title="Index",
                    yaxis_title=selected_col,
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                # Detailed metrics
                st.markdown("### 📋 Detailed Metrics")
                metrics_df = pd.DataFrame([metrics])
                st.dataframe(metrics_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error during anomaly detection: {str(e)}")

# ====================== MODEL EVALUATION SECTION ======================
def show_model_evaluation():
    """Display model evaluation section."""
    st.markdown('<div class="section-header">📋 Model Evaluation</div>', unsafe_allow_html=True)

    eval_results = load_evaluation_results()
    model_metrics = load_model_metrics()

    if eval_results is None and model_metrics is None:
        config = DashboardConfig()
        st.warning("No evaluation results found. Please run model evaluation first.")
        st.info("Expected evaluation files:")
        st.write(f"- {config.EVALUATION_REPORT.relative_to(config.PROJECT_ROOT)}")
        st.write(f"- {config.MODEL_METRICS.relative_to(config.PROJECT_ROOT)}")
        return

    # Display evaluation results
    if eval_results:
        st.markdown("### 📊 Evaluation Results")

        results = eval_results.get('results', {})

        # Classification metrics
        if 'classification' in results:
            st.markdown("#### 🤖 Classification Metrics")
            class_metrics = results['classification']

            accuracy = class_metrics.get('accuracy', class_metrics.get('acc', 0.0))
            precision = class_metrics.get('precision', 0.0)
            recall = class_metrics.get('recall', 0.0)
            f1_score = class_metrics.get('f1_score', class_metrics.get('f1', 0.0))

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")

            with col2:
                st.metric("Precision", f"{precision:.4f}")

            with col3:
                st.metric("Recall", f"{recall:.4f}")

            with col4:
                st.metric("F1-Score", f"{f1_score:.4f}")
        if 'anomalies' in results:
            st.markdown("#### 🔍 Anomaly Detection Metrics")
            anomaly_metrics = results['anomalies']

            total_data = anomaly_metrics.get('total_data', 0)
            anomalies_found = anomaly_metrics.get('anomalies_found', len(anomaly_metrics.get('anomalies', [])))
            anomaly_percentage = anomaly_metrics.get('percentage', anomaly_metrics.get('anomalies_percentage', 0.0))
            normal_data = anomaly_metrics.get('normal_data', total_data - anomalies_found)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Data", f"{total_data:,}")

            with col2:
                st.metric("Anomalies Found", f"{anomalies_found:,}")

            with col3:
                st.metric("Anomaly %", f"{anomaly_percentage:.2f}%")

            with col4:
                st.metric("Normal Data", f"{normal_data:,}")

    # Display model-specific metrics
    if model_metrics:
        st.markdown("### 🤖 Model-Specific Metrics")
        st.json(model_metrics)

# ====================== MODEL CONFIGURATION SECTION ======================
def show_model_configuration():
    """Display model configuration section."""
    st.markdown('<div class="section-header">🔧 Model Configuration</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎛️ Model Tuning & Comparison

    This section allows you to tune model parameters and compare different anomaly detection methods.
    """)

    df = load_processed_data()
    if df is None:
        st.error("No data available for model configuration.")
        return

    # Method comparison
    st.markdown("### 📊 Method Comparison")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.warning("Tidak ada kolom numerik yang tersedia untuk konfigurasi model.")
        return

    selected_col = st.selectbox("Select Column for Comparison", numeric_cols)

    methods_to_compare = st.multiselect(
        "Select Methods to Compare",
        ["Z-Score", "IQR", "Isolation Forest", "Local Outlier Factor", "Mahalanobis"],
        default=["Z-Score", "IQR", "Isolation Forest"]
    )

    if st.button("🔍 Compare Methods", type="primary") and methods_to_compare:
        with st.spinner("Comparing methods..."):
            try:
                detector = create_anomaly_detector()
                data = df[selected_col].values

                comparison_results = []

                for method in methods_to_compare:
                    if method == "Z-Score":
                        anomalies, metrics = detector.detect_outliers_zscore(data)
                    elif method == "IQR":
                        anomalies, metrics = detector.detect_outliers_iqr(data)
                    elif method == "Isolation Forest":
                        anomalies, metrics = detector.detect_outliers_isolation_forest(data.reshape(-1, 1))
                    elif method == "Local Outlier Factor":
                        anomalies, metrics = detector.detect_outliers_lof(data.reshape(-1, 1))
                    elif method == "Mahalanobis":
                        anomalies, metrics = detector.detect_outliers_mahalanobis(data.reshape(-1, 1))

                    comparison_results.append({
                        'Method': method,
                        'Anomalies': len(anomalies),
                        'Percentage': metrics['anomalies_percentage']
                    })

                # Display comparison table
                comparison_df = pd.DataFrame(comparison_results)
                st.dataframe(comparison_df, use_container_width=True)

                # Visualization
                fig = px.bar(comparison_df, x='Method', y='Anomalies',
                           title="Anomaly Detection Methods Comparison",
                           color='Method', color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error during method comparison: {str(e)}")

    # Parameter tuning for Isolation Forest
    st.markdown("### 🔧 Parameter Tuning - Isolation Forest")

    contamination_range = st.slider("Contamination Range", 0.01, 0.30, (0.05, 0.20), 0.01)

    if st.button("🎯 Find Optimal Contamination", type="secondary"):
        with st.spinner("Finding optimal contamination..."):
            try:
                detector = create_anomaly_detector()
                data = df[selected_col].values.reshape(-1, 1)

                optimal_results = detector.identify_optimal_contamination(
                    data,
                    contamination_range=contamination_range,
                    step=0.01
                )

                # Display results
                results_df = pd.DataFrame([
                    {'Contamination': k, 'Anomalies': v}
                    for k, v in optimal_results.items()
                ])

                st.dataframe(results_df, use_container_width=True)

                # Plot
                fig = px.line(results_df, x='Contamination', y='Anomalies',
                            title="Anomaly Count vs Contamination Rate",
                            markers=True)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error during parameter tuning: {str(e)}")

# ====================== RUN DASHBOARD ======================
if __name__ == "__main__":
    main()