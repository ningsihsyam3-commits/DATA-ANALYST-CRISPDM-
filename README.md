# 🔍 CRISP-DM Anomaly Detection Dashboard

**Advanced Healthcare Data Analysis & Anomaly Detection System**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Project ini mengimplementasikan **CRISP-DM methodology** untuk analisis data kesehatan dan deteksi anomali menggunakan machine learning. Sistem ini dirancang untuk mengidentifikasi pola abnormal dalam data pasien rumah sakit dan memberikan insights bisnis yang actionable.

## 🎯 Tujuan Project

- **Deteksi Anomali**: Mengidentifikasi pola abnormal dalam data kesehatan pasien
- **Analisis Risiko**: Menganalisis faktor risiko readmission pasien
- **Dashboard Interaktif**: Menyediakan interface untuk eksplorasi data real-time
- **Insights Bisnis**: Menghasilkan rekomendasi untuk perbaikan layanan kesehatan

## 📊 Dataset

**Diabetes 130-Hospitals Dataset** dari UCI Machine Learning Repository
- **Sumber**: [UCI Repository ID: 296](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- **Ukuran**: ~100K records, 50+ features
- **Domain**: Healthcare, Hospital Readmissions
- **Target**: Prediksi risiko readmission pasien diabetes

## 🏗️ Arsitektur Sistem

```
CRISP-DM Anomaly Detection System
├── 📁 data/
│   ├── raw/           # Data mentah dari UCI
│   ├── processed/     # Data yang sudah dibersihkan
│   └── metadata/      # Metadata ingestion
├── 📁 src/
│   ├── preparation/
│   │   ├── data_ingestion.py    # Pipeline ingestion data
│   │   └── data_cleaning.py     # Pipeline pembersihan data
│   ├── modeling.py              # Anomaly detection models
│   └── evaluation.py            # Model evaluation metrics
├── 📁 notebooks/                # Jupyter notebooks per fase
├── 📁 reports/
│   └── dashboard/
│       └── app.py               # Streamlit dashboard
├── 📁 results/                  # Model results & evaluation
├── 📁 logs/                     # Application logs
└── 📁 docs/                     # Dokumentasi per fase
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager
- Git

### Installation

1. **Clone repository:**
   ```bash
   git clone https://github.com/your-username/DATA-ANALYST-CRISPDM-.git
   cd DATA-ANALYST-CRISPDM-
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run data pipeline:**
   ```bash
   # 1. Ingest data dari UCI
   python src/preparation/data_ingestion.py

   # 2. Clean dan preprocess data
   python src/preparation/data_cleaning.py
   ```

4. **Launch dashboard:**
   ```bash
   cd reports/dashboard
   streamlit run app.py
   ```

## 📋 Workflow CRISP-DM

### 1. 🔍 Business Understanding
- **Objective**: Mengidentifikasi pola anomali dalam data kesehatan
- **Success Criteria**: Akurasi deteksi > 85%, insights actionable
- **Business Questions**:
  - Faktor apa yang mempengaruhi readmission pasien?
  - Bagaimana mendeteksi pasien high-risk?
  - Strategi apa untuk mengurangi readmission rate?

### 2. 📊 Data Understanding
- **Data Profiling**: Statistical analysis, missing values, distributions
- **EDA**: Correlation analysis, feature importance
- **Data Quality**: Validation, consistency checks

### 3. 🛠️ Data Preparation
- **Data Ingestion**: Automated pipeline dari UCI repository
- **Data Cleaning**: Missing value handling, outlier detection
- **Feature Engineering**: Scaling, encoding, transformation

### 4. 🤖 Modeling
- **Anomaly Detection Algorithms**:
  - Z-Score Method
  - IQR (Interquartile Range)
  - Isolation Forest
  - Local Outlier Factor (LOF)
  - Mahalanobis Distance
  - Ensemble Methods
- **Model Training**: Automated hyperparameter tuning
- **Model Persistence**: Save/load trained models

### 5. 📈 Evaluation
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
- **Anomaly Metrics**: Detection rate, false positive rate
- **Cross-validation**: K-fold validation
- **Model Comparison**: Performance benchmarking

### 6. 🚀 Deployment
- **Interactive Dashboard**: Real-time anomaly detection
- **API Endpoints**: RESTful API untuk predictions
- **Monitoring**: Model performance tracking
- **Documentation**: Comprehensive system docs

## 🎛️ Dashboard Features

### 🏠 Home
- System overview dan feature highlights
- Quick statistics metrics
- Real-time status indicators

### 📊 Data Overview
- Dataset statistics (records, features, memory usage)
- Data preview dengan pagination
- Data types analysis
- Statistical summaries

### 📈 Data Visualization
- **Interactive Charts**: Plotly-powered visualizations
- **Chart Types**: Distribution, Correlation, Scatter, Box, Bar
- **Dynamic Controls**: Column selection, filtering

### 🔍 Anomaly Detection
- **Real-time Detection**: 6 algoritma deteksi anomali
- **Interactive Parameters**: Slider controls untuk tuning
- **Live Visualization**: Real-time plotting dengan anomalies highlighted
- **Detailed Metrics**: Comprehensive performance metrics

### 📋 Model Evaluation
- **Performance Metrics**: Classification & anomaly metrics
- **Confusion Matrix**: Visual confusion matrix
- **Model Comparison**: Side-by-side algorithm comparison

### 🔧 Model Configuration
- **Parameter Tuning**: Automated hyperparameter optimization
- **Method Comparison**: Multi-algorithm benchmarking
- **Optimal Settings**: Find best contamination rates

## 🛠️ Technical Stack

### Core Libraries
- **Data Processing**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`, `scipy`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Dashboard**: `streamlit`
- **Data Source**: `ucimlrepo`

### Development Tools
- **Version Control**: Git
- **Environment**: Python virtualenv
- **Documentation**: Markdown
- **Testing**: pytest (future implementation)

## 📈 Key Features

### 🔄 Automated Pipeline
- End-to-end data processing pipeline
- Automated model training dan evaluation
- Continuous integration dengan GitHub Actions

### 🎯 Multiple Algorithms
- 6 state-of-the-art anomaly detection algorithms
- Ensemble methods untuk improved accuracy
- Algorithm comparison dan benchmarking

### 📊 Interactive Dashboard
- Real-time anomaly detection
- Interactive parameter tuning
- Professional UI dengan custom styling
- Responsive design untuk desktop/mobile

### 🔍 Comprehensive Analytics
- Statistical data profiling
- Feature importance analysis
- Model performance monitoring
- Detailed evaluation reports

### 💾 Model Persistence
- Save/load trained models
- Model versioning dengan timestamps
- Configuration management
- Audit trails

## 📝 Usage Examples

### Basic Anomaly Detection
```python
from src.modeling import AnomalyDetector

# Initialize detector
detector = AnomalyDetector()

# Load data
data = pd.read_csv('data/processed/hospital_data_cleaned.csv')
target_column = data['time_in_hospital']

# Detect anomalies using Isolation Forest
anomalies, metrics = detector.detect_outliers_isolation_forest(
    target_column.values.reshape(-1, 1),
    contamination=0.1
)

print(f"Found {len(anomalies)} anomalies ({metrics['anomalies_percentage']}%)")
```

### Model Evaluation
```python
from src.evaluation import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator()

# Evaluate classification model
metrics = evaluator.evaluate_classification(y_true, y_pred, y_pred_proba)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
```

### ✅ Data File Checker Script
Untuk memastikan file yang dibutuhkan tersedia, gunakan skrip berikut:
```bash
python scripts/check_data_files.py
```
Skrip ini akan memeriksa keberadaan:
- `data/raw/hospital_data_raw.csv`
- `data/processed/hospital_data_cleaned.csv`
- `results/evaluation/evaluation_report.json`
- `results/evaluation/model_metrics_isolation_forest.json`

Jika file raw belum ada, skrip akan merekomendasikan menambahkan dataset raw terlebih dahulu lalu jalankan pipeline pembersihan data.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: UCI Machine Learning Repository
- **Methodology**: CRISP-DM framework
- **Libraries**: Open-source Python ecosystem
- **Community**: Data science dan healthcare analytics community

## 📞 Contact

**Wahyunia Ningsih Syam**
- **Email**: [your-email@example.com]
- **LinkedIn**: [your-linkedin]
- **GitHub**: [your-github]

---

**⭐ Star this repository if you find it helpful!**

*Built with ❤️ for healthcare analytics and anomaly detection*## ✅ Business Understanding Completed: Tue May  5 05:48:28 UTC 2026
## ✅ Data Understanding Completed: Tue May  5 05:48:41 UTC 2026
## ✅ Business Understanding Completed: Tue May  5 05:50:49 UTC 2026
## ✅ Data Understanding Completed: Tue May  5 05:50:57 UTC 2026
## ✅ Business Understanding Completed: Tue May  5 05:57:04 UTC 2026
## ✅ Data Understanding Completed: Tue May  5 05:57:11 UTC 2026
## ✅ Data Preparation Completed: Tue May  5 05:57:54 UTC 2026
## ✅ Modeling Completed: Tue May  5 05:58:36 UTC 2026
## ✅ Evaluation Completed: Tue May  5 05:59:21 UTC 2026
## ✅ Deployment Completed: Tue May  5 06:00:24 UTC 2026
