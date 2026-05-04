import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, Dict, List, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ====================== KONFIGURASI ======================
RAW_DATA_FILE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../data/raw/hospital_data_raw.csv"
))
PROCESSED_DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../data/processed"
))
LOGS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../logs"
))

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ====================== SETUP LOGGING ======================
def setup_logging(log_file: str = None) -> logging.Logger:
    """Setup logger dengan file dan console handler."""
    log_file = log_file or os.path.join(LOGS_DIR, "data_cleaning.log")
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = setup_logging()

# ====================== DATA CLEANING CLASSES ======================
class DataCleaner:
    """Class untuk menangani pembersihan dan transformasi data secara comprehensive."""
    
    COLS_TO_KEEP = [
        'encounter_id', 'patient_nbr', 'age', 'time_in_hospital',
        'num_lab_procedures', 'num_procedures', 'num_medications',
        'number_diagnoses', 'race', 'gender', 'readmitted'
    ]
    
    NUMERIC_COLS = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_diagnoses'
    ]
    
    CATEGORICAL_COLS = ['race', 'gender', 'readmitted']
    
    def __init__(self, outlier_threshold: float = 3.0):
        """
        Initialize DataCleaner.
        
        Args:
            outlier_threshold: Z-score threshold untuk deteksi outliers (default: 3.0)
        """
        self.outlier_threshold = outlier_threshold
        self.cleaning_report = {}
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Membaca data dengan error handling."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File tidak ditemukan: {file_path}")
        
        logger.info(f"Membaca data dari: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Data berhasil dibaca. Shape: {df.shape}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values dengan strategi yang berbeda per kolom."""
        logger.info("Memulai penanganan missing values...")
        
        df_cleaned = df.copy()
        missing_before = df_cleaned.isnull().sum().sum()
        
        # Replace '?' dengan NaN
        df_cleaned.replace('?', np.nan, inplace=True)
        
        # Strategy 1: Drop rows untuk ID columns
        df_cleaned = df_cleaned.dropna(subset=['encounter_id', 'patient_nbr'])
        
        # Strategy 2: Fill dengan median untuk numeric columns
        for col in self.NUMERIC_COLS:
            if col in df_cleaned.columns and df_cleaned[col].isnull().any():
                median_val = df_cleaned[col].median()
                df_cleaned[col].fillna(median_val, inplace=True)
                logger.debug(f"  {col}: diisi dengan median {median_val}")
        
        # Strategy 3: Fill dengan mode untuk categorical columns
        for col in self.CATEGORICAL_COLS:
            if col in df_cleaned.columns and df_cleaned[col].isnull().any():
                mode_val = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown'
                df_cleaned[col].fillna(mode_val, inplace=True)
                logger.debug(f"  {col}: diisi dengan mode {mode_val}")
        
        missing_after = df_cleaned.isnull().sum().sum()
        self.cleaning_report['missing_values'] = {
            'before': missing_before,
            'after': missing_after,
            'rows_dropped': len(df) - len(df_cleaned)
        }
        
        logger.info(f"Missing values: {missing_before} → {missing_after}")
        
        return df_cleaned
    
    def select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Memilih kolom yang relevan."""
        logger.info("Memilih kolom penting...")
        
        available_cols = [col for col in self.COLS_TO_KEEP if col in df.columns]
        logger.info(f"Kolom yang dipilih: {available_cols}")
        
        return df[available_cols].copy()
    
    def transform_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform kolom age dengan error handling yang lebih baik."""
        logger.info("Transformasi kolom 'age'...")
        
        df_transformed = df.copy()
        
        if 'age' not in df_transformed.columns:
            logger.warning("Kolom 'age' tidak ditemukan")
            return df_transformed
        
        def extract_age(val):
            try:
                if pd.isna(val):
                    return np.nan
                # Extract numeric value dari string
                match = pd.Series(val).str.extract(r'(\d+)', expand=False)
                return float(match.iloc[0]) if not pd.isna(match.iloc[0]) else np.nan
            except:
                return np.nan
        
        df_transformed['age'] = df_transformed['age'].apply(extract_age)
        
        # Validasi range age
        df_transformed['age'] = df_transformed['age'].clip(lower=0, upper=120)
        
        logger.info(f"Age range: {df_transformed['age'].min()} - {df_transformed['age'].max()}")
        
        return df_transformed
    
    def detect_and_handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Deteksi outliers menggunakan Z-score dan IQR method."""
        logger.info("Deteksi outliers...")
        
        df_cleaned = df.copy()
        outlier_info = {}
        
        for col in self.NUMERIC_COLS:
            if col not in df_cleaned.columns:
                continue
            
            # Z-score method
            z_scores = np.abs(stats.zscore(df_cleaned[col].dropna()))
            outlier_indices = np.where(z_scores > self.outlier_threshold)[0]
            
            # IQR method
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = df_cleaned[(df_cleaned[col] < lower_bound) | 
                                      (df_cleaned[col] > upper_bound)].index.tolist()
            
            # Flag outliers tanpa drop (untuk transparansi)
            df_cleaned[f'{col}_is_outlier'] = False
            df_cleaned.loc[iqr_outliers, f'{col}_is_outlier'] = True
            
            outlier_info[col] = {
                'z_score_outliers': len(outlier_indices),
                'iqr_outliers': len(iqr_outliers),
                'bounds': (lower_bound, upper_bound)
            }
            
            logger.debug(f"  {col}: {len(iqr_outliers)} outliers terdeteksi (bounds: {lower_bound:.2f} - {upper_bound:.2f})")
        
        self.cleaning_report['outliers'] = outlier_info
        
        return df_cleaned, outlier_info
    
    def standardize_numeric_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
        """Standardisasi features numeric untuk analisis lebih lanjut."""
        logger.info("Standardisasi features numeric...")
        
        df_scaled = df.copy()
        scaler = StandardScaler()
        
        numeric_cols_present = [col for col in self.NUMERIC_COLS if col in df.columns]
        
        if numeric_cols_present:
            df_scaled[numeric_cols_present] = scaler.fit_transform(df[numeric_cols_present])
            logger.debug(f"Kolom yang distandardisasi: {numeric_cols_present}")
        
        return df_scaled, scaler
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        logger.info("Encoding categorical features...")
        
        df_encoded = df.copy()
        
        for col in self.CATEGORICAL_COLS:
            if col in df_encoded.columns:
                unique_values = df_encoded[col].nunique()
                if unique_values <= 10:  # One-hot encode jika <= 10 categories
                    dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=False)
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                    logger.debug(f"  {col}: One-hot encoded ({unique_values} categories)")
        
        return df_encoded
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Validasi kualitas data dengan berbagai metrics."""
        logger.info("Validasi kualitas data...")
        
        quality_metrics = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'missing_values_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        }
        
        for col in self.NUMERIC_COLS:
            if col in df.columns:
                quality_metrics[f'{col}_std'] = df[col].std()
                quality_metrics[f'{col}_cv'] = (df[col].std() / abs(df[col].mean())) if df[col].mean() != 0 else 0
        
        logger.info(f"Data quality metrics:")
        for key, val in quality_metrics.items():
            logger.info(f"  {key}: {val}")
        
        self.cleaning_report['quality_metrics'] = quality_metrics
        
        return quality_metrics
    
    def clean_and_transform(self, input_file: str, output_file: str = None, 
                           standardize: bool = True, encode: bool = True) -> pd.DataFrame:
        """Main function untuk pembersihan dan transformasi data."""
        logger.info("=" * 80)
        logger.info("MEMULAI PROSES PEMBERSIHAN DATA")
        logger.info("=" * 80)
        
        try:
            # Load data
            df = self.load_data(input_file)
            
            # Cleaning pipeline
            df = self.handle_missing_values(df)
            df = self.select_columns(df)
            df = self.transform_age(df)
            df, outlier_info = self.detect_and_handle_outliers(df)
            
            # Transformasi
            if standardize:
                df_scaled, _ = self.standardize_numeric_features(df)
                logger.info("Data telah distandardisasi")
            
            if encode:
                df = self.encode_categorical_features(df)
                logger.info("Categorical features telah di-encode")
            
            # Validasi
            quality_metrics = self.validate_data_quality(df)
            
            # Validasi final
            if df.empty:
                raise ValueError("Data bersih menghasilkan dataframe kosong!")
            
            # Save results
            if output_file:
                df.to_csv(output_file, index=False)
                logger.info(f"Data bersih disimpan ke: {output_file}")
                
                # Save quality report
                report_file = output_file.replace('.csv', '_quality_report.txt')
                self._save_report(report_file)
            
            logger.info("=" * 80)
            logger.info("PEMBERSIHAN DATA SELESAI")
            logger.info("=" * 80)
            
            return df
            
        except Exception as e:
            logger.error(f"Error selama pembersihan data: {str(e)}", exc_info=True)
            raise
    
    def _save_report(self, file_path: str):
        """Simpan quality report ke file."""
        with open(file_path, 'w') as f:
            f.write("DATA CLEANING QUALITY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            for section, content in self.cleaning_report.items():
                f.write(f"\n{section.upper()}\n")
                f.write("-" * 80 + "\n")
                if isinstance(content, dict):
                    for key, val in content.items():
                        f.write(f"  {key}: {val}\n")
                else:
                    f.write(f"  {content}\n")
        
        logger.info(f"Quality report disimpan ke: {file_path}")


# ====================== MAIN EXECUTION ======================
def clean_data():
    """Main function untuk eksekusi pembersihan data."""
    try:
        cleaner = DataCleaner(outlier_threshold=3.0)
        
        output_file = os.path.join(PROCESSED_DATA_DIR, "hospital_data_cleaned.csv")
        
        df_cleaned = cleaner.clean_and_transform(
            input_file=RAW_DATA_FILE,
            output_file=output_file,
            standardize=True,
            encode=True
        )
        
        logger.info(f"✓ Pembersihan data berhasil! Final shape: {df_cleaned.shape}")
        
    except Exception as e:
        logger.error(f"✗ Pembersihan data gagal: {str(e)}")
        raise

if __name__ == "__main__":
    clean_data()
