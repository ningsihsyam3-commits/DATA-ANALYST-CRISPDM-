import pandas as pd
import os
import logging
import json
import time
from datetime import datetime
from typing import Optional, Dict, Tuple
from ucimlrepo import fetch_ucirepo
from pathlib import Path

# ====================== KONFIGURASI ======================
class DataIngestionConfig:
    """Configuration untuk Data Ingestion."""
    
    # UCI Dataset
    DATASET_ID = 296
    DATASET_NAME = "Diabetes 130-Hospitals"
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    LOGS_DIR = PROJECT_ROOT / "logs"
    METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
    
    # File names
    RAW_FILENAME = "hospital_data_raw.csv"
    METADATA_FILENAME = "ingestion_metadata.json"
    BACKUP_SUFFIX = "_backup"
    
    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    BACKOFF_FACTOR = 2
    
    # Validation
    MIN_ROWS = 1000
    MIN_COLUMNS = 5
    
    def __init__(self):
        """Initialize config dan create directories."""
        self.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.METADATA_DIR.mkdir(parents=True, exist_ok=True)

# ====================== LOGGING SETUP ======================
def setup_logging(config: DataIngestionConfig) -> logging.Logger:
    """Setup logging system."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    log_file = config.LOGS_DIR / "data_ingestion.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# ====================== DATA INGESTION CLASS ======================
class DataIngestionPipeline:
    """Pipeline untuk data ingestion dari UCI Repository dengan advanced features."""
    
    def __init__(self, config: DataIngestionConfig = None):
        """
        Initialize Data Ingestion Pipeline.
        
        Args:
            config: DataIngestionConfig object
        """
        self.config = config or DataIngestionConfig()
        self.logger = setup_logging(self.config)
        self.ingestion_metadata = {}
        self.dataset = None
        self.dataframe = None
        
    def _retry_with_backoff(self, func, max_retries: int = None, **kwargs):
        """
        Execute function dengan retry mechanism dan exponential backoff.
        
        Args:
            func: Function to execute
            max_retries: Number of retries
            **kwargs: Arguments untuk function
            
        Returns:
            Result dari function
        """
        max_retries = max_retries or self.config.MAX_RETRIES
        attempt = 0
        delay = self.config.RETRY_DELAY
        
        while attempt < max_retries:
            try:
                self.logger.info(f"Attempt {attempt + 1}/{max_retries}: {func.__name__}")
                return func(**kwargs)
            except Exception as e:
                attempt += 1
                if attempt >= max_retries:
                    self.logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                    raise
                
                self.logger.warning(
                    f"Attempt {attempt} failed: {str(e)}. "
                    f"Retrying in {delay} seconds..."
                )
                time.sleep(delay)
                delay *= self.config.BACKOFF_FACTOR
    
    def fetch_dataset(self) -> None:
        """Fetch dataset dari UCI Repository dengan retry mechanism."""
        self.logger.info("=" * 80)
        self.logger.info("MEMULAI PROSES INGESTION DATA")
        self.logger.info("=" * 80)
        
        self.logger.info(f"Dataset: {self.config.DATASET_NAME} (ID: {self.config.DATASET_ID})")
        
        try:
            self.dataset = self._retry_with_backoff(
                fetch_ucirepo,
                id=self.config.DATASET_ID
            )
            self.logger.info("✓ Dataset berhasil diunduh dari UCI Repository")
        except Exception as e:
            self.logger.error(f"✗ Gagal mengunduh dataset: {str(e)}")
            raise
    
    def validate_dataset(self) -> bool:
        """Validasi struktur dataset."""
        self.logger.info("Validasi struktur dataset...")
        
        try:
            # Check attributes
            if not hasattr(self.dataset, 'data'):
                raise ValueError("Dataset tidak memiliki attribute 'data'")
            
            if not hasattr(self.dataset.data, 'features'):
                raise ValueError("Dataset tidak memiliki 'features'")
            
            if not hasattr(self.dataset.data, 'targets'):
                raise ValueError("Dataset tidak memiliki 'targets'")
            
            features = self.dataset.data.features
            targets = self.dataset.data.targets
            
            self.logger.info(f"  Features shape: {features.shape}")
            self.logger.info(f"  Targets shape: {targets.shape}")
            
            # Check consistency
            if len(features) != len(targets):
                raise ValueError(
                    f"Inconsistent row count: features={len(features)}, targets={len(targets)}"
                )
            
            self.logger.info("✓ Validasi struktur dataset berhasil")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Validasi dataset gagal: {str(e)}")
            raise
    
    def merge_features_targets(self) -> pd.DataFrame:
        """Merge features dan targets menjadi single DataFrame."""
        self.logger.info("Merge features dan targets...")
        
        try:
            features = self.dataset.data.features
            targets = self.dataset.data.targets
            
            self.logger.debug(f"Features columns: {features.shape[1]}")
            self.logger.debug(f"Targets columns: {targets.shape[1]}")
            
            df = pd.concat([features, targets], axis=1)
            
            self.logger.info(f"✓ Merge berhasil. Shape: {df.shape}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"✗ Merge gagal: {str(e)}")
            raise
    
    def profile_data(self, df: pd.DataFrame) -> Dict:
        """Generate data profile/summary statistics."""
        self.logger.info("Profiling data...")
        
        profile = {
            'shape': df.shape,
            'rows': len(df),
            'columns': len(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'column_info': {}
        }
        
        # Column statistics
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'non_null': df[col].notna().sum(),
                'null_pct': round((df[col].isnull().sum() / len(df)) * 100, 2)
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_info.update({
                    'min': float(df[col].min()) if not df[col].empty else None,
                    'max': float(df[col].max()) if not df[col].empty else None,
                    'mean': float(df[col].mean()) if not df[col].empty else None,
                    'std': float(df[col].std()) if not df[col].empty else None,
                })
            elif df[col].dtype == 'object':
                col_info['unique_values'] = df[col].nunique()
                col_info['top_values'] = df[col].value_counts().head(3).to_dict()
            
            profile['column_info'][col] = col_info
        
        self.logger.info(f"✓ Data profile generated")
        self.logger.debug(f"  Rows: {profile['rows']}, Columns: {profile['columns']}")
        self.logger.debug(f"  Missing values total: {sum(profile['missing_values'].values())}")
        self.logger.debug(f"  Duplicates: {profile['duplicates']}")
        
        return profile
    
    def validate_data_quality(self, df: pd.DataFrame, profile: Dict) -> bool:
        """Validasi data quality."""
        self.logger.info("Validasi data quality...")
        
        try:
            # Check minimum rows
            if len(df) < self.config.MIN_ROWS:
                raise ValueError(
                    f"Insufficient rows: {len(df)} < {self.config.MIN_ROWS}"
                )
            
            # Check minimum columns
            if len(df.columns) < self.config.MIN_COLUMNS:
                raise ValueError(
                    f"Insufficient columns: {len(df.columns)} < {self.config.MIN_COLUMNS}"
                )
            
            # Check for completely empty columns
            empty_cols = [col for col, count in profile['missing_values'].items() 
                         if count == len(df)]
            if empty_cols:
                raise ValueError(f"Empty columns detected: {empty_cols}")
            
            self.logger.info("✓ Data quality validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Data quality validation failed: {str(e)}")
            raise
    
    def backup_existing_file(self) -> Optional[str]:
        """Backup file lama jika ada."""
        raw_file = self.config.RAW_DATA_DIR / self.config.RAW_FILENAME
        
        if raw_file.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{raw_file.stem}_{timestamp}{raw_file.suffix}"
            backup_path = raw_file.parent / backup_name
            
            try:
                import shutil
                shutil.copy2(raw_file, backup_path)
                self.logger.info(f"✓ Backup file lama: {backup_path}")
                return str(backup_path)
            except Exception as e:
                self.logger.warning(f"Gagal backup file: {str(e)}")
                return None
        
        return None
    
    def save_data(self, df: pd.DataFrame) -> str:
        """Simpan data ke CSV dengan compression option."""
        self.logger.info("Saving data...")
        
        try:
            output_path = self.config.RAW_DATA_DIR / self.config.RAW_FILENAME
            
            # Backup existing file
            backup_path = self.backup_existing_file()
            
            # Save new file
            df.to_csv(output_path, index=False, compression=None)
            
            self.logger.info(f"✓ Data disimpan: {output_path}")
            self.logger.info(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"✗ Gagal menyimpan data: {str(e)}")
            raise
    
    def save_metadata(self, df: pd.DataFrame, profile: Dict, output_path: str) -> None:
        """Simpan metadata ingestion."""
        self.logger.info("Saving metadata...")
        
        try:
            metadata = {
                'ingestion_timestamp': datetime.now().isoformat(),
                'dataset_id': self.config.DATASET_ID,
                'dataset_name': self.config.DATASET_NAME,
                'output_file': output_path,
                'data_shape': profile['shape'],
                'data_profile': profile,
                'backup_created': self.backup_existing_file() is not None
            }
            
            metadata_file = self.config.METADATA_DIR / self.config.METADATA_FILENAME
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.logger.info(f"✓ Metadata disimpan: {metadata_file}")
            self.ingestion_metadata = metadata
            
        except Exception as e:
            self.logger.warning(f"Gagal menyimpan metadata: {str(e)}")
    
    def run(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Execute full data ingestion pipeline.
        
        Returns:
            Tuple of (DataFrame, metadata)
        """
        try:
            # Step 1: Fetch dataset
            self.fetch_dataset()
            
            # Step 2: Validate dataset structure
            self.validate_dataset()
            
            # Step 3: Merge features and targets
            df = self.merge_features_targets()
            self.dataframe = df
            
            # Step 4: Profile data
            profile = self.profile_data(df)
            
            # Step 5: Validate data quality
            self.validate_data_quality(df, profile)
            
            # Step 6: Save data
            output_path = self.save_data(df)
            
            # Step 7: Save metadata
            self.save_metadata(df, profile, output_path)
            
            self.logger.info("=" * 80)
            self.logger.info("✓ INGESTION DATA BERHASIL SELESAI")
            self.logger.info("=" * 80)
            
            return df, self.ingestion_metadata
            
        except Exception as e:
            self.logger.error("=" * 80)
            self.logger.error(f"✗ INGESTION DATA GAGAL")
            self.logger.error("=" * 80)
            raise


# ====================== MAIN EXECUTION ======================
def ingest_data():
    """Main function untuk execute data ingestion."""
    try:
        config = DataIngestionConfig()
        pipeline = DataIngestionPipeline(config)
        df, metadata = pipeline.run()
        
    except Exception as e:
        print(f"Ingestion failed: {str(e)}")
        raise

if __name__ == "__main__":
    ingest_data()