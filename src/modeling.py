import numpy as np
import pandas as pd
import os
import logging
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy import stats
from scipy.spatial.distance import mahalanobis

import warnings
warnings.filterwarnings('ignore')

# ====================== KONFIGURASI ======================
class ModelingConfig:
    """Configuration untuk Anomaly Detection Modeling."""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    MODELS_DIR = PROJECT_ROOT / "models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    PLOTS_DIR = PROJECT_ROOT / "reports" / "plots"
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    # File names
    MODEL_FILENAME = "anomaly_detector_{}.pkl"
    MODEL_CONFIG_FILENAME = "model_config_{}.json"
    METRICS_FILENAME = "model_metrics_{}.json"
    
    def __init__(self):
        """Initialize config dan create directories."""
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ====================== LOGGING SETUP ======================
def setup_logging(config: ModelingConfig) -> logging.Logger:
    """Setup logging system."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    logger.handlers.clear()
    
    # File handler
    log_file = config.LOGS_DIR / "modeling.log"
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

# ====================== ANOMALY DETECTION MODELS ======================
class AnomalyDetector:
    """Base class untuk anomaly detection models."""
    
    def __init__(self, config: ModelingConfig = None):
        """
        Initialize Anomaly Detector.
        
        Args:
            config: ModelingConfig object
        """
        self.config = config or ModelingConfig()
        self.logger = setup_logging(self.config)
        self.models = {}
        self.scaler = StandardScaler()
        self.training_data = None
        self.model_history = {}
        
    def detect_outliers_zscore(self, data: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, Dict]:
        """
        Deteksi anomali menggunakan Z-score method.
        
        Args:
            data: Input data
            threshold: Z-score threshold
            
        Returns:
            Tuple of (anomaly_indices, metrics)
        """
        self.logger.info(f"Detecting outliers using Z-score (threshold={threshold})...")
        
        try:
            # Ensure input is numpy array
            data = np.asarray(data).flatten()
            
            # Calculate Z-scores
            mean = np.mean(data)
            std = np.std(data)
            
            if std == 0:
                self.logger.warning("Standard deviation is zero. No outliers detected.")
                return np.array([]), {'method': 'zscore', 'anomalies': 0}
            
            z_scores = np.abs((data - mean) / std)
            anomaly_indices = np.where(z_scores > threshold)[0]
            
            metrics = {
                'method': 'zscore',
                'threshold': threshold,
                'anomalies_count': len(anomaly_indices),
                'anomalies_percentage': round((len(anomaly_indices) / len(data)) * 100, 2),
                'mean': float(mean),
                'std': float(std)
            }
            
            self.logger.info(f"✓ Found {len(anomaly_indices)} anomalies ({metrics['anomalies_percentage']}%)")
            self.model_history['zscore'] = metrics
            
            return anomaly_indices, metrics
            
        except Exception as e:
            self.logger.error(f"✗ Error in Z-score detection: {str(e)}")
            raise
    
    def detect_outliers_iqr(self, data: np.ndarray, k: float = 1.5) -> Tuple[np.ndarray, Dict]:
        """
        Deteksi anomali menggunakan Interquartile Range (IQR) method.
        
        Args:
            data: Input data
            k: IQR multiplier (default: 1.5 untuk outliers, 3.0 untuk extreme outliers)
            
        Returns:
            Tuple of (anomaly_indices, metrics)
        """
        self.logger.info(f"Detecting outliers using IQR method (k={k})...")
        
        try:
            data = np.asarray(data).flatten()
            
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - k * IQR
            upper_bound = Q3 + k * IQR
            
            anomaly_mask = (data < lower_bound) | (data > upper_bound)
            anomaly_indices = np.where(anomaly_mask)[0]
            
            metrics = {
                'method': 'iqr',
                'k_multiplier': k,
                'Q1': float(Q1),
                'Q3': float(Q3),
                'IQR': float(IQR),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'anomalies_count': len(anomaly_indices),
                'anomalies_percentage': round((len(anomaly_indices) / len(data)) * 100, 2)
            }
            
            self.logger.info(f"✓ Found {len(anomaly_indices)} anomalies ({metrics['anomalies_percentage']}%)")
            self.model_history['iqr'] = metrics
            
            return anomaly_indices, metrics
            
        except Exception as e:
            self.logger.error(f"✗ Error in IQR detection: {str(e)}")
            raise
    
    def detect_outliers_isolation_forest(self, data: np.ndarray, contamination: float = 0.1,
                                        n_estimators: int = 100, random_state: int = 42) -> Tuple[np.ndarray, Dict]:
        """
        Deteksi anomali menggunakan Isolation Forest.
        
        Args:
            data: Input data
            contamination: Proportion of outliers
            n_estimators: Number of trees
            random_state: Random seed
            
        Returns:
            Tuple of (anomaly_indices, metrics)
        """
        self.logger.info(f"Detecting outliers using Isolation Forest (contamination={contamination})...")
        
        try:
            data = np.asarray(data)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            # Scale data
            data_scaled = self.scaler.fit_transform(data)
            
            # Train Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1
            )
            predictions = iso_forest.fit_predict(data_scaled)
            anomaly_indices = np.where(predictions == -1)[0]
            
            metrics = {
                'method': 'isolation_forest',
                'contamination': contamination,
                'n_estimators': n_estimators,
                'anomalies_count': len(anomaly_indices),
                'anomalies_percentage': round((len(anomaly_indices) / len(data)) * 100, 2),
                'anomaly_score_mean': float(iso_forest.score_samples(data_scaled).mean()),
                'anomaly_score_std': float(iso_forest.score_samples(data_scaled).std())
            }
            
            self.logger.info(f"✓ Found {len(anomaly_indices)} anomalies ({metrics['anomalies_percentage']}%)")
            self.models['isolation_forest'] = iso_forest
            self.model_history['isolation_forest'] = metrics
            
            return anomaly_indices, metrics
            
        except Exception as e:
            self.logger.error(f"✗ Error in Isolation Forest detection: {str(e)}")
            raise
    
    def detect_outliers_lof(self, data: np.ndarray, n_neighbors: int = 20,
                           contamination: float = 0.1) -> Tuple[np.ndarray, Dict]:
        """
        Deteksi anomali menggunakan Local Outlier Factor.
        
        Args:
            data: Input data
            n_neighbors: Number of neighbors
            contamination: Proportion of outliers
            
        Returns:
            Tuple of (anomaly_indices, metrics)
        """
        self.logger.info(f"Detecting outliers using Local Outlier Factor (n_neighbors={n_neighbors})...")
        
        try:
            data = np.asarray(data)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            # Scale data
            data_scaled = self.scaler.fit_transform(data)
            
            # Train LOF
            lof = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=contamination,
                novelty=False,
                n_jobs=-1
            )
            predictions = lof.fit_predict(data_scaled)
            anomaly_indices = np.where(predictions == -1)[0]
            
            metrics = {
                'method': 'lof',
                'n_neighbors': n_neighbors,
                'contamination': contamination,
                'anomalies_count': len(anomaly_indices),
                'anomalies_percentage': round((len(anomaly_indices) / len(data)) * 100, 2),
                'lof_score_mean': float(lof.negative_outlier_factor_.mean()),
                'lof_score_std': float(lof.negative_outlier_factor_.std())
            }
            
            self.logger.info(f"✓ Found {len(anomaly_indices)} anomalies ({metrics['anomalies_percentage']}%)")
            self.models['lof'] = lof
            self.model_history['lof'] = metrics
            
            return anomaly_indices, metrics
            
        except Exception as e:
            self.logger.error(f"✗ Error in LOF detection: {str(e)}")
            raise
    
    def detect_outliers_mahalanobis(self, data: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, Dict]:
        """
        Deteksi anomali menggunakan Mahalanobis distance.
        
        Args:
            data: Input data
            threshold: Distance threshold
            
        Returns:
            Tuple of (anomaly_indices, metrics)
        """
        self.logger.info(f"Detecting outliers using Mahalanobis distance (threshold={threshold})...")
        
        try:
            data = np.asarray(data)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            # Calculate mean and covariance
            mean = np.mean(data, axis=0)
            cov = np.cov(data.T)
            
            # Avoid singular matrix
            if np.linalg.matrix_rank(cov) < cov.shape[0]:
                cov += np.eye(cov.shape[0]) * 1e-6
            
            # Calculate Mahalanobis distances
            inv_cov = np.linalg.inv(cov)
            distances = np.array([
                np.sqrt(mahalanobis(x, mean, inv_cov)) for x in data
            ])
            
            anomaly_indices = np.where(distances > threshold)[0]
            
            metrics = {
                'method': 'mahalanobis',
                'threshold': threshold,
                'anomalies_count': len(anomaly_indices),
                'anomalies_percentage': round((len(anomaly_indices) / len(data)) * 100, 2),
                'distance_mean': float(distances.mean()),
                'distance_std': float(distances.std()),
                'distance_max': float(distances.max())
            }
            
            self.logger.info(f"✓ Found {len(anomaly_indices)} anomalies ({metrics['anomalies_percentage']}%)")
            self.model_history['mahalanobis'] = metrics
            
            return anomaly_indices, metrics
            
        except Exception as e:
            self.logger.error(f"✗ Error in Mahalanobis detection: {str(e)}")
            raise
    
    def detect_outliers_ensemble(self, data: np.ndarray, methods: List[str] = None,
                                voting_threshold: float = 0.5) -> Tuple[np.ndarray, Dict]:
        """
        Deteksi anomali menggunakan ensemble dari multiple methods.
        
        Args:
            data: Input data
            methods: List of methods to use
            voting_threshold: Threshold untuk classify sebagai anomaly
            
        Returns:
            Tuple of (anomaly_indices, metrics)
        """
        self.logger.info("Detecting outliers using ensemble method...")
        
        methods = methods or ['zscore', 'iqr', 'isolation_forest', 'lof']
        data = np.asarray(data).flatten()
        
        try:
            ensemble_votes = np.zeros(len(data))
            method_results = {}
            
            for method in methods:
                self.logger.info(f"  Running {method}...")
                
                if method == 'zscore':
                    anomalies, _ = self.detect_outliers_zscore(data)
                elif method == 'iqr':
                    anomalies, _ = self.detect_outliers_iqr(data)
                elif method == 'isolation_forest':
                    anomalies, _ = self.detect_outliers_isolation_forest(data.reshape(-1, 1))
                elif method == 'lof':
                    anomalies, _ = self.detect_outliers_lof(data.reshape(-1, 1))
                elif method == 'mahalanobis':
                    anomalies, _ = self.detect_outliers_mahalanobis(data.reshape(-1, 1))
                else:
                    self.logger.warning(f"Unknown method: {method}")
                    continue
                
                ensemble_votes[anomalies] += 1
                method_results[method] = len(anomalies)
            
            # Calculate voting score
            voting_score = ensemble_votes / len(methods)
            anomaly_indices = np.where(voting_score >= voting_threshold)[0]
            
            metrics = {
                'method': 'ensemble',
                'base_methods': methods,
                'voting_threshold': voting_threshold,
                'anomalies_count': len(anomaly_indices),
                'anomalies_percentage': round((len(anomaly_indices) / len(data)) * 100, 2),
                'method_results': method_results,
                'voting_score_mean': float(voting_score.mean()),
                'voting_score_std': float(voting_score.std())
            }
            
            self.logger.info(f"✓ Ensemble found {len(anomaly_indices)} anomalies ({metrics['anomalies_percentage']}%)")
            self.model_history['ensemble'] = metrics
            
            return anomaly_indices, metrics
            
        except Exception as e:
            self.logger.error(f"✗ Error in ensemble detection: {str(e)}")
            raise
    
    def identify_optimal_contamination(self, data: np.ndarray, 
                                      contamination_range: Tuple[float, float] = (0.01, 0.20),
                                      step: float = 0.01) -> Dict[float, int]:
        """
        Identify optimal contamination rate by testing different values.
        
        Args:
            data: Input data
            contamination_range: Range untuk test
            step: Step size
            
        Returns:
            Dictionary of contamination vs anomaly count
        """
        self.logger.info("Identifying optimal contamination rate...")
        
        results = {}
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        data_scaled = self.scaler.fit_transform(data)
        
        contamination_values = np.arange(contamination_range[0], contamination_range[1] + step, step)
        
        for cont in contamination_values:
            iso_forest = IsolationForest(contamination=cont, random_state=42, n_jobs=-1)
            predictions = iso_forest.fit_predict(data_scaled)
            anomaly_count = np.sum(predictions == -1)
            results[round(cont, 2)] = anomaly_count
            self.logger.debug(f"  Contamination={cont:.2f}: {anomaly_count} anomalies")
        
        self.logger.info(f"✓ Tested {len(results)} contamination rates")
        return results
    
    def save_model(self, model_name: str, model: Any = None) -> str:
        """Save trained model to disk."""
        self.logger.info(f"Saving model: {model_name}...")
        
        try:
            model_to_save = model or self.models.get(model_name)
            if model_to_save is None:
                raise ValueError(f"Model '{model_name}' not found")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.config.MODEL_FILENAME.format(f"{model_name}_{timestamp}")
            filepath = self.config.MODELS_DIR / filename
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_to_save, f)
            
            self.logger.info(f"✓ Model saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"✗ Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> Any:
        """Load model from disk."""
        self.logger.info(f"Loading model: {filepath}...")
        
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            
            self.logger.info(f"✓ Model loaded successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"✗ Error loading model: {str(e)}")
            raise
    
    def save_model_config(self, config_name: str, config_dict: Dict) -> str:
        """Save model configuration to JSON."""
        self.logger.info(f"Saving model config: {config_name}...")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.config.MODEL_CONFIG_FILENAME.format(f"{config_name}_{timestamp}")
            filepath = self.config.MODELS_DIR / filename
            
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            self.logger.info(f"✓ Config saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"✗ Error saving config: {str(e)}")
            raise
    
    def plot_anomalies(self, data: np.ndarray, anomaly_indices: np.ndarray,
                      method: str = "zscore", filename: str = None) -> str:
        """Visualize detected anomalies."""
        self.logger.info(f"Generating anomaly plot for {method}...")
        
        try:
            data = np.asarray(data).flatten()
            
            plt.figure(figsize=(14, 6))
            plt.plot(data, label='Data', alpha=0.7)
            plt.scatter(anomaly_indices, data[anomaly_indices], color='red', 
                       s=100, label='Anomalies', zorder=5)
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title(f'Anomaly Detection - {method.upper()}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            filename = filename or f"anomalies_{method}.png"
            filepath = self.config.PLOTS_DIR / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✓ Plot saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.warning(f"Error generating plot: {str(e)}")
            return None
    
    def plot_model_comparison(self, data: np.ndarray, methods: List[str] = None) -> str:
        """Compare anomalies detected by different methods."""
        self.logger.info("Generating model comparison plot...")
        
        methods = methods or ['zscore', 'iqr', 'isolation_forest', 'lof']
        data_flat = np.asarray(data).flatten()
        
        try:
            fig, axes = plt.subplots(len(methods), 1, figsize=(14, 4*len(methods)))
            if len(methods) == 1:
                axes = [axes]
            
            for idx, method in enumerate(methods):
                if method == 'zscore':
                    anomalies, _ = self.detect_outliers_zscore(data_flat)
                elif method == 'iqr':
                    anomalies, _ = self.detect_outliers_iqr(data_flat)
                elif method == 'isolation_forest':
                    anomalies, _ = self.detect_outliers_isolation_forest(data_flat.reshape(-1, 1))
                elif method == 'lof':
                    anomalies, _ = self.detect_outliers_lof(data_flat.reshape(-1, 1))
                elif method == 'mahalanobis':
                    anomalies, _ = self.detect_outliers_mahalanobis(data_flat.reshape(-1, 1))
                else:
                    continue
                
                ax = axes[idx]
                ax.plot(data_flat, label='Data', alpha=0.7)
                ax.scatter(anomalies, data_flat[anomalies], color='red', s=50, label='Anomalies')
                ax.set_title(f'{method.upper()} - {len(anomalies)} anomalies')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            filepath = self.config.PLOTS_DIR / "model_comparison.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✓ Comparison plot saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.warning(f"Error generating comparison plot: {str(e)}")
            return None


# ====================== LEGACY FUNCTION (BACKWARD COMPATIBILITY) ======================
def detect_outliers_zscore(data, threshold=3):
    """
    Legacy function: Mendeteksi anomali menggunakan skor Z.
    
    Args:
        data: Input data
        threshold: Z-score threshold
        
    Returns:
        Numpy array dengan indices of anomalies
    """
    mean = np.mean(data)
    std = np.std(data)
    z_scores = [(y - mean) / std for y in data]
    return np.where(np.abs(z_scores) > threshold)[0]


# ====================== MAIN EXECUTION ======================
if __name__ == "__main__":
    print("Modul Modeling (Anomaly Detection) siap digunakan.")