import pandas as pd
import numpy as np
import os
import logging
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score, accuracy_score, precision_score,
    recall_score, matthews_corrcoef, cohen_kappa_score
)
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings('ignore')

# ====================== KONFIGURASI ======================
class EvaluationConfig:
    """Configuration untuk Model Evaluation."""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation"
    LOGS_DIR = PROJECT_ROOT / "logs"
    PLOTS_DIR = PROJECT_ROOT / "reports" / "plots"
    
    # File names
    EVALUATION_REPORT = "evaluation_report.json"
    METRICS_SUMMARY = "metrics_summary.txt"
    
    def __init__(self):
        """Initialize config dan create directories."""
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ====================== LOGGING SETUP ======================
def setup_logging(config: EvaluationConfig) -> logging.Logger:
    """Setup logging system."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    logger.handlers.clear()
    
    # File handler
    log_file = config.LOGS_DIR / "evaluation.log"
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

# ====================== EVALUATION CLASS ======================
class ModelEvaluator:
    """Comprehensive model evaluation dengan multiple metrics dan visualizations."""
    
    def __init__(self, config: EvaluationConfig = None):
        """
        Initialize Model Evaluator.
        
        Args:
            config: EvaluationConfig object
        """
        self.config = config or EvaluationConfig()
        self.logger = setup_logging(self.config)
        self.evaluation_results = {}
        
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_pred_proba: Optional[np.ndarray] = None,
                               labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate classification model dengan multiple metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for ROC-AUC)
            labels: Label names
            
        Returns:
            Dictionary dengan evaluation metrics
        """
        self.logger.info("=" * 80)
        self.logger.info("EVALUASI CLASSIFICATION MODEL")
        self.logger.info("=" * 80)
        
        metrics = {}
        
        try:
            # Basic metrics
            self.logger.info("Calculating basic metrics...")
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
            metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics['f1_score'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics['matthew_corrcoef'] = float(matthews_corrcoef(y_true, y_pred))
            metrics['cohen_kappa'] = float(cohen_kappa_score(y_true, y_pred))
            
            self.logger.info(f"✓ Basic metrics calculated")
            
            # ROC-AUC (jika multiclass or binary dengan probability)
            if y_pred_proba is not None:
                try:
                    if len(np.unique(y_true)) == 2:
                        metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
                    else:
                        metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted'))
                    self.logger.info(f"✓ ROC-AUC calculated: {metrics['roc_auc']:.4f}")
                except Exception as e:
                    self.logger.warning(f"Could not calculate ROC-AUC: {str(e)}")
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            self.logger.info(f"✓ Confusion matrix generated: shape {cm.shape}")
            
            # Classification report
            class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            metrics['classification_report'] = class_report
            self.logger.info(f"✓ Classification report generated")
            
            # Sensitivity dan Specificity (untuk binary classification)
            if len(np.unique(y_true)) == 2:
                cm_binary = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm_binary.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                metrics['sensitivity'] = float(sensitivity)
                metrics['specificity'] = float(specificity)
                self.logger.info(f"✓ Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
            
            self.evaluation_results['classification'] = metrics
            
        except Exception as e:
            self.logger.error(f"✗ Error during classification evaluation: {str(e)}")
            raise
        
        return metrics
    
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate regression model.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary dengan regression metrics
        """
        self.logger.info("=" * 80)
        self.logger.info("EVALUASI REGRESSION MODEL")
        self.logger.info("=" * 80)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {}
        
        try:
            # Calculate metrics
            metrics['mse'] = float(mean_squared_error(y_true, y_pred))
            metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
            metrics['r2_score'] = float(r2_score(y_true, y_pred))
            
            # MAPE
            mask = y_true != 0
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            metrics['mape'] = float(mape)
            
            self.logger.info(f"✓ MSE: {metrics['mse']:.4f}")
            self.logger.info(f"✓ RMSE: {metrics['rmse']:.4f}")
            self.logger.info(f"✓ MAE: {metrics['mae']:.4f}")
            self.logger.info(f"✓ R² Score: {metrics['r2_score']:.4f}")
            self.logger.info(f"✓ MAPE: {metrics['mape']:.4f}%")
            
            self.evaluation_results['regression'] = metrics
            
        except Exception as e:
            self.logger.error(f"✗ Error during regression evaluation: {str(e)}")
            raise
        
        return metrics
    
    def evaluate_anomalies(self, total_data: int, anomalies_found: List[Any]) -> Dict[str, Any]:
        """
        Evaluate anomaly detection model.
        
        Args:
            total_data: Total data points
            anomalies_found: List of detected anomalies or indices
            
        Returns:
            Dictionary dengan anomaly metrics
        """
        self.logger.info("=" * 80)
        self.logger.info("EVALUASI ANOMALY DETECTION")
        self.logger.info("=" * 80)
        
        metrics = {}
        
        try:
            n_anomalies = len(anomalies_found)
            percentage = (n_anomalies / total_data) * 100
            
            metrics['total_data'] = total_data
            metrics['anomalies_found'] = n_anomalies
            metrics['percentage'] = round(percentage, 2)
            metrics['normal_data'] = total_data - n_anomalies
            metrics['normal_percentage'] = round(100 - percentage, 2)
            
            self.logger.info(f"✓ Total data: {total_data}")
            self.logger.info(f"✓ Anomalies found: {n_anomalies} ({percentage:.2f}%)")
            self.logger.info(f"✓ Normal data: {metrics['normal_data']} ({metrics['normal_percentage']:.2f}%)")
            
            self.evaluation_results['anomalies'] = metrics
            
        except Exception as e:
            self.logger.error(f"✗ Error during anomaly evaluation: {str(e)}")
            raise
        
        return metrics
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray,
                           cv: int = 5, scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            cv: Number of folds
            scoring: Scoring metric
            
        Returns:
            Dictionary dengan cross-validation results
        """
        self.logger.info(f"Performing {cv}-fold cross-validation...")
        
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            
            cv_results = {
                'cv_folds': cv,
                'scoring_metric': scoring,
                'scores': scores.tolist(),
                'mean_score': float(scores.mean()),
                'std_score': float(scores.std()),
                'min_score': float(scores.min()),
                'max_score': float(scores.max())
            }
            
            self.logger.info(f"✓ Cross-validation completed")
            self.logger.info(f"  Mean score: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
            
            self.evaluation_results['cross_validation'] = cv_results
            
            return cv_results
            
        except Exception as e:
            self.logger.error(f"✗ Error during cross-validation: {str(e)}")
            raise
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            labels: Optional[List[str]] = None,
                            filename: str = "confusion_matrix.png") -> str:
        """Visualize confusion matrix."""
        self.logger.info("Generating confusion matrix plot...")
        
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            output_path = self.config.PLOTS_DIR / filename
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✓ Plot saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.warning(f"Error generating confusion matrix plot: {str(e)}")
            return None
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      filename: str = "roc_curve.png") -> str:
        """Visualize ROC curve."""
        self.logger.info("Generating ROC curve plot...")
        
        try:
            if len(np.unique(y_true)) == 2:
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
                auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(y_true, classes=np.unique(y_true))
                fpr, tpr, auc_score = {}, {}, {}
                plt.figure(figsize=(10, 8))
                for i in range(y_bin.shape[1]):
                    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
                    auc_score[i] = roc_auc_score(y_bin[:, i], y_pred_proba[:, i])
                    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC={auc_score[i]:.3f})')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve (One-vs-Rest)')
                plt.legend()
                output_path = self.config.PLOTS_DIR / filename
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info(f"✓ ROC curve plot saved: {output_path}")
                return str(output_path)
            
            plt.figure(figsize=(10, 8))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC={auc_score:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            
            output_path = self.config.PLOTS_DIR / filename
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✓ ROC curve plot saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.warning(f"Error generating ROC curve plot: {str(e)}")
            return None
    
    def save_evaluation_report(self, report_name: str = None) -> str:
        """Save evaluation report ke file JSON."""
        self.logger.info("Saving evaluation report...")
        
        try:
            report_name = report_name or self.config.EVALUATION_REPORT
            report_path = self.config.RESULTS_DIR / report_name
            
            report = {
                'evaluation_timestamp': datetime.now().isoformat(),
                'results': self.evaluation_results
            }
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"✓ Evaluation report saved: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"✗ Error saving evaluation report: {str(e)}")
            raise
    
    def generate_summary_report(self, summary_name: str = None) -> str:
        """Generate text summary report."""
        self.logger.info("Generating summary report...")
        
        try:
            summary_name = summary_name or self.config.METRICS_SUMMARY
            summary_path = self.config.RESULTS_DIR / summary_name
            
            with open(summary_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("MODEL EVALUATION SUMMARY REPORT\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for eval_type, metrics in self.evaluation_results.items():
                    f.write(f"\n{eval_type.upper()}\n")
                    f.write("-" * 80 + "\n")
                    
                    if isinstance(metrics, dict):
                        for key, value in metrics.items():
                            if key not in ['confusion_matrix', 'classification_report']:
                                if isinstance(value, (int, float)):
                                    f.write(f"  {key}: {value:.4f}\n")
                                else:
                                    f.write(f"  {key}: {value}\n")
            
            self.logger.info(f"✓ Summary report saved: {summary_path}")
            return str(summary_path)
            
        except Exception as e:
            self.logger.error(f"✗ Error generating summary report: {str(e)}")
            raise


# ====================== LEGACY FUNCTION (BACKWARD COMPATIBILITY) ======================
def evaluate_anomalies(total_data, anomalies_found):
    """
    Legacy function: Menghitung persentase temuan anomali.
    
    Args:
        total_data: Total data points
        anomalies_found: List of detected anomalies
        
    Returns:
        String dengan informasi anomali
    """
    percentage = (len(anomalies_found) / total_data) * 100
    return f"Ditemukan {len(anomalies_found)} anomali ({percentage:.2f}% dari total data)."


# ====================== MAIN EXECUTION ======================
if __name__ == "__main__":
    print("Modul Evaluation siap digunakan.")