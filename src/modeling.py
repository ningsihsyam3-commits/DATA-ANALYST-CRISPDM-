import numpy as np

def detect_outliers_zscore(data, threshold=3):
    """
    Mendeteksi anomali menggunakan skor Z.
    """
    mean = np.mean(data)
    std = np.std(data)
    z_scores = [(y - mean) / std for y in data]
    return np.where(np.abs(z_scores) > threshold)

if __name__ == "__main__":
    print("Modul Modeling (Anomaly Detection) siap digunakan.")