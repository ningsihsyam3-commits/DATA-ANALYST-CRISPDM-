def evaluate_anomalies(total_data, anomalies_found):
    """
    Menghitung persentase temuan anomali.
    """
    percentage = (len(anomalies_found) / total_data) * 100
    return f"Ditemukan {len(anomalies_found)} anomali ({percentage:.2f}% dari total data)."

if __name__ == "__main__":
    print("Modul Evaluation siap digunakan.")