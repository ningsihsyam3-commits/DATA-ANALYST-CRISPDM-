import pandas as pd
import os
import numpy as np

# Definisi jalur folder
RAW_DATA_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw/hospital_data_raw.csv"))
PROCESSED_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed"))
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def clean_data():
    """
    Fungsi ini membersihkan data mentah dan menyimpannya sebagai file CSV terpisah 
    yang siap untuk analisis lebih lanjut di direktori processed/.
    """
    if not os.path.exists(RAW_DATA_FILE):
        print(f"Error: File mentah tidak ditemukan di path: {RAW_DATA_FILE}\n"
              f"Silakan jalankan data_ingestion.py terlebih dahulu untuk mendapatkan file data mentah.")
        return

    print("Memulai proses pembersihan data...")
    
    try:
        # Membaca data mentah
        print(f"Membaca file data mentah dari: {RAW_DATA_FILE}")
        df = pd.read_csv(RAW_DATA_FILE)
        print("File berhasil dibaca!")

        # 1. Menangani Missing Values
        print("Menangani missing values...")
        df.replace('?', np.nan, inplace=True)

        # Fokus pada kolom penting untuk deteksi anomali
        print("Memilih kolom penting...")
        cols_to_keep = [
            'encounter_id', 'patient_nbr', 'age', 'time_in_hospital',
            'num_lab_procedures', 'num_procedures', 'num_medications',
            'number_diagnoses', 'race', 'gender', 'readmitted'
        ]
        df_clean = df[cols_to_keep].dropna().copy()

        # 2. Transformasi Sederhana: Membersihkan format kolom 'age'
        print("Melakukan transformasi pada kolom 'age'...")
        df_clean['age'] = df_clean['age'].str.extract(r'(\d+)').astype(float) + 5

        # Validasi data yang bersih
        if df_clean.empty:
            print("Error: Data bersih menghasilkan file kosong. Periksa data mentah Anda.")
            return

        # 3. Simpan hasil ke folder processed
        output_file = os.path.join(PROCESSED_DATA_DIR, "hospital_data_cleaned.csv")
        print(f"Menyimpan hasil pembersihan ke {output_file}...")
        df_clean.to_csv(output_file, index=False)
        print(f"Selesai! Data bersih telah disimpan di: {output_file}")

    except Exception as e:
        print(f"Terjadi kesalahan selama proses pembersihan data: {e}")

if __name__ == "__main__":
    clean_data()