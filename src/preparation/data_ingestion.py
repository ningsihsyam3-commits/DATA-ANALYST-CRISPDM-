import pandas as pd
import os
from ucimlrepo import fetch_ucirepo

# Path untuk folder data/raw
RAW_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw"))
os.makedirs(RAW_DATA_DIR, exist_ok=True)

def ingest_data():
    """
    Fungsi ini mengunduh dataset dari UCI Machine Learning Repository,
    menyimpannya sebagai file CSV di direktori data/raw.
    """
    print("Memulai proses pengambilan data dari UCI...")

    try:
        # Mengambil dataset Diabetes 130-Hospitals (ID: 296)
        # Dataset ini memiliki fitur medis yang kaya untuk analisis anomali
        print("Sedang mengunduh dataset dengan ID 296...")
        dataset = fetch_ucirepo(id=296)
        print("Dataset berhasil diunduh!")
    except Exception as e:
        print(f"Terjadi kesalahan saat mengambil dataset: {e}")
        return

    try:
        # Menggabungkan fitur dan target menjadi satu DataFrame
        print("Menggabungkan fitur dan target menjadi DataFrame...")
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

        # Nama file tujuan
        file_name = "hospital_data_raw.csv"
        file_dest = os.path.join(RAW_DATA_DIR, file_name)

        # Menyimpan DataFrame sebagai file CSV
        print(f"Menyimpan DataFrame ke {file_dest}...")
        df.to_csv(file_dest, index=False)

        print(f"Berhasil! Data mentah disimpan di: {file_dest}")
    except Exception as e:
        print(f"Terjadi kesalahan saat memproses atau menyimpan data: {e}")

if __name__ == "__main__":
    ingest_data()