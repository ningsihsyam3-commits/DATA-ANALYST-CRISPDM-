import pandas as pd

def clean_data(df):
    """
    Fungsi untuk membersihkan data kesehatan.
    """
    # Menghapus duplikat
    df = df.drop_duplicates()
    # Mengisi missing values pada kolom biaya dengan angka 0
    if 'biaya' in df.columns:
        df['biaya'] = df['biaya'].fillna(0)
    return df

if __name__ == "__main__":
    print("Modul Data Preparation siap digunakan.")