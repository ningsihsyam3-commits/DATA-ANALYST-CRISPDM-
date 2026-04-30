# 📊 Laporan Analisis: Deteksi Anomali Biaya Kesehatan
**Penyusun:** Ningsih (Data Admin Specialist)
**Metodologi:** CRISP-DM

## 1. Pendahuluan (Business Understanding)
Laporan ini merangkum temuan dari analisis data penagihan medis untuk mengidentifikasi pola yang tidak wajar atau anomali menggunakan Python.

## 2. Metodologi Analisis
Proyek ini menggunakan beberapa tahapan utama:
*   **Pembersihan Data**: Dilakukan melalui skrip modular di `src/data_preparation.py`.
*   **Pemodelan**: Menggunakan algoritma statistik (seperti Z-Score) untuk menentukan batas anomali.
*   **Visualisasi**: Dashboard interaktif yang dibangun menggunakan Streamlit.

## 3. Temuan Utama
*   **Total Data**: [Isi jumlah baris data Anda]
*   **Persentase Anomali**: [Isi persentase temuan]
*   **Kategori Anomali**: [Contoh: Biaya ganda, klaim di atas rata-rata]

## 4. Rekomendasi Teknis
Berdasarkan hasil pemodelan, disarankan untuk melakukan audit mendalam pada data yang memiliki tingkat penyimpangan di atas ambang batas (threshold) yang telah ditentukan.

## 5. Cara Menjalankan Proyek
Untuk mereplikasi analisis ini, pastikan pustaka pendukung telah terinstal melalui file `requirements.txt`.