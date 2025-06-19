# Aplikasi Prediksi Risiko COVID-19 di Indonesia

Aplikasi ini menggunakan model machine learning (Algoritma C4.5/Decision Tree) untuk memprediksi tingkat risiko penyebaran COVID-19 di Indonesia berdasarkan berbagai parameter.

## Fitur Aplikasi

- Prediksi tingkat risiko COVID-19 (Rendah, Sedang, Tinggi) berdasarkan input parameter
- Visualisasi interaktif sebaran risiko COVID-19 di provinsi-provinsi Indonesia
- Analisis kontribusi parameter terhadap skor risiko
- Informasi detail tentang model dan metode perhitungan

## Prerequisites

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Folium
- Streamlit-Folium
- Joblib

## Cara Menjalankan Aplikasi

1. Install dependencies menggunakan pip:

   ```bash
   pip install -r requirements.txt
   ```

2. Jalankan aplikasi:

   ```bash
   streamlit run app.py
   ```

3. Buka aplikasi di browser (biasanya tersedia di http://localhost:8501)

## Penggunaan Aplikasi

1. Masukkan nilai parameter di sidebar:

   - New Cases per Million
   - Total Cases per Million
   - Case Fatality Rate (%)
   - Population Density (per km²)
   - Cases Growth Rate
   - Deaths Growth Rate

2. Klik tombol "Prediksi Risiko!"

3. Lihat hasil prediksi, kontribusi parameter, dan visualisasi peta risiko di tab yang tersedia.

## Dataset dan Model

Aplikasi ini menggunakan model yang dilatih dengan dataset COVID-19 Indonesia (covid_19_indonesia.csv). Model disimpan sebagai file model_c45_covid_risiko.pkl dan dapat dimuat untuk melakukan prediksi secara real-time.

## Struktur Model

- **Algoritma**: C4.5 (Decision Tree dengan kriteria entropy)

## Tugas Mata Kuliah Kecerdasan Bisnis
