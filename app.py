import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import folium
from streamlit_folium import folium_static
import os

# Set halaman dengan konfigurasi lebar
st.set_page_config(
    page_title="COVID-19 Risk Prediction App",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul dan deskripsi aplikasi
st.title("🦠 Aplikasi Prediksi Risiko COVID-19 di Indonesia")
st.markdown("""
Aplikasi ini menggunakan model machine learning (Algoritma C4.5/Decision Tree) untuk memprediksi 
tingkat risiko penyebaran COVID-19 berdasarkan berbagai parameter. Model ini dilatih menggunakan 
dataset COVID-19 Indonesia yang telah dianalisis dan diproses.
""")

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    model = joblib.load('model_c45_covid_risiko.pkl')
    return model

# Fungsi untuk memuat dataset hasil prediksi untuk visualisasi
@st.cache_data
def load_prediction_data():
    try:
        data = pd.read_csv('hasil_prediksi_risiko_covid.csv')
        return data
    except FileNotFoundError:
        st.error("File hasil_prediksi_risiko_covid.csv tidak ditemukan!")
        return None

# Fungsi untuk normalisasi input
def normalize_features(input_data, features_for_scoring):
    # Siapkan dummy data untuk MinMaxScaler
    dummy_data = pd.DataFrame({
        'New Cases per Million': [0, 1459.04],  # Nilai min-max dari dataset
        'Case Fatality Rate': [0, 100],         # Asumsi persentase 0-100%
        'Cases_Growth_Rate': [-1, 175],         # Nilai dari notebook
        'Deaths_Growth_Rate': [-1, 134.5],      # Nilai dari notebook
        'Population Density': [8.59, 16334.31]  # Nilai min-max dari dataset
    })
    
    # Buat scaler dan fit dengan dummy data
    scaler = MinMaxScaler()
    scaler.fit(dummy_data[features_for_scoring])
    
    # Normalisasi input data
    # Konversi input array ke DataFrame dengan kolom yang sama
    input_df = pd.DataFrame([input_data], columns=features_for_scoring)
    normalized = scaler.transform(input_df)
    return normalized

# Fungsi untuk menghitung risk score
def calculate_risk_score(normalized_data, weights):
    return np.dot(normalized_data, weights)[0]

# Fungsi untuk memprediksi tingkat risiko berdasarkan K-means clustering
def predict_risk_level(risk_score):
    # Threshold berdasarkan K-means centroids dari notebook yang diperbarui
    if risk_score <= 0.0043:  # Centroid cluster "Rendah"
        return "Rendah"
    elif risk_score <= 0.0103:  # Centroid cluster "Sedang"
        return "Sedang"
    else:
        return "Tinggi"

# Fungsi untuk visualisasi peta sebaran risiko
def visualize_risk_map(data):
    # Siapkan peta
    center_lat = data['Latitude'].mean()
    center_lon = data['Longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles='OpenStreetMap'
    )
    
    # Definisikan warna untuk tingkat risiko
    risk_colors = {
        'Rendah': 'green',
        'Sedang': 'orange',
        'Tinggi': 'red'
    }
    
    # Tambahkan marker untuk setiap provinsi
    for _, row in data.iterrows():
        color = risk_colors.get(row['Risk_Level'], 'gray')
        
        popup_text = f"""
        <b>{row['Province']}</b><br>
        Risiko: <b>{row['Risk_Level']}</b><br>
        Risk Score: {row['Risk_Score']:.4f}<br>
        Total Kasus: {row['Total Cases']:,.0f}<br>
        Kasus per Juta: {row['Total Cases per Million']:,.0f}<br>
        CFR: {row['Case Fatality Rate']:.2f}%
        """
        
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=max(8, min(25, row['Total Cases per Million'] / 1000 if not pd.isna(row['Total Cases per Million']) else 8)),
            popup=folium.Popup(popup_text, max_width=300),
            color='black',
            fillColor=color,
            fillOpacity=0.8,
            weight=2
        ).add_to(m)
        
    # Tambahkan legend
    legend_html = '''
    <div style="position: fixed; 
         top: 10px; right: 10px; width: 150px; height: 90px; 
         background-color: white; border:2px solid grey; z-index:9999; 
         font-size:14px; padding: 10px">
    <p><b>Tingkat Risiko</b></p>
    <p><i class="fa fa-circle" style="color:green"></i> Rendah</p>
    <p><i class="fa fa-circle" style="color:orange"></i> Sedang</p>
    <p><i class="fa fa-circle" style="color:red"></i> Tinggi</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

# Load model
try:
    model = load_model()
    st.success("✅ Model C4.5 berhasil dimuat!")
except Exception as e:
    st.error(f"❌ Error saat memuat model: {e}")
    st.stop()

# Sidebar untuk input pengguna
st.sidebar.header("Input Parameter:")

# Form input untuk prediksi
with st.sidebar.form("prediction_form"):
    st.write("Masukkan nilai parameter untuk prediksi:")
    
    new_cases_per_million = st.number_input(
        "New Cases per Million",
        min_value=0.0,
        max_value=1500.0,
        value=10.0,
        step=1.0,
        help="Jumlah kasus baru per juta penduduk"
    )
    
    total_cases_per_million = st.number_input(
        "Total Cases per Million",
        min_value=0.0,
        max_value=150000.0,
        value=5000.0,
        step=100.0,
        help="Total kasus per juta penduduk"
    )
    
    case_fatality_rate = st.number_input(
        "Case Fatality Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=2.5,
        step=0.1,
        help="Persentase kematian dari total kasus"
    )
    
    population_density = st.number_input(
        "Population Density (per km²)",
        min_value=0.0,
        max_value=20000.0,
        value=100.0,
        step=10.0,
        help="Kepadatan penduduk per kilometer persegi"
    )
    
    cases_growth_rate = st.number_input(
        "Cases Growth Rate",
        min_value=-1.0,
        max_value=10.0,
        value=0.05,
        step=0.01,
        format="%.2f",
        help="Laju pertumbuhan kasus baru (perubahan persentase)"
    )
    
    deaths_growth_rate = st.number_input(
        "Deaths Growth Rate",
        min_value=-1.0,
        max_value=10.0,
        value=0.01,
        step=0.01,
        format="%.2f",
        help="Laju pertumbuhan kematian baru (perubahan persentase)"
    )
    
    submit_button = st.form_submit_button(label="Prediksi Risiko!")

# Tab untuk hasil dan visualisasi
tab1, tab2, tab3 = st.tabs(["📊 Hasil Prediksi", "🗺️ Peta Risiko", "ℹ️ Informasi Model"])

# Tab 1: Hasil Prediksi
with tab1:
    st.header("Hasil Prediksi Risiko COVID-19")
    
    if submit_button:
        st.write("### Analisis Input Parameter")
        
        # PERUBAHAN: Menghapus Risk_Score dari model input
        # Siapkan data input untuk perhitungan risk score
        features_for_scoring = ['New Cases per Million', 'Case Fatality Rate', 
                              'Cases_Growth_Rate', 'Deaths_Growth_Rate', 'Population Density']
        
        # Pastikan urutan nilai input sesuai dengan urutan features_for_scoring
        input_data = np.array([
            new_cases_per_million,  # New Cases per Million
            case_fatality_rate,     # Case Fatality Rate
            cases_growth_rate,      # Cases_Growth_Rate
            deaths_growth_rate,     # Deaths_Growth_Rate
            population_density      # Population Density
        ])
        
        # Normalisasi data input untuk risk score
        try:
            normalized_data = normalize_features(input_data, features_for_scoring)
            
            # Hitung risk score
            weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Bobot berdasarkan notebook
            risk_score = calculate_risk_score(normalized_data, weights)
            
        except Exception as e:
            st.error(f"Error saat normalisasi data: {str(e)}")
            st.stop()
        
        # PERUBAHAN: Model tidak lagi menggunakan Risk_Score sebagai fitur
        # Prediksi dengan model C4.5 tanpa risk score
        try:
            model_input = np.array([
                new_cases_per_million,
                total_cases_per_million,
                case_fatality_rate,
                population_density,
                cases_growth_rate,
                deaths_growth_rate
            ]).reshape(1, -1)
            
            prediction = model.predict(model_input)[0]
            
        except Exception as e:
            st.error(f"Error saat melakukan prediksi: {str(e)}")
            st.stop()
        
        # Tampilkan hasil
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Risk Score", f"{risk_score:.4f}")
            
        with col2:
            st.metric("Prediksi Tingkat Risiko", prediction)
            
        with col3:
            if prediction == "Rendah":
                st.success("Risiko Rendah")
            elif prediction == "Sedang":
                st.warning("Risiko Sedang")
            else:
                st.error("Risiko Tinggi")
        
        # Bar chart untuk menunjukkan feature importance relatif
        st.write("### Kontribusi Parameter pada Risk Score")
        
        # Kontribusi setiap fitur berdasarkan bobot
        contributions = normalized_data[0] * weights
        contribution_df = pd.DataFrame({
            'Feature': features_for_scoring,
            'Contribution': contributions,
            'Weight': weights
        })
        contribution_df['Percentage'] = contribution_df['Contribution'] / risk_score * 100
        contribution_df = contribution_df.sort_values('Contribution', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='Contribution', y='Feature', data=contribution_df, palette='viridis')
        plt.title('Kontribusi Parameter pada Risk Score')
        plt.xlabel('Kontribusi')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Tampilkan persentase kontribusi
        st.write("### Persentase Kontribusi Parameter")
        
        # Tabel kontribusi
        contribution_table = pd.DataFrame({
            'Parameter': contribution_df['Feature'],
            'Kontribusi': contribution_df['Contribution'].round(6),
            'Persentase (%)': contribution_df['Percentage'].round(2)
        })
        st.table(contribution_table)

# Tab 2: Peta Risiko
with tab2:
    st.header("Peta Sebaran Risiko COVID-19 di Indonesia")
    
    # Load data untuk visualisasi
    prediction_data = load_prediction_data()
    
    if prediction_data is not None:
        st.markdown("""
        Peta di bawah ini menunjukkan sebaran tingkat risiko COVID-19 di berbagai provinsi di Indonesia 
        berdasarkan data historis yang dianalisis. Warna marker menunjukkan tingkat risiko:
        - 🟢 **Hijau**: Risiko Rendah
        - 🟠 **Oranye**: Risiko Sedang
        - 🔴 **Merah**: Risiko Tinggi
        """)
        
        # Buat dan tampilkan peta
        risk_map = visualize_risk_map(prediction_data)
        folium_static(risk_map, width=1000, height=600)
        
        # Tampilkan data provinsi dalam tabel
        st.write("### Data Lengkap per Provinsi")
        st.dataframe(
            prediction_data[['Province', 'Risk_Level', 'Risk_Score', 
                           'Total Cases', 'Total Cases per Million', 
                           'Case Fatality Rate']].sort_values('Province'),
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("Data hasil prediksi tidak tersedia. Silakan jalankan notebook untuk menghasilkan prediksi per provinsi.")

# Tab 3: Informasi Model
with tab3:
    st.header("Informasi Model C4.5")
    
    st.markdown("""
    ### Tentang Model
    
    Model yang digunakan dalam aplikasi ini adalah **Decision Tree Classifier** dengan kriteria split **entropy** 
    (implementasi algoritma C4.5). Model ini dilatih menggunakan dataset COVID-19 Indonesia yang mencakup 
    data dari seluruh provinsi di Indonesia.
    
    ### Parameter Model:
    - **Criterion**: Entropy (Information Gain)
    - **Max Depth**: 5 (dibatasi untuk mencegah overfitting)
    - **Min Samples Split**: 20
    - **Min Samples Leaf**: 10
    
    ### Feature yang Digunakan:
    1. **New Cases per Million**: Jumlah kasus baru per juta penduduk
    2. **Total Cases per Million**: Total kasus per juta penduduk
    3. **Case Fatality Rate**: Persentase kasus yang berakhir dengan kematian
    4. **Population Density**: Kepadatan penduduk per kilometer persegi
    5. **Cases Growth Rate**: Tingkat pertumbuhan kasus baru
    6. **Deaths Growth Rate**: Tingkat pertumbuhan kematian
    
    ### CATATAN PENTING:
    Risk Score **tidak** digunakan sebagai fitur input model untuk menghindari data leakage, 
    karena Risk Score digunakan untuk membuat label Risk Level. Menggunakan Risk Score sebagai 
    fitur akan membuat model hanya "menyalin" informasi yang sudah ada.
    
    ### Performa Model:
    Model mencapai **akurasi 70-85%** pada data testing, yang merupakan performa realistis untuk 
    model prediksi risiko COVID-19.
    
    ### Cara Perhitungan Risk Score:
    Risk Score dihitung sebagai kombinasi berbobot dari berbagai parameter yang telah dinormalisasi:
    - New Cases per Million: **Bobot 30%**
    - Case Fatality Rate: **Bobot 25%**
    - Cases Growth Rate: **Bobot 20%**
    - Deaths Growth Rate: **Bobot 15%**
    - Population Density: **Bobot 10%**
    
    ### Kategori Tingkat Risiko:
    Kategori risiko ditentukan dengan metode K-means clustering untuk menemukan pengelompokan alamiah
    dalam data, bukannya hanya memotong berdasarkan persentil.
    """)
    
    # Menampilkan informasi teknis model
    if model:
        st.subheader("Struktur Model Decision Tree:")
        
        # Informasi tentang tree
        tree_info = {
            "Jumlah Nodes": model.tree_.node_count,
            "Kedalaman Tree": model.tree_.max_depth,
            "Jumlah Feature": model.n_features_in_,
            "Jumlah Kelas": len(model.classes_),
            "Kelas Output": ", ".join(model.classes_)
        }
        
        # Tampilkan sebagai tabel
        st.table(pd.DataFrame(list(tree_info.items()), columns=["Parameter", "Nilai"]))
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance:")
            
            # PERUBAHAN: Menghapus Risk_Score dari daftar fitur
            feature_cols = ['New Cases per Million', 'Total Cases per Million', 'Case Fatality Rate',
                           'Population Density', 'Cases_Growth_Rate', 'Deaths_Growth_Rate']
            
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
            plt.title('Feature Importance dalam Model', fontsize=14)
            plt.xlabel('Importance Score', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Aplikasi Prediksi Risiko COVID-19 | Tugas Mata Kuliah Kecerdasan Bisnis</p>
    <p>Model: Algoritma C4.5 (Decision Tree) | Dataset: COVID-19 Indonesia</p>
</div>
""", unsafe_allow_html=True)
