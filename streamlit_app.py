import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the preprocessing pipeline and model
preprocess = joblib.load("preprocess_pipeline.joblib")
model = load_model("model.keras")

st.title("İstanbul Kiralık Ev Fiyat Tahmin Uygulaması")

# Sidebar inputs for the user
st.sidebar.header("Girdi Bilgileri")

categorical_input1 = st.sidebar.selectbox("Eşyalı/Eşyasız", ["Eşyalı", "Eşyasız"])
categorical_input2 = st.sidebar.selectbox("Otopark", ["Var", "Yok"])
categorical_input3 = st.sidebar.selectbox("Binanın Bulunduğu Kat", ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10+"])
categorical_input4 = st.sidebar.selectbox("Güvenlik", ["Site İçinde", "Sağlam", "Standart"])

numerical_input1 = st.sidebar.number_input("Alışveriş Merkezine Uzaklık (km)", min_value=0.0, max_value=100.0, value=0.0)
numerical_input2 = st.sidebar.number_input("Eğitim Kurumlarına Uzaklık (km)", min_value=0.0, max_value=100.0, value=0.0)
numerical_input3 = st.sidebar.number_input("Toplu Taşımaya Uzaklık (km)", min_value=0.0, max_value=100.0, value=0.0)
numerical_input4 = st.sidebar.number_input("Oda Sayısı", min_value=1, max_value=10, value=1)

# Istanbul districts
istanbul_ilceler = [
    "Adalar", "Arnavutköy", "Ataşehir", "Avcılar", "Bağcılar", "Bağdat", "Bakırköy",
    "Başakşehir", "Bayrampaşa", "Beşiktaş", "Beykoz", "Beyoğlu", "Büyükçekmece",
    "Çatalca", "Esenler", "Eyüpsultan", "Fatih", "Gaziosmanpaşa",
    "Güngören", "Kadıköy", "Kağıthane", "Kartal", "Küçükçekmece", "Maltepe",
    "Pendik", "Sancaktepe", "Sarıyer", "Şişli", "Sultanbeyli", "Sultanahmet",
    "Tuzla", "Ümraniye", "Üsküdar", "Zeytinburnu"
]

categorical_input5 = st.sidebar.selectbox("İlçe", istanbul_ilceler)
categorical_input6 = st.sidebar.selectbox("Balkon", ["Var", "Yok"])
numerical_input5 = st.sidebar.number_input("Metrekare", min_value=10, max_value=1000, value=50)
numerical_input6 = st.sidebar.number_input("Bina Yaşı", min_value=0, max_value=100, value=0)

isitma_tipleri = ['Doğalgaz Merkezi', 'Doğalgaz Bireysel', 'Elektrik', 'Kömür', 'Şehir Isıtması', 'Soba']
categorical_input7 = st.sidebar.selectbox("Isıtma Tipi", isitma_tipleri)

numerical_input7 = st.sidebar.number_input("Kat Sayısı", min_value=1, max_value=50, value=1)

# Create a DataFrame for the input
input_data = pd.DataFrame({
    'Eşyalı/Eşyasız': [categorical_input1],
    'Otopark': [categorical_input2],
    'Binanın Bulunduğu Kat': [categorical_input3],
    'Güvenlik': [categorical_input4],
    'Alışveriş Merkezine Uzaklık(km)': [numerical_input1],
    'Eğitim Kurumları Uzaklık(km)': [numerical_input2],
    'Toplu Taşımaya Uzaklık(km)': [numerical_input3],
    'Oda Sayısı': [numerical_input4],
    'İlçe': [categorical_input5],
    'Balkon': [categorical_input6],
    'Metrekare': [numerical_input5],
    'Bina Yaşı': [numerical_input6],
    'Isıtma Tipi': [categorical_input7],
    'Kat Sayısı': [numerical_input7]
})

# Preprocess the input data
input_processed = preprocess.transform(input_data)

# Predict the price
prediction = model.predict(input_processed)

# Display the prediction
st.write(f"Tahmin Edilen Fiyat: {prediction[0][0]:,.2f} TL")
