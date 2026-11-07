import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import zipfile

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Arabica Bean Quality Evaluator",
    page_icon="☕",
    layout="centered"
)

st.title("☕ Arabica Bean Quality Evaluator")
st.markdown("""
Aplikasi Sistem Pakar Pengenalan dan Klasifikasi Mutu Biji Kopi Arabika (**Dataset USK-Coffee**) 
berbasis **Ensemble Deep Feature Extraction (ResNet50 + EfficientNetB0)**, **PCA (1024)**, dan **Support Vector Machine (SVM)**.
""")

# ==========================================
# 2. PEMUATAN MODEL ARTEFAK (MODEL V3)
# ==========================================
@st.cache_resource
def load_feature_extractors():
    device = torch.device("cpu")
    
    # ResNet50 Feature Extractor
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet.fc = nn.Identity()
    resnet = resnet.to(device).eval()

    # EfficientNetB0 Feature Extractor
    effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    effnet.classifier = nn.Identity()
    effnet = effnet.to(device).eval()

    return resnet, effnet, device

@st.cache_resource
def load_ml_pipeline_v3():
    scaler = joblib.load('scaler_weighted_ensemble_v3.pkl')
    pca = joblib.load('pca_1024_v3.pkl')
    
    # Ekstrak berkas zip SVM jika berkas .pkl langsung belum ada di folder
    if os.path.exists('svm_classifier_v3.zip') and not os.path.exists('svm_classifier_v3.pkl'):
        with zipfile.ZipFile('svm_classifier_v3.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
            
    svm = joblib.load('svm_classifier_v3.pkl')
    w_res, w_eff = np.load('best_ratio.npy')
    class_names = ['defect', 'longberry', 'peaberry', 'premium']
    
    return scaler, pca, svm, w_res, w_eff, class_names

with st.spinner("Memuat arsitektur AI dan bobot model v3 teroptimasi..."):
    resnet_model, effnet_model, device = load_feature_extractors()
    scaler, pca, svm_model, w_res, w_eff, class_names = load_ml_pipeline_v3()

# Preprocessing Gambar sesuai Standar ImageNet
IMAGE_SIZE = (256, 256)
transform_pipeline = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. INTERFASE UNGGAH & PREDIKSI
# ==========================================
st.write("---")
uploaded_file = st.file_uploader("Unggah Gambar Biji Kopi Arabika (Format: JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar Citra Uji Biji Kopi", use_container_width=True)
    
    with st.spinner("Mengekstraksi fitur ensemble & mengklasifikasikan mutu..."):
        # 1. Preprocessing Citra
        img_tensor = transform_pipeline(image).unsqueeze(0).to(device)
        
        # 2. Ekstraksi Fitur Deep Learning
        with torch.no_grad():
            res_feat = resnet_model(img_tensor).cpu().numpy()
            eff_feat = effnet_model(img_tensor).cpu().numpy()
        
        # 3. Weighted Feature Fusion
        fused_feat = np.hstack((w_res * res_feat, w_eff * eff_feat))
        
        # 4. Standard Scaler & Transformasi PCA (1024 Komponen)
        scaled_feat = scaler.transform(fused_feat)
        pca_feat = pca.transform(scaled_feat)
        
        # 5. Prediksi Klasifikasi & Probability Scaling SVM
        prediction_idx = svm_model.predict(pca_feat)[0]
        probabilities = svm_model.predict_proba(pca_feat)[0]
        
        predicted_class = class_names[prediction_idx]
        max_prob = probabilities[prediction_idx] * 100

    # ==========================================
    # 4. VISUALISASI HASIL PREDIKSI & METRIK
    # ==========================================
    st.write("---")
    st.subheader("Hasil Evaluasi Mutu Biji Kopi:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**Kelas Terprediksi:**\n### `{predicted_class.upper()}`")
    with col2:
        st.info(f"**Tingkat Keyakinan (Confidence):**\n### `{max_prob:.2f}%`")
    
    st.subheader("Distribusi Probabilitas Klasifikasi Per Kelas:")
    
    # Chart Horizontal menggunakan Matplotlib
    fig, ax = plt.subplots(figsize=(8, 3.5))
    y_pos = np.arange(len(class_names))
    prob_percent = probabilities * 100
    
    colors = ['#1f77b4' if i != prediction_idx else '#2ca02c' for i in range(len(class_names))]
    bars = ax.barh(y_pos, prob_percent, align='center', color=colors)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([c.capitalize() for c in class_names], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel('Probabilitas Terkalibrasi (%)', fontsize=11)
    ax.set_xlim(0, 100)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.2f}%',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    with st.expander("Lihat Rincian Probabilitas Numerik"):
        prob_df = {
            "Kelas Biji Kopi": [c.capitalize() for c in class_names],
            "Probabilitas (%)": [f"{p * 100:.4f}%" for p in probabilities]
        }
        st.dataframe(prob_df, use_container_width=True)