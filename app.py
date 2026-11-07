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
# 1. KONFIGURASI HALAMAN & CUSTOM CSS TEMA KOPI
# ==========================================
st.set_page_config(
    page_title="Arabica Bean Quality Evaluator",
    page_icon="☕",
    layout="centered"
)

# Custom Styling (Theme: Warm Coffee & Cream)
st.markdown("""
<style>
    /* Background Utama & Font Base */
    .stApp {
        background-color: #FAF6F0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header Utama */
    .main-header {
        color: #3C2A21;
        text-align: center;
        font-weight: 700;
        padding-bottom: 5px;
        margin-bottom: 0px;
    }
    
    .sub-header {
        color: #6E5343;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }

    /* Card Hasil Prediksi */
    .metric-card-success {
        background-color: #E8F5E9;
        border-left: 6px solid #2E7D32;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .metric-card-info {
        background-color: #E3F2FD;
        border-left: 6px solid #1565C0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .card-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #555555;
        margin-bottom: 5px;
    }
    
    .card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #111111;
    }
    
    /* Uploader Box */
    .stFileUploader > div {
        border: 2px dashed #D5CEA3;
        background-color: #FFFFFF;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Header Aplikasi
st.markdown("<h1 class='main-header'>☕ Arabica Bean Quality Evaluator</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Sistem Pakar Evaluasi Mutu Biji Kopi Arabika • <b>Dataset USK-Coffee</b></p>", unsafe_allow_html=True)

st.markdown("""
<div style='background-color: #FFF8EA; padding: 12px 18px; border-radius: 8px; border: 1px solid #E5D9B6; font-size: 0.92rem; color: #574B90; margin-bottom: 20px;'>
<b>Metode AI:</b> Ensemble Deep Feature Extraction (<i>ResNet50 + EfficientNetB0</i>) dengan pembobotan optimal $64\%:36\%$, <i>PCA 1024 Komponen</i>, serta <i>Support Vector Machine (SVM RBF)</i> terkalibrasi probabilitas.
</div>
""", unsafe_allow_html=True)

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
    
    # Ekstrak berkas zip SVM jika berkas .pkl belum diekstrak
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
uploaded_file = st.file_uploader("Unggah Citra Biji Kopi Arabika (Format: JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    st.write("")
    col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
    with col_img2:
        st.image(image, caption="Citra Uji Biji Kopi Arabika", use_container_width=True)
    
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
    st.markdown("<h3 style='color: #3C2A21; text-align: center;'>Hasil Evaluasi Mutu Biji Kopi</h3>", unsafe_allow_html=True)
    st.write("")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card-success">
            <div class="card-title">KELAS TERPREDIKSI</div>
            <div class="card-value" style="color: #2E7D32;">{predicted_class.upper()}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card-info">
            <div class="card-title">TINGKAT KEYAKINAN (CONFIDENCE)</div>
            <div class="card-value" style="color: #1565C0;">{max_prob:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")
    st.markdown("<h4 style='color: #6E5343;'>Distribusi Probabilitas Terkalibrasi Per Kelas:</h4>", unsafe_allow_html=True)
    
    # Custom Palette Warm Coffee Gradient
    bar_colors = []
    for i in range(len(class_names)):
        if i == prediction_idx:
            bar_colors.append('#3C2A21') # Espresso / Winner Class
        else:
            bar_colors.append('#D5CEA3') # Light Cream
            
    fig, ax = plt.subplots(figsize=(8, 3.2))
    fig.patch.set_facecolor('#FAF6F0')
    ax.set_facecolor('#FAF6F0')
    
    y_pos = np.arange(len(class_names))
    prob_percent = probabilities * 100
    
    bars = ax.barh(y_pos, prob_percent, align='center', color=bar_colors, edgecolor='#3C2A21', linewidth=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([c.capitalize() for c in class_names], fontsize=11, fontweight='bold', color='#3C2A21')
    ax.invert_yaxis()
    ax.set_xlabel('Probabilitas Terkalibrasi (%)', fontsize=10, fontweight='bold', color='#3C2A21')
    ax.set_xlim(0, 105)
    
    # Hilangkan border kotak atas & kanan grafik
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#3C2A21')
    ax.spines['bottom'].set_color('#3C2A21')
    
    for bar in bars:
        width = bar.get_width()
        txt_color = '#3C2A21' if width < 80 else '#FFFFFF'
        x_pos_text = width + 1.5 if width < 80 else width - 12
        ax.text(x_pos_text, bar.get_y() + bar.get_height()/2, f'{width:.2f}%',
                ha='left' if width < 80 else 'right', va='center', 
                fontsize=10, fontweight='bold', color=txt_color)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    with st.expander("📊 Lihat Rincian Probabilitas Numerik"):
        prob_df = {
            "Kelas Biji Kopi": [c.capitalize() for c in class_names],
            "Probabilitas (%)": [f"{p * 100:.4f}%" for p in probabilities]
        }
        st.dataframe(prob_df, use_container_width=True)