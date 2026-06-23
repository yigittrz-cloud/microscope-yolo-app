import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --------------------------------------------------
# SAYFA AYARI (Daha Geniş ve Modern Düzen)
# --------------------------------------------------
st.set_page_config(
    page_title="Breed TPC Hesaplama",
    page_icon="🧫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# MODERN CSS TASARIM
# --------------------------------------------------
st.markdown(
    """
    <style>
    /* Global Font ve Arka Plan */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
    }
    
    /* Hero Kartı */
    .hero-card {
        background: linear-gradient(135deg, #0f766e 0%, #0d9488 100%);
        padding: 35px;
        border-radius: 24px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 25px -5px rgba(15, 118, 110, 0.2), 0 8px 10px -6px rgba(15, 118, 110, 0.2);
    }
    
    .hero-title {
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin-bottom: 5px;
    }
    
    .hero-subtitle {
        font-size: 16px;
        opacity: 0.9;
        font-weight: 400;
    }

    /* Görsel Bilgi Kartları */
    .image-card {
        background-color: white;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        transition: transform 0.2s ease;
    }
    .image-card:hover {
        transform: translateY(-2px);
    }

    /* Tab Tasarımları */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
        color: #64748b;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        color: #0f766e !important;
        border-bottom: 3px solid #0f766e !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# ÜST BAŞLIK (Hero Section)
# --------------------------------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🧫 Breed TPC Analiz Paneli</div>
        <div class="hero-subtitle">YOLO tabanlı mikroskopik görüntü analiz ve otomatik hesaplama sistemi</div>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# MODEL YÜKLEME
# --------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception:
    st.error("Model yüklenemedi. best.pt dosyasının app.py ile aynı klasörde olduğundan emin ol.")
    st.stop()

# --------------------------------------------------
# 10 ÜZERİ FORMAT
# --------------------------------------------------
def bilimsel_10_uzeri(sayi):
    if sayi is None or sayi == 0:
        return "0"

    us = int(np.floor(np.log10(abs(sayi))))
    katsayi = sayi / (10 ** us)

    ust_karakterler = {
        "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
        "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻"
    }

    us_yazi = "".join(ust_karakterler.get(char, char) for char in str(us))
    return f"{katsayi:.2f} × 10{us_yazi}"

# --------------------------------------------------
# GÖRÜNTÜ OKUMA
# --------------------------------------------------
def resmi_oku(uploaded_file):
    image = Image.open(uploaded_file)
    try:
        image.seek(0)
    except Exception:
        pass
    image = image.convert("RGB")
    return np.array(image)

# --------------------------------------------------
# SINIF SAYMA VE İŞARETLEME
# --------------------------------------------------
def say_ve_isaretle(image_np, conf_value):
    results = model(image_np, conf=conf_value)

    bakteri_sayisi = 0
    maya_kuf_sayisi = 0
    spor_sayisi = 0

    annotated_image = image_np.copy()

    for result in results:
        annotated_image = result.plot()
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id].lower().strip()

            if any(x in class_name for x in ["bakteri", "bacteria", "bacteri"]):
                bakteri_sayisi += 1
            elif any(x in class_name for x in ["maya", "yeast", "küf", "kuf", "mold", "fungus", "fungi"]):
                maya_kuf_sayisi += 1
            elif any(x in class_name for x in ["spor", "spore"]):
                spor_sayisi += 1

    return bakteri_sayisi, maya_kuf_sayisi, spor_sayisi, annotated_image

# --------------------------------------------------
# BREED HESABI
# --------------------------------------------------
def breed_hesapla(ortalama_sayi, tek_goruntu_alani_mm2, yayma_alani_mm2, damlatilan_hacim_ml, seyreltme_carpani):
    if damlatilan_hacim_ml == 0 or tek_goruntu_alani_mm2 == 0:
        return 0
    return ortalama_sayi * (yayma_alani_mm2 / tek_goruntu_alani_mm2) * seyreltme_carpani / damlatilan_hacim_ml

def seyreltme_carpani_bul(seyreltme_secimi):
    sozluk = {"Seyreltme yok": 1, "10^-1": 10, "10^-2": 100, "10^-3": 1000, "10^-4": 10000, "10^-5": 100000, "10^-6": 1000000}
    return sozluk.get(seyreltme_secimi, 1)

# --------------------------------------------------
# SOL PANEL (SIDEBAR) - AYARLAR BURAYA TAŞINDI
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Analiz Ayarları")
    st.markdown("---")
    
    conf_value = st.slider("Confidence (Güven) Eşiği", min_value=0.01, max_value=0.90, value=0.05, step=0.01)
    
    st.markdown("### 📏 Alan & Hacim Ayarları")
    tek_goruntu_alani_mm2 = st.number_input("Tek Görüntü Alanı (mm²)", min_value=0.0001, value=0.20, step=0.01, format="%.4f")
    yayma_alani_mm2 = st.number_input("Breed Yayma Alanı (mm²)", min_value=1.0, value=100.0, step=1.0, format="%.2f")
    damlatilan_hacim_ml = st.number_input("Damlatılan Hacim (mL)", min_value=0.001, value=0.01, step=0.001, format="%.3f")
    
    seyreltme_secimi = st.selectbox("Seyreltme Faktörü", ["Seyreltme yok", "10^-1", "10^-2", "10^-3", "10^-4", "10^-5", "10^-6"])
    seyreltme_carpani = seyreltme_carpani_bul(seyreltme_secimi)

# --------------------------------------------------
# ANA SAYFA - GÖRÜNTÜ YÜKLEME
# --------------------------------------------------
uploaded_files = st.file_uploader(
    "📤 Mikroskop görüntülerini sürükleyin veya seçin",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    accept_multiple_files=True
)

# --------------------------------------------------
# ANALİZ VE SONUÇLAR
# --------------------------------------------------
if uploaded_files:
    toplam_bakteri = 0
    toplam_maya_kuf = 0
    toplam_spor = 0
    isaretli_gorseller = []

    with st.spinner("🚀 Yapay zeka görüntüleri analiz ediyor..."):
        for uploaded_file in uploaded_files:
            try:
                image_np = resmi_oku(uploaded_file)
            except Exception:
                st.error(f"{uploaded_file.name} okunamadı.")
                continue

            bakteri_sayisi, maya_kuf_sayisi, spor_sayisi, annotated_image = say_ve_isaretle(
                image_np=image_np,
                conf_value=conf_value
            )

            toplam_bakteri += bakteri_sayisi
            toplam_maya_kuf += maya_kuf_sayisi
            toplam_spor += spor_sayisi

            isaretli_gorseller.append({
                "dosya_adi": uploaded_file.name,
                "isaretli": annotated_image,
                "bakteri": bakteri_sayisi,
                "maya_kuf": maya_kuf_sayisi,
                "spor": spor_sayisi
            })

    if not isaretli_gorseller:
        st.error("Hiçbir görüntü analiz edilemedi.")
        st.stop()

    analiz_edilen_goruntu_sayisi = len(isaretli_gorseller)
    ortalama_bakteri = toplam_bakteri / analiz_edilen_goruntu_sayisi
    ortalama_maya_kuf = toplam_maya_kuf / analiz_edilen_goruntu_sayisi

    bakteri_tpc = breed_hesapla(ortalama_bakteri, tek_goruntu_alani_mm2, yayma_alani_mm2, damlatilan_hacim_ml, seyreltme_carpani)
    maya_kuf_tpc = breed_hesapla(ortalama_maya_kuf, tek_goruntu_alani_mm2, yayma_alani_mm2, damlatilan_hacim_ml, seyreltme_carpani)

    st.toast("Analiz başarıyla tamamlandı!", icon="✅")

    # Sekmeler
    tab1, tab2 = st.tabs(["📊 Analiz Raporu", "🔍 İşaretlenmiş Görüntüler"])

    with tab1:
        st.subheader("📊 Hesaplanan TPC Sonuçları")
        
        # Streamlit Native Metrik Kartları ile modern görünüm
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            with st.container(border=True):
                st.metric(label="🦠 Bakteri TPC (/mL)", value=bilimsel_10_uzeri(bakteri_tpc))
        with m_col2:
            with st.container(border=True):
                st.metric(label="🍄 Maya-Küf TPC (/mL)", value=bilimsel_10_uzeri(maya_kuf_tpc))

        st.markdown("### 📌 Veri Özet Tablosu")
        
        # Özet verileri şık bir tablo yapısında sunalım
        ozet_data = {
            "Metrik": [
                "Analiz Edilen Görüntü Sayısı", 
                "Toplam / Ortalama Bakteri", 
                "Toplam / Ortalama Maya-Küf", 
                "Toplam Spor Sayısı"
            ],
            "Değer": [
                f"{analiz_edilen_goruntu_sayisi} adet",
                f"{toplam_bakteri} / {ortalama_bakteri:.2f}",
                f"{toplam_maya_kuf} / {ortalama_maya_kuf:.2f}",
                f"{toplam_spor} adet"
            ]
        }
        st.table(ozet_data)

    with tab2:
        st.subheader("🔍 Nesne Tespiti Çıktıları")
        
        # Görselleri yan yana 2'li grid düzeninde göstermek alanı daha iyi kullanır
        img_cols = st.columns(2)
        for i, gorsel in enumerate(isaretli_gorseller):
            col_idx = i % 2
            with img_cols[col_idx]:
                st.markdown(
                    f"""
                    <div class="image-card">
                        <span style="color:#0f766e; font-weight:bold;">📷 Görüntü {i+1}:</span> {gorsel['dosya_adi']}<br>
                        <span style="font-size:13px; color:#64748b;">
                            Bakteri: <b>{gorsel['bakteri']}</b> | Maya-Küf: <b>{gorsel['maya_kuf']}</b> | Spor: <b>{gorsel['spor']}</b>
                        </span>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                st.image(gorsel["isaretli"], use_container_width=True)

else:
    # Boş durum (Empty State) tasarımı
    st.info("💡 Başlamak için yukarıdaki alana mikroskop görüntülerini (JPG, PNG, TIF) sürükleyip bırakın.")
