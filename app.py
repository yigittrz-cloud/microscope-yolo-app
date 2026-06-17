import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np


# --------------------------------------------------
# SAYFA AYARI
# --------------------------------------------------
st.set_page_config(
    page_title="Breed TPC Hesaplama",
    layout="centered"
)

st.title("Breed TPC Hesaplama")
st.write("Mikroskop görüntülerini yükle. Sistem bakteri sayısını analiz edip tahmini TPC/mL sonucunu verir.")


# --------------------------------------------------
# MODEL YÜKLEME
# --------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")


try:
    model = load_model()
except Exception as e:
    st.error("Model yüklenemedi. best.pt dosyasının app.py ile aynı klasörde olduğundan emin ol.")
    st.stop()


# --------------------------------------------------
# FONKSİYONLAR
# --------------------------------------------------
def bakteri_say(image_np, conf_value):
    results = model(image_np, conf=conf_value)

    bakteri_sayisi = 0

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id].lower().strip()

            if "bakteri" in class_name:
                bakteri_sayisi += 1

    return bakteri_sayisi


def breed_tpc_hesapla(
    ortalama_bakteri,
    tek_goruntu_alani_mm2,
    yayma_alani_mm2,
    damlatilan_hacim_ml,
    seyreltme_carpani
):
    tpc_ml = (
        ortalama_bakteri
        * (yayma_alani_mm2 / tek_goruntu_alani_mm2)
        * seyreltme_carpani
        / damlatilan_hacim_ml
    )

    return tpc_ml


def seyreltme_carpani_bul(seyreltme_secimi):
    if seyreltme_secimi == "Seyreltme yok":
        return 1
    elif seyreltme_secimi == "10^-1":
        return 10
    elif seyreltme_secimi == "10^-2":
        return 100
    elif seyreltme_secimi == "10^-3":
        return 1000
    elif seyreltme_secimi == "10^-4":
        return 10000
    elif seyreltme_secimi == "10^-5":
        return 100000
    elif seyreltme_secimi == "10^-6":
        return 1000000
    else:
        return 1


# --------------------------------------------------
# KISA AYARLAR
# --------------------------------------------------
with st.expander("Ayarlar"):
    conf_value = st.slider(
        "Confidence eşiği",
        min_value=0.01,
        max_value=0.90,
        value=0.05,
        step=0.01
    )

    tek_goruntu_alani_mm2 = st.number_input(
        "Tek görüntünün alanı (mm²)",
        min_value=0.0001,
        value=0.20,
        step=0.01,
        format="%.4f"
    )

    yayma_alani_mm2 = st.number_input(
        "Breed yayma alanı (mm²)",
        min_value=1.0,
        value=100.0,
        step=1.0,
        format="%.2f"
    )

    damlatilan_hacim_ml = st.number_input(
        "Damlatılan hacim (mL)",
        min_value=0.001,
        value=0.01,
        step=0.001,
        format="%.3f"
    )

    seyreltme_secimi = st.selectbox(
        "Seyreltme",
        [
            "Seyreltme yok",
            "10^-1",
            "10^-2",
            "10^-3",
            "10^-4",
            "10^-5",
            "10^-6"
        ]
    )

seyreltme_carpani = seyreltme_carpani_bul(seyreltme_secimi)


# --------------------------------------------------
# FOTOĞRAF YÜKLEME
# --------------------------------------------------
uploaded_files = st.file_uploader(
    "Mikroskop görüntülerini yükle",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)


# --------------------------------------------------
# ANALİZ
# --------------------------------------------------
if uploaded_files:

    toplam_bakteri = 0
    goruntu_sayisi = len(uploaded_files)

    with st.spinner("Görüntüler analiz ediliyor..."):
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)

            bakteri_sayisi = bakteri_say(
                image_np=image_np,
                conf_value=conf_value
            )

            toplam_bakteri += bakteri_sayisi

    ortalama_bakteri = toplam_bakteri / goruntu_sayisi

    tpc_sonuc = breed_tpc_hesapla(
        ortalama_bakteri=ortalama_bakteri,
        tek_goruntu_alani_mm2=tek_goruntu_alani_mm2,
        yayma_alani_mm2=yayma_alani_mm2,
        damlatilan_hacim_ml=damlatilan_hacim_ml,
        seyreltme_carpani=seyreltme_carpani
    )

    st.success("Analiz tamamlandı.")

    st.metric(
        label="Tahmini Breed TPC",
        value=f"{tpc_sonuc:.2e} TPC/mL"
    )

    st.write(f"Yaklaşık sonuç: **{tpc_sonuc:,.0f} TPC/mL**")

    with st.expander("Kısa özet"):
        st.write(f"Yüklenen görüntü sayısı: **{goruntu_sayisi}**")
        st.write(f"Toplam bakteri sayısı: **{toplam_bakteri}**")
        st.write(f"Ortalama bakteri/görüntü: **{ortalama_bakteri:.2f}**")

else:
    st.warning("Analiz için mikroskop görüntüsü yükle.")
