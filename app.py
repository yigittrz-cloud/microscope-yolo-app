import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd


# --------------------------------------------------
# SAYFA AYARI
# --------------------------------------------------
st.set_page_config(
    page_title="Mikroskop YOLO Analizi",
    layout="centered"
)

st.title("Mikroskop Görüntü Analizi")
st.write(
    "Çoklu mikroskop görüntüsü yükleyerek bakteri, maya ve spor sayımı yapar. "
    "Bakteri ortalamasından Breed formülüne göre tahmini TPC/mL hesaplar."
)


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
    st.exception(e)
    st.stop()


# --------------------------------------------------
# YARDIMCI FONKSİYONLAR
# --------------------------------------------------
def sayim_yap(image_np, conf_value):
    """
    YOLO sonucundan bakteri, maya ve spor sayısı hesaplar.
    Sınıf adında 'bakteri', 'maya' veya 'spor' kelimesi geçmesi yeterlidir.
    Örnek:
    bakteri
    bakteri_x
    bakteri-1
    maya
    spor
    """

    results = model(image_np, conf=conf_value)

    bakteri_sayisi = 0
    maya_sayisi = 0
    spor_sayisi = 0
    annotated_image = None

    tespit_edilen_siniflar = []

    for result in results:
        annotated_image = result.plot()

        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id].lower().strip()

            tespit_edilen_siniflar.append(class_name)

            if "bakteri" in class_name:
                bakteri_sayisi += 1
            elif "maya" in class_name:
                maya_sayisi += 1
            elif "spor" in class_name:
                spor_sayisi += 1

    return bakteri_sayisi, maya_sayisi, spor_sayisi, annotated_image, tespit_edilen_siniflar


def breed_tpc_hesapla(
    ortalama_bakteri,
    tek_goruntu_alani_mm2,
    yayma_alani_mm2,
    damlatilan_hacim_ml,
    seyreltme_carpani
):
    """
    Breed TPC hesabı.

    Formül:
    TPC/mL =
    Ortalama bakteri sayısı
    × Yayma alanı / Tek görüntü alanı
    × Seyreltme çarpanı
    / Damlatılan hacim
    """

    if tek_goruntu_alani_mm2 <= 0:
        return None

    if damlatilan_hacim_ml <= 0:
        return None

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
# SIDEBAR AYARLARI
# --------------------------------------------------
st.sidebar.header("YOLO Ayarları")

conf_value = st.sidebar.slider(
    "Confidence eşiği",
    min_value=0.01,
    max_value=0.90,
    value=0.05,
    step=0.01
)

st.sidebar.header("Breed Hesaplama Parametreleri")

tek_goruntu_alani_mm2 = st.sidebar.number_input(
    "Tek görüntünün alanı (mm²)",
    min_value=0.0001,
    value=0.20,
    step=0.01,
    format="%.4f"
)

yayma_alani_mm2 = st.sidebar.number_input(
    "Breed yayma alanı (mm²)",
    min_value=1.0,
    value=100.0,
    step=1.0,
    format="%.2f"
)

damlatilan_hacim_ml = st.sidebar.number_input(
    "Damlatılan hacim (mL)",
    min_value=0.001,
    value=0.01,
    step=0.001,
    format="%.3f"
)

seyreltme_secimi = st.sidebar.selectbox(
    "Seyreltme",
    [
        "Seyreltme yok",
        "10^-1",
        "10^-2",
        "10^-3",
        "10^-4",
        "10^-5",
        "10^-6",
        "Manuel"
    ]
)

if seyreltme_secimi == "Manuel":
    seyreltme_carpani = st.sidebar.number_input(
        "Manuel seyreltme çarpanı",
        min_value=1,
        value=1,
        step=1
    )
else:
    seyreltme_carpani = seyreltme_carpani_bul(seyreltme_secimi)


debug_modu = st.sidebar.checkbox(
    "Tespit edilen sınıf isimlerini göster",
    value=False
)


# --------------------------------------------------
# FORMÜL BİLGİSİ
# --------------------------------------------------
with st.expander("Kullanılan Breed formülü"):
    st.write(
        """
        **TPC/mL = Ortalama bakteri sayısı × (Yayma alanı / Tek görüntü alanı) × Seyreltme çarpanı / Damlatılan hacim**

        Örnek:

        - Foto 1: 40 bakteri
        - Foto 2: 55 bakteri
        - Foto 3: 35 bakteri

        Ortalama bakteri sayısı:

        **(40 + 55 + 35) / 3 = 43,3**

        Bu ortalama değer formüle yazılır.

        Bu durumda formülde **tek görüntü alanı** kullanılır.
        """
    )


# --------------------------------------------------
# FOTOĞRAF YÜKLEME
# --------------------------------------------------
uploaded_files = st.file_uploader(
    "Mikroskop görüntülerini yükle",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)


# --------------------------------------------------
# ANA ANALİZ
# --------------------------------------------------
if uploaded_files:

    st.success(f"{len(uploaded_files)} adet görüntü yüklendi.")

    sonuc_listesi = []

    toplam_bakteri = 0
    toplam_maya = 0
    toplam_spor = 0

    tum_tespit_edilen_siniflar = []

    for i, uploaded_file in enumerate(uploaded_files, start=1):

        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        bakteri_sayisi, maya_sayisi, spor_sayisi, annotated_image, tespit_edilen_siniflar = sayim_yap(
            image_np=image_np,
            conf_value=conf_value
        )

        toplam_bakteri += bakteri_sayisi
        toplam_maya += maya_sayisi
        toplam_spor += spor_sayisi

        tum_tespit_edilen_siniflar.extend(tespit_edilen_siniflar)

        sonuc_listesi.append(
            {
                "Görüntü": uploaded_file.name,
                "Bakteri": bakteri_sayisi,
                "Maya": maya_sayisi,
                "Spor": spor_sayisi
            }
        )

        st.subheader(f"Görüntü {i}: {uploaded_file.name}")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Orijinal görüntü")
            st.image(image, use_container_width=True)

        with col2:
            st.write("YOLO sonucu")
            if annotated_image is not None:
                st.image(annotated_image, use_container_width=True)
            else:
                st.warning("Bu görüntüde YOLO sonucu oluşmadı.")

        c1, c2, c3 = st.columns(3)

        c1.metric("Bakteri", bakteri_sayisi)
        c2.metric("Maya", maya_sayisi)
        c3.metric("Spor", spor_sayisi)

        if debug_modu:
            st.write("Bu görüntüde tespit edilen sınıflar:")
            st.write(tespit_edilen_siniflar)

        st.divider()

    # --------------------------------------------------
    # ORTALAMA HESAPLAMA
    # --------------------------------------------------
    goruntu_sayisi = len(uploaded_files)

    ortalama_bakteri = toplam_bakteri / goruntu_sayisi
    ortalama_maya = toplam_maya / goruntu_sayisi
    ortalama_spor = toplam_spor / goruntu_sayisi

    tpc_sonuc = breed_tpc_hesapla(
        ortalama_bakteri=ortalama_bakteri,
        tek_goruntu_alani_mm2=tek_goruntu_alani_mm2,
        yayma_alani_mm2=yayma_alani_mm2,
        damlatilan_hacim_ml=damlatilan_hacim_ml,
        seyreltme_carpani=seyreltme_carpani
    )

    # --------------------------------------------------
    # GENEL SONUÇ
    # --------------------------------------------------
    st.header("Genel Sonuç")

    sonuc_df = pd.DataFrame(sonuc_listesi)
    st.dataframe(sonuc_df, use_container_width=True)

    st.subheader("Toplam Sayımlar")

    t1, t2, t3 = st.columns(3)

    t1.metric("Toplam bakteri", toplam_bakteri)
    t2.metric("Toplam maya", toplam_maya)
    t3.metric("Toplam spor", toplam_spor)

    st.subheader("Ortalama Sayımlar")

    o1, o2, o3 = st.columns(3)

    o1.metric("Ortalama bakteri / görüntü", f"{ortalama_bakteri:.2f}")
    o2.metric("Ortalama maya / görüntü", f"{ortalama_maya:.2f}")
    o3.metric("Ortalama spor / görüntü", f"{ortalama_spor:.2f}")

    st.subheader("Breed TPC Sonucu")

    if tpc_sonuc is not None:
        st.metric(
            "Tahmini Breed TPC",
            f"{tpc_sonuc:.2e} TPC/mL"
        )

        st.write(f"Yaklaşık sonuç: **{tpc_sonuc:,.0f} TPC/mL**")

        st.write("Hesapta kullanılan değerler:")

        hesap_df = pd.DataFrame(
            [
                {
                    "Ortalama bakteri/görüntü": ortalama_bakteri,
                    "Tek görüntü alanı mm²": tek_goruntu_alani_mm2,
                    "Breed yayma alanı mm²": yayma_alani_mm2,
                    "Damlatılan hacim mL": damlatilan_hacim_ml,
                    "Seyreltme çarpanı": seyreltme_carpani,
                    "Tahmini TPC/mL": tpc_sonuc
                }
            ]
        )

        st.dataframe(hesap_df, use_container_width=True)

    else:
        st.error("TPC hesaplanamadı. Alan veya hacim değerlerini kontrol et.")

    st.info(
        "Bu değer YOLO ile tespit edilen mikroskobik bakteri sayısına göre hesaplanan tahmini Breed TPC/mL sonucudur. "
        "Klasik plak sayımıyla birebir aynı kabul edilmemelidir."
    )

    if debug_modu:
        st.subheader("Tüm tespit edilen sınıf isimleri")
        st.write(tum_tespit_edilen_siniflar)

    # --------------------------------------------------
    # CSV İNDİRME
    # --------------------------------------------------
    ozet_df = pd.DataFrame(
        [
            {
                "Görüntü sayısı": goruntu_sayisi,
                "Toplam bakteri": toplam_bakteri,
                "Toplam maya": toplam_maya,
                "Toplam spor": toplam_spor,
                "Ortalama bakteri/görüntü": ortalama_bakteri,
                "Ortalama maya/görüntü": ortalama_maya,
                "Ortalama spor/görüntü": ortalama_spor,
                "Tek görüntü alanı mm2": tek_goruntu_alani_mm2,
                "Breed yayma alanı mm2": yayma_alani_mm2,
                "Damlatılan hacim mL": damlatilan_hacim_ml,
                "Seyreltme çarpanı": seyreltme_carpani,
                "Tahmini Breed TPC/mL": tpc_sonuc
            }
        ]
    )

    csv_df = pd.concat([sonuc_df, ozet_df], axis=0)

    csv = csv_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="Sonuçları CSV olarak indir",
        data=csv,
        file_name="breed_tpc_sonuc.csv",
        mime="text/csv"
    )

else:
    st.warning("Analiz için bir veya birden fazla mikroskop görüntüsü yükle.")
