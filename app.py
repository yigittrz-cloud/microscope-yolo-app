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
st.write("APP VERSION: TIFF_BAKTERI_MAYA_KUF_5")
st.write("Mikroskop görüntülerinden bakteri ve maya-küf sayımını ayrı hesaplar.")


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
    if sayi is None:
        return "0"

    if sayi == 0:
        return "0"

    us = int(np.floor(np.log10(abs(sayi))))
    katsayi = sayi / (10 ** us)

    ust_karakterler = {
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹",
        "-": "⁻"
    }

    us_yazi = "".join(ust_karakterler.get(char, char) for char in str(us))

    return f"{katsayi:.2f} × 10{us_yazi}"


# --------------------------------------------------
# GÖRÜNTÜ OKUMA
# --------------------------------------------------
def resmi_oku(uploaded_file):
    image = Image.open(uploaded_file)

    # TIFF çok sayfalıysa ilk sayfayı alır
    try:
        image.seek(0)
    except Exception:
        pass

    image = image.convert("RGB")
    image_np = np.array(image)

    return image_np


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

            # BAKTERİ
            if (
                "bakteri" in class_name
                or "bacteria" in class_name
                or "bacteri" in class_name
            ):
                bakteri_sayisi += 1

            # MAYA-KÜF
            elif (
                "maya" in class_name
                or "yeast" in class_name
                or "küf" in class_name
                or "kuf" in class_name
                or "mold" in class_name
                or "fungus" in class_name
                or "fungi" in class_name
            ):
                maya_kuf_sayisi += 1

            # SPOR
            elif (
                "spor" in class_name
                or "spore" in class_name
            ):
                spor_sayisi += 1

    return bakteri_sayisi, maya_kuf_sayisi, spor_sayisi, annotated_image


# --------------------------------------------------
# BREED HESABI
# --------------------------------------------------
def breed_hesapla(
    ortalama_sayi,
    tek_goruntu_alani_mm2,
    yayma_alani_mm2,
    damlatilan_hacim_ml,
    seyreltme_carpani
):
    sonuc = (
        ortalama_sayi
        * (yayma_alani_mm2 / tek_goruntu_alani_mm2)
        * seyreltme_carpani
        / damlatilan_hacim_ml
    )

    return sonuc


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
# AYARLAR
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
# GÖRÜNTÜ YÜKLEME
# --------------------------------------------------
uploaded_files = st.file_uploader(
    "Mikroskop görüntülerini yükle",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    accept_multiple_files=True
)


# --------------------------------------------------
# ANALİZ
# --------------------------------------------------
if uploaded_files:

    toplam_bakteri = 0
    toplam_maya_kuf = 0
    toplam_spor = 0

    isaretli_gorseller = []

    with st.spinner("Görüntüler analiz ediliyor..."):

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

            isaretli_gorseller.append(
                {
                    "dosya_adi": uploaded_file.name,
                    "isaretli": annotated_image,
                    "bakteri": bakteri_sayisi,
                    "maya_kuf": maya_kuf_sayisi,
                    "spor": spor_sayisi
                }
            )

    if len(isaretli_gorseller) == 0:
        st.error("Hiçbir görüntü analiz edilemedi.")
        st.stop()

    analiz_edilen_goruntu_sayisi = len(isaretli_gorseller)

    ortalama_bakteri = toplam_bakteri / analiz_edilen_goruntu_sayisi
    ortalama_maya_kuf = toplam_maya_kuf / analiz_edilen_goruntu_sayisi

    bakteri_tpc = breed_hesapla(
        ortalama_sayi=ortalama_bakteri,
        tek_goruntu_alani_mm2=tek_goruntu_alani_mm2,
        yayma_alani_mm2=yayma_alani_mm2,
        damlatilan_hacim_ml=damlatilan_hacim_ml,
        seyreltme_carpani=seyreltme_carpani
    )

    maya_kuf_tpc = breed_hesapla(
        ortalama_sayi=ortalama_maya_kuf,
        tek_goruntu_alani_mm2=tek_goruntu_alani_mm2,
        yayma_alani_mm2=yayma_alani_mm2,
        damlatilan_hacim_ml=damlatilan_hacim_ml,
        seyreltme_carpani=seyreltme_carpani
    )

    st.success("Analiz tamamlandı.")

    tab1, tab2 = st.tabs(["Sonuç", "İşaretlemeler"])

    # --------------------------------------------------
    # SONUÇ SEKMESİ
    # --------------------------------------------------
    with tab1:
        st.subheader("Analiz Sonucu")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Bakteri TPC",
                value=f"{bilimsel_10_uzeri(bakteri_tpc)} /mL"
            )

        with col2:
            st.metric(
                label="Maya-Küf",
                value=f"{bilimsel_10_uzeri(maya_kuf_tpc)} /mL"
            )

        with st.expander("Kısa özet"):
            st.write(f"Analiz edilen görüntü sayısı: **{analiz_edilen_goruntu_sayisi}**")
            st.write(f"Toplam bakteri sayısı: **{toplam_bakteri}**")
            st.write(f"Toplam maya-küf sayısı: **{toplam_maya_kuf}**")
            st.write(f"Toplam spor sayısı: **{toplam_spor}**")
            st.write(f"Ortalama bakteri/görüntü: **{ortalama_bakteri:.2f}**")
            st.write(f"Ortalama maya-küf/görüntü: **{ortalama_maya_kuf:.2f}**")

    # --------------------------------------------------
    # İŞARETLEMELER SEKMESİ
    # --------------------------------------------------
    with tab2:
        for i, gorsel in enumerate(isaretli_gorseller, start=1):
            st.subheader(f"Görüntü {i}: {gorsel['dosya_adi']}")

            st.image(
                gorsel["isaretli"],
                caption=(
                    f"Bakteri: {gorsel['bakteri']} | "
                    f"Maya-Küf: {gorsel['maya_kuf']} | "
                    f"Spor: {gorsel['spor']}"
                ),
                use_container_width=True
            )

            st.divider()

else:
    st.warning("Analiz için mikroskop görüntüsü yükle.")
