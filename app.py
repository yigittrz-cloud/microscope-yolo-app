import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Mikroskop YOLO Analizi", layout="centered")

st.title("Mikroskop Görüntü Analizi")


@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

conf_value = st.slider("Confidence eşiği", 0.05, 0.90, 0.05, 0.05))

uploaded_file = st.file_uploader(
    "Mikroskop fotoğrafı yükle",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")



    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    results = model.predict(
        source=image_path,
        conf=conf_value
    )

    counts = {}
    result_image = None

    for r in results:
        names = r.names

        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            counts[cls_name] = counts.get(cls_name, 0) + 1

        result_image = r.plot()

    st.subheader("Model Tahmini")
    if result_image is not None:
        st.image(result_image, use_container_width=True)

    st.subheader("Sayım Sonucu")

    if counts:
        for cls_name, count in counts.items():
            st.write(f"**{cls_name}:** {count}")
    else:
        st.warning("Nesne tespit edilmedi. Confidence değerini düşürmeyi dene.")

    os.remove(image_path)
