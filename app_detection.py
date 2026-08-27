import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("Denal x-ray detection")
st.write("Upload a full panaromic xray to detect disease areas")
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")
model= load_model()
upload_file=st.file_uploader("Choose a full X-ray image", type=["png", "jpg", "jpeg"])
if upload_file is not None:
    image= Image.open(upload_file).convert("RGB")
    results=model.predict(image)
    annotated_image = results[0].plot()

    st.image(annotated_image, channels="BGR", caption="Detection Result")