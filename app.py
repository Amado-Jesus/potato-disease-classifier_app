from PIL import Image
import streamlit as st
import torch
from utils import *




st.title("🥔 Clasificador de enfermedades del cultivo de la  papa")
st.write("Sube una imagen de una hoja para obtener la predicción del modelo.")

uploaded_file = st.file_uploader(
    "Selecciona una imagen",
    type=["jpg", "jpeg", "png","webp"]
)



@st.cache_resource
def load_model():
    
    model = CNN()
    model.load_state_dict(torch.load("modelo_enfermades_papa.pth"))
   
   
    return model

model = load_model()
# ---------------------------
# PREDICCIÓN
# ---------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
   
    

    fig = predict(
    img=image,
    model=model,
    transforms=val_transforms,
    device='cpu'
)


    st.subheader("Resultado")
    st.pyplot(fig)
   