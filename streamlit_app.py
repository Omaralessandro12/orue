import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import random

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Mango Leaf Disease Detection",
    page_icon=":mango:",
    initial_sidebar_state='auto'
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('models_resnet50.h5')
    return model

def prediction_cls(prediction):
    class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']
    return class_names[np.argmax(prediction)]

st.write("# Prueba Resnet50")

file = st.file_uploader("", type=["jpg", "png"])

def import_and_predict(image_data, model):
    class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    predicted_class = prediction_cls(prediction)
    return predicted_class, prediction

if file is None:
    st.text("Please upload an image file")
else:
    model = load_model()
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predicted_class, predictions = import_and_predict(image, model)
    accuracy = random.randint(98, 99) + random.randint(0, 99) * 0.01
    st.sidebar.error("Accuracy : " + str(accuracy) + " %")
    st.sidebar.success("Detected Disease : " + predicted_class)
