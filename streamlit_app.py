from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

import streamlit as st
import numpy as np

# Path del modelo preentrenado
MODEL_PATH = 'modelo1_VGG16.h5'

# Tamaño de entrada esperado por el modelo VGG16
input_shape = (224, 224)

# Clases de salida del modelo VGG16
names = ['ARAÑA ROJA', 'MOSCA BLANCA', 'MOSCA FRUTA', 'PULGON VERDE', 'PICUDO ROJO']

def load_and_preprocess_image(img):
    img = img.resize(input_shape)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def model_prediction(img, model):
    img = load_and_preprocess_image(img)
    preds = model.predict(img)
    return preds

def main():
    model = VGG16(weights='imagenet', include_top=True)

    st.title("bichos :sunglasses:")
    
    predictS = ""
    img_file_buffer = st.file_uploader("Carga una imagen", type=["png", "jpg", "jpeg"])
    
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        st.image(image, caption="Imagen", use_column_width=False)
    
    if st.button("Predicción"):
         predictS = model_prediction(image, model)
         st.success('LA CLASE ES: {}'.format(names[np.argmax(predictS)]))

if __name__ == '__main__':
    main()
