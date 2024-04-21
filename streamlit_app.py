import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import model_from_json
import io

# Clases de salida del modelo VGG16
names = ['ARAÑA ROJA', 'MOSCA BLANCA', 'MOSCA FRUTA', 'PULGON VERDE', 'PICUDO ROJO']

def load_and_preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')
    img = img / 255.0
    return img

def model_prediction(img, model):
    img = load_and_preprocess_image(img)
    preds = model.predict(img)
    return preds

def main():
    model = None

    # Cargar arquitectura del modelo
    with open('modelo_vgg16.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)

    # Cargar pesos del modelo
    model.load_weights('/ruta/completa/a/vgg16_weights_tf_dim_ordering_tf_kernels.h5')

    st.title("bichos :sunglasses:")
    
    predictS = ""
    img_file_buffer = st.file_uploader("Carga una imagen", type=["png", "jpg", "jpeg"])
    
    if img_file_buffer is not None:
        # Convertir buffer de archivo a objeto de imagen
        img = Image.open(io.BytesIO(img_file_buffer.read()))
        st.image(img, caption="Imagen", use_column_width=False)
    
    if st.button("Predicción"):
        if model is not None:
            predictS = model_prediction(img, model)
            st.success('LA CLASE ES: {}'.format(names[np.argmax(predictS)]))
        else:
            st.error('El modelo no se ha cargado correctamente. Verifica que los archivos del modelo estén en la ubicación correcta.')

if __name__ == '__main__':
    main()
