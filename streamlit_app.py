import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image

# Load your trained model
MODEL_PATH ='model_inception.h5'
model = load_model(MODEL_PATH)

# Function to make predictions
def model_predict(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)
    preds = np.argmax(preds, axis=1)
    return preds

# Streamlit app
st.title('Plant Disease Classifier')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make prediction
    if st.button('Predict'):
        with st.spinner('Predicting...'):
            prediction = model_predict(image)
            st.success(f'The predicted class is {prediction}')

