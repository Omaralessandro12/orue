import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications import resnet50

# Loading the Model
model = load_model('model_inception.h5', compile=False)

st.markdown("## Object Classifier App with Deep Learning")
st.markdown("""
This app uses Deep learning (ResNet50) libraries namely keras to identify objects from images.

ResNet-50 is a convolutional neural network that is 50 layers deep. You can load a pretrained version of the network trained on more than a million images from the ImageNet database. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals.

**Made by Ifeanyi Nneji**

""")

# Uploading the dog image
object_image = st.file_uploader("Upload an image...", type=['png','jpg','webp','jpeg'])
submit = st.button('Predict')

# On predict button click
if submit:
    if object_image is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(object_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        st.image(opencv_image, channels="BGR")

        # Resize image to match model input size
        opencv_image = cv2.resize(opencv_image, (224, 224))

        # Preprocess the image for model prediction
        opencv_image = image.img_to_array(opencv_image)
        opencv_image = np.expand_dims(opencv_image, axis=0)
        opencv_image = resnet50.preprocess_input(opencv_image)

        # Get predictions from the model
        predictions = model.predict(opencv_image)
        predicted_classes = resnet50.decode_predictions(predictions, top=5)

        # Displaying the image
        st.markdown("""This is an image of: """)

        for imagenet_id, name, likelihood in predicted_classes[0]:
            st.text('- {}: {:.2f} likelihood'.format(name, likelihood))
