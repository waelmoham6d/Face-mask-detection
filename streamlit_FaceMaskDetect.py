import streamlit as st
import numpy as np
import matplotlib.image as mimg
import cv2
from keras.models import load_model
from PIL import Image

model = load_model(r"C:\Users\mwael\OneDrive\Desktop\home\course\mask_detectoin\saved_models\face_mask_finally.keras")


st.title("Face Mask Detection")
st.write("Upload an image to verify if the person is wearing a mask or not.")


uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


if uploaded_image is not None:
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image.", use_column_width=True)
    

    img = np.array(img)
    img_resized = cv2.resize(img, (128, 128)) 
    img_normalized = img_resized / 255.0  
    img_reshaped = np.reshape(img_normalized, [1, 128, 128, 3])  


    pred = model.predict(img_reshaped)
    st.write(f"Model prediction: {pred[0][0]:.4f}")


    if pred[0][0] > 0.6:
        
        st.write("**This person is wearing a mask.**")
    else:
        st.write("**This person is not wearing a mask.**")