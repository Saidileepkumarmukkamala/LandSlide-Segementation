import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
from tensorflow import keras
import h5py
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

# recall 
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# precision
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

#f1 score
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Load the model
model = keras.models.load_model('best_model.h5', custom_objects={'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m})
imga = np.zeros((1, 128, 128, 6))
threshold = 0.5

def predict(img):
    with h5py.File(img) as f:
        data = np.array(f.get('img'))
        data[np.isnan(data)] = 0.000001
        # to normalize the data 
        mid_rgb = data[:, :, 1:4].max() / 2.0
        mid_slope = data[:, :, 12].max() / 2.0
        mid_elevation = data[:, :, 13].max() / 2.0

        # ndvi calculation
        data_red = data[:, :, 3]
        data_nir = data[:, :, 7]
        data_ndvi = np.divide(data_nir - data_red,np.add(data_nir, data_red))
        
        # final array
        imga[0, :, :, 0] = 1 - data[:, :, 3] / mid_rgb  #RED
        imga[0, :, :, 1] = 1 - data[:, :, 2] / mid_rgb #GREEN
        imga[0, :, :, 2] = 1 - data[:, :, 1] / mid_rgb #BLUE
        imga[0, :, :, 3] = data_ndvi #NDVI
        imga[0, :, :, 4] = 1 - data[:, :, 12] / mid_slope #SLOPE
        imga[0, :, :, 5] = 1 - data[:, :, 13] / mid_elevation #ELEVATION

        fig,ax1= plt.subplots(1,1,figsize=(15,10))

        st.write("Given Image")
        ax1.set_title("Given RGB image")
        #ax2.set_title("NDVI")
        #ax3.set_title("Slope")
        #ax4.set_title("Elevation")
        ax1.imshow(imga[0, :, :, 0:3])
        #ax2.imshow(imga[0, :, :, 3])
        #ax3.imshow(imga[0, :, :, 4])
        #ax4.imshow(imga[0, :, :, 5])

        st.pyplot(fig)

        # predict
        pred_img = model.predict(imga)

        pred_img = (pred_img > threshold).astype(np.uint8)

        return pred_img[0, :, :, 0]

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def plot(res):
    fig,ax1 = plt.subplots(1,1,figsize=(15,10))
    ax1.set_title("Predicted Image")
    ax1.imshow(res)
    st.pyplot(fig)

def main():
    with st.sidebar:
        
        selected = option_menu('LandSlide Detection App',
                            
                            ['Home','Detector'],
                            icons=['activity','heart'],
                            default_index=0)
    
    if selected == 'Home':
        st.markdown("<h1 style='text-align: center; color: #08e08a;'>LandSlide Detector</h1>", unsafe_allow_html=True)
        st.markdown("<h4 '>A web app to detect landslides in given image using Deep Learning techiques</h4>", unsafe_allow_html=True)
        lottie_hello = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_3uLMgcknAG.json")
        st_lottie(
            lottie_hello,
            speed=1,
            reverse=False,
            loop=True,
            quality="High",
            #renderer="svg",
            height=400,
            width=-900,
            key=None,
        )
        st.markdown("<h4 '>About:</h4>", unsafe_allow_html=True)
        st.write("Landslides can cause devastating damage to infrastructure and communities, making early detection critical for effective response and mitigation efforts. In recent years, deep learning techniques have shown promising results in detecting and predicting landslides from remote sensing data.One example of a landslide detection using deep learning project involves the use of convolutional neural networks (CNNs) to analyze satellite imagery and identify areas at risk of landslides. The CNNs are trained on labeled data sets of landslide and non-landslide images, allowing the algorithm to learn the unique visual patterns associated with landslides.Once trained, the CNNs can be applied to new satellite imagery to automatically identify potential landslide sites. The results can be visualized on maps, enabling authorities to prioritize high-risk areas for further investigation and mitigation efforts.Overall, this approach offers a promising solution for early detection and prevention of landslides, helping to protect communities and infrastructure from the devastating impacts of these natural disasters.")

        st.markdown("<h4 '>Features:</h4>", unsafe_allow_html=True)
        st.write("Easy Detection of landslide in image: Just need to click and upload image.")
        st.write("Fast and Accurate: Provides the classification with high accuracy and fast")
        st.markdown("<h4 '>Architecture:</h4>", unsafe_allow_html=True)
        st.write("UNet Architecture: Model is trained on UNet architecture which is state of the art architecture for image segmentation")
        st.markdown("<h4 '>Model Performance:</h4>", unsafe_allow_html=True)
        st.write("Model is trained on 3800 images of landslide")
        st.write("Model is capable of detecting landslide in image with 99% accuracy")
        st.image('model.png')
        st.markdown("<h4 '>How to use:</h4>", unsafe_allow_html=True)
        st.write("1. Click on Detector tab on left side")
        st.write("2. Upload image")
        st.write("3. Click on Predict")
        st.write("4. Wait for few seconds")
        st.write("5. Result will be displayed")
        st.markdown("<h4 '>Credits:</h4>", unsafe_allow_html=True)
        st.write("1. Dataset: https://www.iarai.ac.at/landslide4sense/challenge/")
        lotti = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_srcvuh0h.json")
        st_lottie(
            lotti,
            speed=1,
            reverse=False,
            loop=True,
            quality="High",
            #renderer="svg",
            height=400,
            width=-900,
            key=None,
        )
    
    elif selected == 'Detector':
        st.markdown("<h1 style='text-align: center; color: #08e08a;'>LandSlide Detector</h1>", unsafe_allow_html=True)
        st.markdown("<h4 '>A web app to detect landslides in given image using Deep Learning techiques</h4>", unsafe_allow_html=True)

        st.markdown("<h4 '>Use our example images</h4>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button('Image 1'):
                resu = predict('1.h5')
                st.write('Predicted Image')
                plot(resu)
        with col2:
            if st.button('Image 2'):
                resul = predict('2.h5')
                st.write('Predicted Image')
                plot(resul)
        with col3:
            if st.button('Image 3'):
                result = predict('3.h5')
                st.write('Predicted Image')
                plot(result)
        with col4:
            if st.button('Image 4'):
                resulta = predict('4.h5')
                st.write('Predicted Image')
                plot(resulta)
        
        st.write('Or')
        st.markdown("<h4 '>Upload Image:</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image...", type="h5")
        if st.button('Predict'):
            if uploaded_file is not None:
                res = predict(uploaded_file)
                st.write('Predicted Image')
                plot(res)
if __name__ == '__main__':
    main()