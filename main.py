import streamlit as st
import os
import pydicom as dcm
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from stqdm import stqdm
import matplotlib.pyplot as plt

st.title('Sinus')

st.header('Upload the DCM image folder')

def custom_loss(y_true, y_pred):
    # Calculate the individual losses and multiply them by their respective loss weights
    loss = tf.keras.losses.mean_absolute_error(y_true, y_pred),
    return loss

model = load_model('final_model_sinus_v2.h5')

def window_image(img, minn,maxx, intercept, slope, rescale=True):
    img = (img*slope +intercept) 
    
    img[img<minn] = minn 
    img[img>maxx] = maxx 
    if rescale: 
        img = (img - minn) / (maxx - minn)
    return img
    
def get_first_of_dicom_field_as_int(x):
    if type(x) == dcm.multival.MultiValue: return int(x[0])
    else: return int(x)
    
def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value,
                    data[('0028','1051')].value,
                    data[('0028','1052')].value,
                    data[('0028','1053')].value]
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

file = st.file_uploader("Choose files")

final_image_max = np.array([])
final_results_max = np.array([])
final_conf_max = 0

final_results = []
index_conf = []
# my_bar = st.progress(0, text='Model Loaded')
# files_num = len(files)
# print(files_num)
# final_percent = 0
# if files:
#     percent = 100/files_num


if file:
    ds = dcm.dcmread(file)
    sp = ds.PixelSpacing

    image = ds.pixel_array
    _ , _, intercept, slope = get_windowing(ds)
    image = window_image(image, 50, 150, intercept, slope)
    image = np.stack([image,image,image], axis=-1)
    # image = cv2.resize(image, (512,512))  
    # final_percent = final_percent + percent
    st.image(image, use_column_width=True)

    pred = model.predict(image.reshape(1,512,512,3)) 

    pred = np.argmax(pred, axis=-1)[0]

# if pred[4].squeeze() > final_conf_max:
#     sp = ds.PixelSpacing
#     final_conf_max = pred[4].squeeze()
#     final_image_max = image
#     final_results_max = pred
        # my_bar.progress(final_percent + 1, text='Predicting')

# my_bar.empty()

    etmoid_max = np.where(pred == 2)[0].min()
    maxila_max = np.where(pred == 1)[0].max()
    maxila_min = np.where(pred == 1)[0].min()
    maxila_h = np.abs(maxila_max - maxila_min)*sp
    etmoid_h = np.abs(maxila_min - etmoid_max)*sp

    me_ratio = maxila_h/etmoid_h

    print(final_conf_max)

    pred_img1 = np.zeros(shape=(512,512))
    pred_img1[np.where(pred == 1)] = 1
    pred_img2 = np.zeros(shape=(512,512))
    pred_img2[np.where(pred == 2)] = 1
    pred_img3 = np.zeros(shape=(512,512))
    fianl_pred = np.stack([pred_img1,pred_img2,pred_img3], axis=-1)

if file:
    st.image(fianl_pred, use_column_width=True)

    st.write('maxillary sinius height: ', maxila_h[0])
    st.write('etmoid sinius height: ', etmoid_h[0])
    st.write('maxillry/etmoid ratio: ', me_ratio[0])
