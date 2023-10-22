import streamlit as st
import os
import pydicom
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from stqdm import stqdm

st.title('Keros Typing AI')

st.header('Upload the DCM image folder')

def custom_loss(y_true, y_pred):
    # Calculate the individual losses and multiply them by their respective loss weights
    loss = tf.keras.losses.mean_absolute_error(y_true, y_pred),
    return loss

model = load_model('opmodel.h5', custom_objects={"custom_loss":custom_loss})

model.compile(optimizer='Adam', 
              loss={'x':custom_loss,'x1':custom_loss,'x2':custom_loss,'x3':custom_loss, 'x4':tf.keras.losses.binary_crossentropy}, 
              metrics={'x':'mse','x1':'mse','x2':'mse','x3':'mse','x4':'accuracy' })

def keros(m):
    if m < 4:
        return 'Keros Type 1'
    elif m > 8:
        return 'Keros Type 3'
    else:
        return 'Keros Type 2'

def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)
    
def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


def window_image(img, window_center,window_width, intercept, slope, rescale=True):

    img = (img*slope +intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    
    if rescale:
        # Extra rescaling to 0-1, not in the original notebook
        img = (img - img_min) / (img_max - img_min)
    
    return img

files = st.file_uploader("Choose files", accept_multiple_files=True)

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
for file in stqdm(files):
    try:
        ds = pydicom.dcmread(file)
        image = ds.pixel_array
        window_center , window_width, intercept, slope = get_windowing(ds)
        image = window_image(image, window_center, window_width, intercept, slope)
        image = image [::-1, :]
        image = np.stack([image,image,image], axis=-1)
        image = cv2.resize(image, (224,224))  
        # final_percent = final_percent + percent
        print(image.shape) 
        pred = model.predict(image.reshape(1,224,224,3)) 
        if pred[4].squeeze() > final_conf_max:
            sp = ds.PixelSpacing
            final_conf_max = pred[4].squeeze()
            final_image_max = image
            final_results_max = pred
        # my_bar.progress(final_percent + 1, text='Predicting')
    except:
        continue
# my_bar.empty()
print(final_conf_max)

if files:

    d = (final_results_max[0]*224).astype(np.int16).squeeze()
    d1 = (final_results_max[1]*224).astype(np.int16).squeeze()
    d2 = (final_results_max[2]*224).astype(np.int16).squeeze()
    d3 = (final_results_max[3]*224).astype(np.int16).squeeze()

    red = [1, 0 , 0]
    image = image[0]

    final_image_max[d[1]-1:d[1]+1 , d[0]-1:d[0]+1] = red
    final_image_max[d1[1]-1:d1[1]+1 , d1[0]-1:d1[0]+1] = red
    final_image_max[d2[1]-1:d2[1]+1 , d2[0]-1:d2[0]+1] = red  
    final_image_max[d3[1]-1:d3[1]+1 , d3[0]-1:d3[0]+1] = red

    r = int(np.abs((final_results_max[0]*512).squeeze()[1] - (final_results_max[1]*512).squeeze()[1])*sp[0]*100)/100
    l = int(np.abs((final_results_max[2]*512).squeeze()[1] - (final_results_max[3]*512).squeeze()[1])*sp[0]*100)/100
    b = int(np.abs((final_results_max[1]*512).squeeze()[0] - (final_results_max[3]*512).squeeze()[0])*sp[0]*100)/100


    kr = keros(r)
    kl = keros(l)

    if kr == kl:
        s = 'Symmetrical'
    else:
        s = 'Asymmetrical'

    st.write('Right side: ' , r, 'mm -  ', kr)
    st.write('Left side: ' , l, 'mm - ', kl)
    st.write('Left to Right Distance: ' , b, 'mm')

    st.write(s)
    st.image(final_image_max[::-1, :], use_column_width=True)

