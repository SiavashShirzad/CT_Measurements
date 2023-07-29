import streamlit as st
from tensorflow.keras.models import load_model
import pydicom
import numpy as np
import cv2
st.title('Ehsan')

st.header('Upload the DCM image')

file = st.file_uploader('', type=['dcm'])

model = load_model('opmodel.h5')

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

if file:
    ds = pydicom.dcmread(file, force=True)
    image = ds.pixel_array
    window_center , window_width, intercept, slope = get_windowing(ds)
    image = window_image(image, window_center, window_width, intercept, slope)
    # image = (image*65535).astype(np.uint16)
    image = np.stack([image,image,image], axis=-1)
    image = cv2.resize(image, (224,224))
    print(image.shape)
    print(image.mean())
    image = image.reshape(1,224,224,3)
    pred = model.predict(image)

    sp = ds.PixelSpacing


    d = (pred[0]*224).astype(np.int16).squeeze()
    d1 = (pred[1]*224).astype(np.int16).squeeze()
    d2 = (pred[2]*224).astype(np.int16).squeeze()
    d3 = (pred[3]*224).astype(np.int16).squeeze()

    red = [1, 0 , 0]
    image = image[0]

    # print('Right', ((pred[0]*512).astype(np.int16).squeeze()[0] - (pred[1]*512).astype(np.int16).squeeze()[0]))
    # print('Left', ((pred[2]*512).astype(np.int16).squeeze()[0] - (pred[3]*512).astype(np.int16).squeeze()[0])*sp)

    image[d[1]-1:d[1]+1 , d[0]-1:d[0]+1] = red
    image[d1[1]-1:d1[1]+1 , d1[0]-1:d1[0]+1] = red
    image[d2[1]-1:d2[1]+1 , d2[0]-1:d2[0]+1] = red  
    image[d3[1]-1:d3[1]+1 , d3[0]-1:d3[0]+1] = red
    st.write('Right side: ' , np.abs((pred[0]*512).squeeze()[1] - (pred[1]*512).squeeze()[1])*sp[0])
    st.write('Left side: ' , np.abs((pred[2]*512).squeeze()[1] - (pred[3]*512).squeeze()[1])*sp[0])
    st.image(image, use_column_width=True)