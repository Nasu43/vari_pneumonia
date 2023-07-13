import streamlit as st
from PIL import Image
from keras.models import load_model

from util import classify, set_background

set_background('./bgs/bg5.png')

# set title
st.title('Pneumonia Classification')

# set header
st.header('please upload a chest X-ray image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('./model/final_pneu_cnn.h5')

# load class names
with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
