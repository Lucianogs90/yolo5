import streamlit as st
import torch
from detect import detect
from PIL import Image
from io import *
from datetime import datetime
import os


def imageInput(device, src):
    
    if src == 'Upload your own data.':
        image_file = st.file_uploader("Suba uma Imagem", type=['png', 'jpeg', 'jpg', 'webp'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Imagem Original', use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts)+image_file.name)
            outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            #call Model prediction--
            model = torch.hub.load("ultralytics/yolov5", "yolov5s") 
            model.cuda() if device == 'cuda' else model.cpu()
            pred = model(imgpath)
            pred.show()  # render bbox in image
            for im in pred.ims:
                im_base64 = Image.fromarray(im)
                im_base64.save(outputpath)

            #--Display predicton            
            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='Predição do Modelo', use_column_width='always')


def main():
    # -- Sidebar
    st.sidebar.title('⚙️Options')
    datasrc = st.sidebar.radio("Select input source.", ['Upload your own data.'])
                   
    option = st.sidebar.radio("Select input type.", ['Image', 'Video'], disabled = True)
    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled = False, index=1)
    else:
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled = True, index=0)
    # -- End of Sidebar

    st.header('📦Obstacle Detection Model Demo')
    st.subheader('👈🏽 Select the options')
    st.sidebar.markdown("https://github.com/thepbordin/Obstacle-Detection-for-Blind-people-Deployment")
    if option == "Image":    
        imageInput(deviceoption, datasrc)
    # elif option == "Video": 
    #     videoInput(deviceoption, datasrc)
  

if __name__ == '__main__':
  
    main()
# @st.cache
# def loadModel():
#     start_dl = time.time()
#     model_file = wget.download('https://archive.org/download/yoloTrained/yoloTrained.pt', out="models/")
#     finished_dl = time.time()
#     print(f"Model Downloaded, ETA:{finished_dl-start_dl}")
# loadModel()