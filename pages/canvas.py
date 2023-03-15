import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2 as cv

drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
)
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
width=st.sidebar.slider("Width: ", 1, 25, 300)
height=st.sidebar.slider("Height: ", 1, 25, 300)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)


stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

realtime_update = st.sidebar.checkbox("Update in realtime", True)

class minist_net():
    def __init__(self,w=None,v=None) :
        if w is None and v is None:
            self.V=np.random.random((784,120))*2-1
            #self.H=np.random.random((120,10))*2-1
            self.W=np.random.random((120,10))*2-1
        self.V=v
        self.W=w
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def d_sigmod(self,x):
        return x*(1-x)
    def forward(self,x):
        self.z1=np.dot(x,self.V)
        self.l1=self.sigmoid(self.z1)

        self.z2=np.dot(self.l1,self.W)
        self.l2=self.sigmoid(self.z2)
        return self.l2
    def backforward(self,x,y,lr=0.0065):
        self.loss=y-self.l2
        self.d_l2=self.loss*self.d_sigmod(self.l2)
        self.d_l1=np.dot(self.d_l2,self.W.T)*self.d_sigmod(self.l1)
        #更新权重
        self.W+=np.dot(self.l1.T,self.d_l2) *lr
        self.V+=np.dot(x.T,self.d_l1) *lr
    def save(self):
        path="E:/py_project/lab-bp-minist/"
        np.save(path+"V.npy",self.V)
        np.save(path+"W.npy",self.W)

@st.cache_data
def load_model():
    w=np.load("W.npy")
    v=np.load("V.npy")
    return minist_net(w,v)
# Create a canvas component
net=load_model()
canvas_result = st_canvas(
    fill_color="rgb(255, 255, 255)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    width=width,
    height=height,
    drawing_mode="freedraw",
    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",
)

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)
    #存储图片
    #canvas_result.image_data.save('E:/py_project/streamlit_about/test.png')
    test_arry=np.array(canvas_result.image_data)
    test_resz=cv.resize(test_arry,(28,28))
    test_resz=test_resz[:,:,0]*0.3+test_resz[:,:,1]*0.59+test_resz[:,:,2]*0.11
    test_resz=test_resz.reshape(1,784)
    print(test_resz.shape)
    p=net.forward(test_resz)
    p=np.argmax(p)
    p
    print(p)
if canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    st.dataframe(objects)