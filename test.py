
import streamlit as st
import pandas as pd
import numpy as np  
import cv2 as cv
import pandas as pd
import  matplotlib.pyplot as plt 

@st.cache_data
def load_date():
    path_data_set="E:/Data_set/digit-recognizer"
    train= pd.read_csv(path_data_set+"/train.csv") #共41000行
    #test= pd.read_csv(path_data_set+"/test.csv")
    train= np.array(train) #784=28*28
    print(train.shape)
    v_test= train[-10000:,]
    test_x= v_test[:,1:]
    test_y= v_test[:,0]
    train= train[:-10000,]

    train_y= train[:,0]
    train_x= train[:,1:]
    #train_x= train_x.reshape(-1,28,28)
    train_x= train_x.reshape(-1,1,784)
    test_x= test_x.reshape(-1,1,784)
    def one_hot(y):
        res= np.zeros((y.shape[0],10))
        for i in range(y.shape[0]):
            res[i,y[i]]=1
        return res
    train_y= one_hot(train_y)
    train_y=train_y.reshape(-1,1,10)
    test_y= one_hot(test_y)
    test_y=test_y.reshape(-1,1,10)
    print("@@end")
    return train_x,train_y,test_x,test_y
train_x,train_y,test_x,test_y= load_date()
print("again")
epoch=st.slider('epoch', 2, 300)
lr=st.slider('lr', 0.000001, 0.1)
_train=epoch*10000
epoch
res_acc_train=[]
class minist_net():
    def __init__(self) :
        self.V=np.random.random((784,120))*2-1
        #self.H=np.random.random((120,10))*2-1
        self.W=np.random.random((120,10))*2-1
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
def translate(y):
    return np.argmax(y)
def test(final_net,x,y,num=10000):
    rn=0
    for i in range(num):
        index_fortest=np.random.randint(0,num)
        p_y=final_net.forward(x[index_fortest])
        t_y=y[index_fortest]
        if translate(p_y)==translate(t_y): rn+=1
    print(f"@@@猜对:{rn}/{num} | 也就是{rn/(num)*100}%")
def train(x,y):
    right_num=0
    net= minist_net()
    for i in range(1,epoch*10000+1): #test:300epoch
        index=np.random.randint(0,x.shape[0])
        x_in=x[index]
        label=y[index]
        p_y=net.forward(x_in)
        #print(f"猜测{translate(p_y)} | 实际：{translate(label)}")
        if translate(p_y)==translate(label): right_num+=1
        if i%10000==0:
            print(f"猜对:{right_num}/{i+1} | 也就是{right_num/(i+1)*100}%")
            res_acc_train.append(right_num/(i+1)*100)
            char_=np.array(res_acc_train)
            #st.line_chart(char_)
            test(net,test_x,test_y)
        net.backforward(x_in,label)
    print("训练结束")
    return net

final_net=train(train_x,train_y)
#final_net.save()
res_acc_train=np.array(res_acc_train)
print(res_acc_train)
st.line_chart(res_acc_train)