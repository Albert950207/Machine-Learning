#!/usr/bin/env python
import numpy as np
from numpy import mat

my_matrix_1= np.loadtxt(open("/Users/albert/Documents/Auto-data-train.csv","rb"),delimiter=",",skiprows=0)
my_matrix_2 = np.loadtxt(open("/Users/albert/Documents/Auto-data-test.csv","rb"),delimiter=",",skiprows=0)

m_train = my_matrix_1.T
m_testing = my_matrix_2.T

def standarization(colum):
    i = 0
    for y in colum:
        y = (y - np.min(colum))/(np.max(colum)-np.min(colum))
        colum[i] = y
        i = i+1
    return colum

def prep(mtrx):
    r = len(mtrx)
    r = r - 1
    for x in list(range(0,r)):
        b = mtrx[x]
        mtrx[x]= standarization(b)
    return mtrx.T

y_train = mat(prep(m_train)[:,0]).T
x_train = mat(np.delete(prep(m_train),0,axis = 1))
y_test = mat(prep(m_testing)[:,0]).T
x_test = mat(np.delete(prep(m_testing),0,axis = 1))


def liner_train(train_x,train_y,lr,time):
    
    arr = np.array(train_x.shape,dtype = int)
    n = arr[1]
    w = np.random.randn(1,n)
    b = np.array([[1]])
    for t in range(time):
        pd = np.dot(train_x, w.T) + b
        loss=np.dot((train_y-pd).T,train_y-pd )/train_y.shape[0]
        w_gradient = -(1/train_x.shape[0])*np.dot((train_y-pd).T,train_x)
        b_gradient = -1*np.dot((train_y-pd).T,np.ones(shape=[train_x.shape[0],1]))/train_x.shape[0]
        w = w - lr*w_gradient
        b = b -lr*b_gradient
    
    return(w,b)

def prediction(test_x,test_y,train_x,train_y,lr,time):
    w_new = liner_train(train_x,train_y,lr,time)[0]
    b_new = liner_train(train_x,train_y,lr,time)[1]
    k = (np.max(test_y)-np.min(test_y))+np.min(test_y)
    y_prediction = np.dot(test_x,w_new.T)+b_new
    Error_p=abs((test_y-y_prediction)/test_y)
    
    print y_prediction
    print Error_p
    print w_new.T
    print b_new




print(prediction(x_test,y_test,x_train,y_train,0.000001,100000))
