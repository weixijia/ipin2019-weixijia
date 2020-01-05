#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 22:53:43 2018

@author: weixijia
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, concatenate, LSTM, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, Callback, TensorBoard


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class BatchTensorBoard(TensorBoard):
    def __init__(self,log_dir='./logs',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=False):
        super(BatchTensorBoard, self).__init__()
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_images = write_images
        self.batch = 0
        self.batch_queue = set()
    
    def on_epoch_end(self, epoch, logs=None):
        pass
    
    def on_batch_end(self,batch,logs=None):
        logs = logs or {}
        
        self.batch = self.batch + 1
        
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = float(value)
            summary_value.tag = "batch_" + name
            if (name,self.batch) in self.batch_queue:
                continue
            self.writer.add_summary(summary, self.batch)
            self.batch_queue.add((name,self.batch))
        self.writer.flush()
        
def moving_average(x, n, type='simple'):
    x = np.asarray(x)
    if type == 'simple':
        weights = np.ones(n)
    else:
        weights = np.exp(np.linspace(-1., 0., n))

    weights /= weights.sum()

    a = np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = a[n]
    return a

def load_file(filepath, time_step):
    dataframe = pandas.read_csv(filepath, usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15], engine='python',skipfooter=0)
    skipfooter = len(dataframe)-((len(dataframe)//time_step)*time_step)
    dataframe = pandas.read_csv(filepath, usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15], engine='python',skipfooter=skipfooter)
    dataset = dataframe.values
    dataset = dataset.astype('float64')
    sensordata = dataset[:,0:(dataset.shape[1]-2)]
    #sample_num=dataframe.shape[0]//time_step
    if time_step==1:
        lat=np.array(dataframe['lat']).reshape(-1, 1)
        lng=np.array(dataframe['lng']).reshape(-1, 1)
    else:
        lat=(dataframe.lat.unique())[:(dataframe.shape[0]//time_step)].reshape(-1, 1)
        lng=(dataframe.lng.unique())[:(dataframe.shape[0]//time_step)].reshape(-1, 1)    
    location=np.column_stack((lat,lng))            
    return sensordata, location

def normolization(sensordata, location, time_step):  
    scaler = MinMaxScaler(feature_range=(0, 1))
    sensordata = scaler.fit_transform(sensordata)
    #lat = scaler.fit_transform(location[:,0].reshape(-1,1)) #lat=location[:,0].reshape(-1,1)
    #lng = scaler.fit_transform(location[:,1].reshape(-1,1)) #lng=location[:,1].reshape(-1,1)
    location=scaler.fit_transform(location)
    #sensordata = dataset[:,0:(dataset.shape[1]-2)]#get acc,gyr,mag
    SensorTrain=np.reshape(sensordata, ((sensordata.shape[0]//time_step),time_step,sensordata.shape[1]))
    return SensorTrain, location

def dataprocessing(filepath, feature_num, time_step):#integrate load_file and normolization functions together, just for convenience
    dataframe = pandas.read_csv(filepath, engine='python')
    skipfooter = len(dataframe)-((len(dataframe)//time_step)*time_step)
    df_length = dataframe.shape[1]
    
    if feature_num==0:
        usecols = [i for i in range(2,df_length)]
        dataframe = pandas.read_csv(filepath, usecols=usecols, engine='python', skipfooter=skipfooter)
        dataset = dataframe.values
        dataset = dataset.astype('float64')
        sensordata = dataset[:,0:(dataset.shape[1]-2)]
        lat=np.array(dataframe['lat']).reshape(-1, 1)
        lng=np.array(dataframe['lng']).reshape(-1, 1)
        location=np.column_stack((lat,lng))
        scaler = MinMaxScaler(feature_range=(0, 1))
        sensordata = scaler.fit_transform(sensordata)
        location=scaler.fit_transform(location)
        SensorTrain=np.reshape(sensordata, ((sensordata.shape[0]//time_step),time_step,sensordata.shape[1]))
        
    elif feature_num==1:
        dataframe = pandas.read_csv(filepath, engine='python',skipfooter=0)
        skipfooter = len(dataframe)-((len(dataframe)//time_step)*time_step)
        df_length = dataframe.shape[1]
        usecols = [i for i in range(2,df_length)]
        dataframe = pandas.read_csv(filepath, usecols=usecols, engine='python',skipfooter=skipfooter)
        dataset = dataframe.values
        dataset = dataset.astype('float64')
        sensordata = dataset[:,0:(dataset.shape[1]-2)]
        lat=np.array(dataframe['lat']).reshape(-1, 1)
        lng=np.array(dataframe['lng']).reshape(-1, 1)
        location=np.column_stack((lat,lng))
        scaler = MinMaxScaler(feature_range=(0, 1))
        sensordata = scaler.fit_transform(sensordata)
        location=scaler.fit_transform(location)
        SensorTrain=np.reshape(sensordata, ((sensordata.shape[0]//time_step), time_step,sensordata.shape[1]))
        location=np.reshape(location, ((location.shape[0]//time_step), time_step, location.shape[1]))
    else:
        if feature_num==3:
            usecols=[13,14,15]
        elif feature_num==9:
            usecols=[2,3,4,5,6,7,8,9,10,11,12]
        elif feature_num==12:
            usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15]
     
        dataframe = pandas.read_csv(filepath, usecols=usecols, engine='python', skipfooter=skipfooter)
        dataset = dataframe.values
        dataset = dataset.astype('float64')
        sensordata = dataset[:,0:(dataset.shape[1]-2)]
        if time_step==1:
            lat=np.array(dataframe['lat']).reshape(-1, 1)
            lng=np.array(dataframe['lng']).reshape(-1, 1)
        else:
            lat=(dataframe.lat.unique())[:(dataframe.shape[0]//time_step)].reshape(-1, 1)
            lng=(dataframe.lng.unique())[:(dataframe.shape[0]//time_step)].reshape(-1, 1)
        location=np.column_stack((lat,lng))
        scaler = MinMaxScaler(feature_range=(0, 1))
        sensordata = scaler.fit_transform(sensordata)
        location=scaler.fit_transform(location)
        SensorTrain=np.reshape(sensordata, ((sensordata.shape[0]//time_step),time_step,sensordata.shape[1]))    

    return SensorTrain, location
#include return sequence=True which reshape the label to 3d
def dataprocessing_overlap(filepath, time_step):#integrate load_file and normolization functions together, just for convenience 
    dataframe = pandas.read_csv(filepath, engine='python',skipfooter=0)
    skipfooter = len(dataframe)-((len(dataframe)//time_step)*time_step)
    df_length = dataframe.shape[1]
    usecols = [i for i in range(2,df_length)]
    dataframe = pandas.read_csv(filepath, usecols=usecols, engine='python',skipfooter=skipfooter)
    dataset = dataframe.values
    dataset = dataset.astype('float64')
    sensordata = dataset[:,0:(dataset.shape[1]-2)]
    lat=np.array(dataframe['lat']).reshape(-1, 1)
    lng=np.array(dataframe['lng']).reshape(-1, 1)
    location=np.column_stack((lat,lng))
    scaler = MinMaxScaler(feature_range=(0, 1))
    sensordata = scaler.fit_transform(sensordata)
    location=scaler.fit_transform(location)
    lat=scaler.fit_transform(lat)
    lng=scaler.fit_transform(lng)
    SensorTrain=np.reshape(sensordata, ((sensordata.shape[0]//time_step), time_step,sensordata.shape[1]))
    location=np.reshape(location, ((location.shape[0]//time_step), time_step, location.shape[1]))
    lat=np.reshape(lat, ((lat.shape[0]//time_step), time_step, lat.shape[1]))
    lng=np.reshape(lng, ((lng.shape[0]//time_step), time_step, lng.shape[1]))
    return SensorTrain, location, lat, lng

def overlapping(filepath,feature_num, time_step):
    dataframe = pandas.read_csv(filepath, engine='python',skipfooter=0)
    df_length = dataframe.shape[1]
    usecols = [i for i in range(2,df_length)]
    dataframe = pandas.read_csv(filepath, usecols=usecols, engine='python',skipfooter=0)
    dataset = dataframe.values
    dataset = dataset.astype('float64')
    sensordata = dataset[:,0:(dataset.shape[1]-2)]
    lat=np.array(dataframe['lat']).reshape(-1, 1)
    lng=np.array(dataframe['lng']).reshape(-1, 1)
    location=np.column_stack((lat,lng))
    
    ttt=np.zeros(feature_num)
    if feature_num==3:
        for i in range (len(sensordata)):
            k=sensordata[i,time_step*9:time_step*10];l=sensordata[i,time_step*10:time_step*11];m=sensordata[i,time_step*11:time_step*12]
            k=k.reshape(-1,1);l=l.reshape(-1,1);m=m.reshape(-1,1);
            abc=np.column_stack((k,l,m))
            ttt=np.vstack((ttt,abc))
        ttt=ttt[1:,:]
    elif feature_num==12:
        for i in range (len(sensordata)):
            a=sensordata[i,0:time_step];b=sensordata[i,time_step:time_step*2];c=sensordata[i,time_step*2:time_step*3]
            d=sensordata[i,time_step*3:time_step*4];e=sensordata[i,time_step*4:time_step*5];f=sensordata[i,time_step*5:time_step*6]
            g=sensordata[i,time_step*6:time_step*7];h=sensordata[i,time_step*7:time_step*8];j=sensordata[i,time_step*8:time_step*9]
            k=sensordata[i,time_step*9:time_step*10];l=sensordata[i,time_step*10:time_step*11];m=sensordata[i,time_step*11:time_step*12]
            a=a.reshape(-1,1);b=b.reshape(-1,1);c=c.reshape(-1,1);       
            d=d.reshape(-1,1);e=e.reshape(-1,1);f=f.reshape(-1,1);
            g=g.reshape(-1,1);h=h.reshape(-1,1);j=j.reshape(-1,1);
            k=k.reshape(-1,1);l=l.reshape(-1,1);m=m.reshape(-1,1);    
            abc=np.column_stack((a,b,c,d,e,f,g,h,j,k,l,m))
            ttt=np.vstack((ttt,abc))
        ttt=ttt[1:,:]
        
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    ttt = scaler.fit_transform(ttt)
    location=scaler.fit_transform(location)
    lat=scaler.fit_transform(lat)
    lng=scaler.fit_transform(lng)
    SensorTrain=np.reshape(ttt, ((ttt.shape[0]//time_step), time_step, ttt.shape[1]))
    return SensorTrain, location

def PCA_compress(SensorTrain,n_components):
    pca = PCA(n_components=n_components)
    sensor1=SensorTrain[:,:,0]
    sensor2=SensorTrain[:,:,1]
    sensor3=SensorTrain[:,:,2]
    
    newData1=pca.fit_transform(sensor1)
    newData2=pca.fit_transform(sensor2)
    newData3=pca.fit_transform(sensor3)
    
    SensorTrain=np.dstack((newData1,newData2,newData3))
    return SensorTrain

def SVD_compress(SensorTrain,n_components):
    svd = TruncatedSVD(n_components=n_components)
    sensor1=SensorTrain[:,:,0]
    sensor2=SensorTrain[:,:,1]
    sensor3=SensorTrain[:,:,2]
    
    newData1=svd.fit_transform(sensor1)
    newData2=svd.fit_transform(sensor2)
    newData3=svd.fit_transform(sensor3)
    
    SensorTrain=np.dstack((newData1,newData2,newData3))
    return SensorTrain

def SimpleDownsampling(SensorTrain, downsample_num):    
    ttt=SensorTrain[0,0,:]
    for i in range(SensorTrain.shape[0]):
        for j in range(1,SensorTrain.shape[1]):
            if j*100 > SensorTrain.shape[1]:
                break
            abc=SensorTrain[i,(j*100)-1,:]
            ttt=np.vstack((ttt,abc))
    ttt=ttt[1:,:]
    SensorTrain=np.reshape(ttt, (int(ttt.shape[0]/int(SensorTrain.shape[1]/downsample_num)), int(SensorTrain.shape[1]/downsample_num), ttt.shape[1]))
    return SensorTrain
    
def dataprocessing_stateful(filepath, time_step):#integrate load_file and normolization functions together, just for convenience 
    dataframe = pandas.read_csv(filepath, engine='python',skipfooter=0)
    skipfooter = len(dataframe)-((len(dataframe)//time_step)*time_step)
    df_length = dataframe.shape[1]
    usecols = [i for i in range(2,df_length)]
    dataframe = pandas.read_csv(filepath, usecols=usecols, engine='python',skipfooter=skipfooter)
    dataset = dataframe.values
    dataset = dataset.astype('float64')
    sensordata = dataset[:,0:(dataset.shape[1]-2)]
    lat=np.array(dataframe['lat']).reshape(-1, 1)
    lng=np.array(dataframe['lng']).reshape(-1, 1)
    location=np.column_stack((lat,lng))
    scaler = MinMaxScaler(feature_range=(0, 1))
    sensordata = scaler.fit_transform(sensordata)
    location=scaler.fit_transform(location)
    lat=scaler.fit_transform(lat)
    lng=scaler.fit_transform(lng)
    SensorTrain=sensordata
    #SensorTrain=np.reshape(sensordata, ((sensordata.shape[0]//time_step), time_step,sensordata.shape[1]))
    #location=np.reshape(location, ((location.shape[0]//time_step), time_step, location.shape[1]))
    #lat=np.reshape(lat, ((lat.shape[0]//time_step), time_step, lat.shape[1]))
    #lng=np.reshape(lng, ((lng.shape[0]//time_step), time_step, lng.shape[1]))
    return SensorTrain, location, lat, lng

def get_ave_prediction(locPrediction, n):
    weights = np.ones(n)
    weights /= weights.sum()
    x = np.asarray(locPrediction[:,0])
    y = np.asarray(locPrediction[:,1])  
    avelatPrediction = np.convolve(x, weights, mode='full')[:len(x)]
    avelngPrediction = np.convolve(y, weights, mode='full')[:len(y)]
    avelatPrediction[:n] = avelatPrediction[n]
    avelngPrediction[:n] = avelngPrediction[n]
    avelatPrediction=avelatPrediction.reshape(-1,1)
    avelngPrediction=avelngPrediction.reshape(-1,1)
    aveLocPrediction=np.column_stack((avelatPrediction,avelngPrediction))
    return aveLocPrediction

def CDF(filepath, time_step, locPrediction):
#    dataframe = pandas.read_csv(filepath, usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15], engine='python',skipfooter=0)
#    skipfooter = len(dataframe)-((len(dataframe)//time_step)*time_step)
    dataframe = pandas.read_csv(filepath)
    dataset = dataframe.values
    dataset = dataset.astype('float64')
    if time_step==1:
        lat=np.array(dataframe['lat']).reshape(-1, 1)
        lng=np.array(dataframe['lng']).reshape(-1, 1)
    else:
        lat=(dataframe.lat.unique())[:(dataframe.shape[0]//time_step)].reshape(-1, 1)
        lng=(dataframe.lng.unique())[:(dataframe.shape[0]//time_step)].reshape(-1, 1)
    location=np.column_stack((lat,lng))
    true = location
    scaler = MinMaxScaler(feature_range=(0, 1))
    location=scaler.fit_transform(location)
    prediction=scaler.inverse_transform(locPrediction)
    
    diff=[]
    for i in range (int(len(true))):
        diff_per_point=sqrt((true[i,0]-prediction[i,0])**2+(true[i,1]-prediction[i,1])**2)
        diff=np.append(diff,diff_per_point)
    
    data_size=len(diff)
 
    # Set bins edges
    data_set=sorted(set(diff))
    bins=np.append(data_set, data_set[-1]+1)
    
    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(diff, bins=bins, density=False)
    
    counts=counts.astype(float)/data_size
    
    # Find the cdf
    cdf = np.cumsum(counts)
    
    # Plot the cdf
#    plt.plot(bin_edges[0:-1], cdf,linestyle='--', color='b')
#    plt.ylim((0,1))
#    plt.ylabel("CDF")
#    plt.grid(True)
#    
#    plt.show()
    
    return bin_edges[0:-1], cdf


def inversescaler(filepath, time_step, locPrediction):
#    dataframe = pandas.read_csv(filepath, usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15], engine='python',skipfooter=0)
#    skipfooter = len(dataframe)-((len(dataframe)//time_step)*time_step)
    dataframe = pandas.read_csv(filepath)
    dataset = dataframe.values
    dataset = dataset.astype('float64')
    if time_step==1:
        lat=np.array(dataframe['lat']).reshape(-1, 1)
        lng=np.array(dataframe['lng']).reshape(-1, 1)
    else:
        lat=(dataframe.lat.unique())[:(dataframe.shape[0]//time_step)].reshape(-1, 1)
        lng=(dataframe.lng.unique())[:(dataframe.shape[0]//time_step)].reshape(-1, 1)
    location=np.column_stack((lat,lng))
    true = location
    scaler = MinMaxScaler(feature_range=(0, 1))
    location=scaler.fit_transform(location)
    prediction=scaler.inverse_transform(locPrediction)
    
    return prediction