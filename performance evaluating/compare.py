# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 21:09:44 2018

@author: weixijia
"""
"Plot models"
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
from math import sqrt
import tensorflow as tf
import functions
import json
from keras.models import Sequential
from keras.layers import Dense, concatenate, LSTM, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, Callback, TensorBoard
from keras.models import model_from_json
from keras.models import load_model
from functions import PCA_compress, SVD_compress, CDF_OV, CDF, SimpleDownsampling, overlapping, LossHistory, BatchTensorBoard, moving_average, load_file, normolization, dataprocessing, get_ave_prediction
# fix random seed for reproducibility
np.random.seed(7)
time_step=1000 
epoch=100
batch_size=100
LR=0.005
average_num=100
downsample_num=100
#####################################################################validation
#SensorTrain9, location9 = dataprocessing('9_timestep1000.csv',3, time_step)
#SensorTrain10, location10 = dataprocessing('10_timestep1000.csv',3, time_step)
#SensorTrain11, location11 = dataprocessing('11_timestep1000.csv',3, time_step)
#Sensor_val=np.concatenate((SensorTrain9,SensorTrain10,SensorTrain11),axis=0)
#loc_val=np.concatenate((location9,location10,location11),axis=0)
#
#SensorTrainOL300_9_val, locationOL300_9_val = overlapping('9_timestep1000_overlap300.csv',3, time_step)
#SensorTrainOL300_10_val, locationOL300_10_val = overlapping('10_timestep1000_overlap300.csv',3, time_step)
#SensorTrainOL300_11_val, locationOL300_11_val = overlapping('11_timestep1000_overlap300.csv',3, time_step)
#Sensor_OL300_val=np.concatenate((SensorTrainOL300_9_val,SensorTrainOL300_10_val,SensorTrainOL300_11_val),axis=0)
#loc_OL300_val=np.concatenate((locationOL300_9_val,locationOL300_10_val,locationOL300_11_val),axis=0)
#
#SensorTrainOL500_9_val, locationOL500_9_val = overlapping('9_timestep1000_overlap500.csv',3, time_step)
#SensorTrainOL500_10_val, locationOL500_10_val = overlapping('10_timestep1000_overlap500.csv',3, time_step)
#SensorTrainOL500_11_val, locationOL500_11_val = overlapping('11_timestep1000_overlap500.csv',3, time_step)
#Sensor_OL500_val=np.concatenate((SensorTrainOL500_9_val,SensorTrainOL500_10_val,SensorTrainOL500_11_val),axis=0)
#loc_OL500_val=np.concatenate((locationOL500_9_val,locationOL500_10_val,locationOL500_11_val),axis=0)
#
#SensorTrainOL900_9_val, locationOL900_9_val = overlapping('9_timestep1000_overlap900.csv',3, time_step)
#SensorTrainOL900_10_val, locationOL900_10_val = overlapping('10_timestep1000_overlap900.csv',3, time_step)
#SensorTrainOL900_11_val, locationOL900_11_val = overlapping('11_timestep1000_overlap900.csv',3, time_step)
#Sensor_OL900_val=np.concatenate((SensorTrainOL900_9_val,SensorTrainOL900_10_val,SensorTrainOL900_11_val),axis=0)
#loc_OL900_val=np.concatenate((locationOL900_9_val,locationOL900_10_val,locationOL900_11_val),axis=0)
#
#SensorTrainDS_9_val = SimpleDownsampling(SensorTrainOL900_9_val, downsample_num)
#SensorTrainDS_10_val = SimpleDownsampling(SensorTrainOL900_10_val, downsample_num)
#SensorTrainDS_11_val = SimpleDownsampling(SensorTrainOL900_11_val, downsample_num)

valpath = '11_timestep1000.csv'
Sensor_val, loc_val = dataprocessing(valpath,3, time_step)

valpathOC300 = '11_timestep1000_overlap300.csv'
valpathOC500 = '11_timestep1000_overlap500.csv'
valpathOC900 = '11_timestep1000_overlap900.csv'
Sensor_OC300_val, loc_OC300_val = overlapping(valpathOC300,3, time_step)
Sensor_OC500_val, loc_OC500_val = overlapping(valpathOC500,3, time_step)
Sensor_OC900_val, loc_OC900_val = overlapping(valpathOC900,3, time_step)

SensorTrainDS_val = SimpleDownsampling(Sensor_OC900_val, downsample_num)
SensorTrainPCA_100_val = PCA_compress(Sensor_OC900_val, 100)
SensorTrainPCA_10_val = PCA_compress(Sensor_OC900_val, 10)
SensorTrainSVD_100_val = SVD_compress(Sensor_OC900_val, 100)
SensorTrainSVD_10_val = SVD_compress(Sensor_OC900_val, 10)
#####################################################################validation

#####################################################################test
testpath = 'test_12_timestep1000.csv'

overlappath300='12_timestep1000_overlap300.csv'
overlappath500='12_timestep1000_overlap500.csv'
overlappath900='12_timestep1000_overlap900.csv'

SensorTrain1000, location1000 = dataprocessing(testpath,3, time_step)

SensorTrainOL300, locationOL300 = overlapping(overlappath300,3, time_step)
SensorTrainOL500, locationOL500 = overlapping(overlappath500,3, time_step)
SensorTrainOL900, locationOL900 = overlapping(overlappath900,3, time_step)

SensorTrainDS = SimpleDownsampling(SensorTrainOL900, downsample_num)
SensorTrainPCA_100 = PCA_compress(SensorTrainOL900, 100)
SensorTrainSVD_100 = SVD_compress(SensorTrainOL900, 100)
SensorTrainPCA_10 = PCA_compress(SensorTrainOL900, 10)
SensorTrainSVD_10 = SVD_compress(SensorTrainOL900, 10)
#####################################################################test




model005 = load_model('TS1000LR0.05model.h5')
locPrediction005 = model005.predict(SensorTrain1000,batch_size=batch_size)
bin_edges005, cdf005 = CDF(testpath,time_step,locPrediction005)
locPrediction005val = model005.predict(Sensor_val,batch_size=batch_size)
bin_edges005val, cdf005val = CDF(valpath,time_step,locPrediction005val)
with open('TS1000LR0.05history.json', 'r') as f:
    history005 = json.load(f)


model0005 = load_model('TS1000LR0.005model.h5')
locPrediction0005 = model0005.predict(SensorTrain1000,batch_size=batch_size)
bin_edges0005, cdf0005 = CDF(testpath,time_step,locPrediction0005)
locPrediction0005val = model0005.predict(Sensor_val,batch_size=batch_size)
bin_edges0005val, cdf0005val = CDF(valpath,time_step,locPrediction0005val)
with open('TS1000LR0.005history.json', 'r') as f:
    history0005 = json.load(f)

model00005 = load_model('TS1000LR0.0005model.h5')
locPrediction00005 = model00005.predict(SensorTrain1000,batch_size=batch_size)
bin_edges00005, cdf00005 = CDF(testpath,time_step,locPrediction00005)
locPrediction00005val = model00005.predict(Sensor_val,batch_size=batch_size)
bin_edges00005val, cdf00005val = CDF(valpath,time_step,locPrediction00005val)
with open('TS1000LR0.0005history.json', 'r') as f:
    history00005 = json.load(f)




plt.plot(history005['val_loss'],label='0.05')
plt.plot(history0005['val_loss'],label='0.005')
plt.plot(history00005['val_loss'],label='0.0005')
#plt.ylim((0,1))
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title('Validation Loss')
legend = plt.legend(loc='best')
plt.savefig('1val_loss.pdf')
plt.show()

plt.plot(history005['val_acc'],label='0.05')
plt.plot(history0005['val_acc'],label='0.005')
plt.plot(history00005['val_acc'],label='0.0005')

#plt.ylim((0,1))
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.title('Validation Accuracy')
legend = plt.legend(loc='best')
plt.savefig('1val_acc.pdf')
plt.show()

plt.plot(bin_edges005val, cdf005val,linestyle='--',label='0.05')
plt.plot(bin_edges0005val, cdf0005val,linestyle='--',label='0.005')
plt.plot(bin_edges00005val, cdf00005val,linestyle='--',label='0.0005')
plt.ylabel("CDF")
plt.xlabel("metres")
plt.grid(True)
plt.title('Validation CDF')
legend = plt.legend(loc='best')
plt.savefig('1val_cdf.pdf')
plt.show()


plt.plot(bin_edges005, cdf005,linestyle='--',label='0.05')
plt.plot(bin_edges0005, cdf0005,linestyle='--',label='0.005')
plt.plot(bin_edges00005, cdf00005,linestyle='--',label='0.0005')
plt.ylabel("CDF")
plt.xlabel("metres")
plt.grid(True)
plt.title('Test CDF')
legend = plt.legend(loc='best')
plt.savefig('1test_cdf.pdf')
plt.show()
#################################################################################



modelSGD = load_model('TS_SGD1000LR0.005model.h5')
locPredictionSGD = modelSGD.predict(SensorTrain1000,batch_size=batch_size)
bin_edgesSGD, cdfSGD = CDF(testpath,time_step,locPredictionSGD)
locPredictionSGDval = modelSGD.predict(Sensor_val,batch_size=batch_size)
bin_edgesSGDval, cdfSGDval = CDF(valpath,time_step,locPredictionSGDval)
with open('TS_SGD1000LR0.005history.json', 'r') as f:
    historySGD = json.load(f)


modelADAM = load_model('TS_adam1000LR0.005model.h5')
locPredictionADAM = modelADAM.predict(SensorTrain1000,batch_size=batch_size)
bin_edgesADAM, cdfADAM = CDF(testpath,time_step,locPredictionADAM)
locPredictionADAMval = modelADAM.predict(Sensor_val,batch_size=batch_size)
bin_edgesADAMval, cdfADAMval = CDF(valpath,time_step,locPredictionADAMval)
with open('TS_adam1000LR0.005history.json', 'r') as f:
    historyADAM = json.load(f)




plt.plot(historySGD['val_loss'],label='SGD')
plt.plot(history0005['val_loss'],label='RMSprop')
#plt.plot(historyADAM['val_loss'],label='Adam')
#plt.ylim((0,1))
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title('Validation Loss')
legend = plt.legend(loc='best')
plt.savefig('2val_loss.pdf')
plt.show()

plt.plot(historySGD['val_acc'],label='SGD')
plt.plot(history0005['val_acc'],label='RMSprop')
#plt.plot(historyADAM['val_acc'],label='Adam')

#plt.ylim((0,1))
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.title('Validation Accuracy')
legend = plt.legend(loc='best')
plt.savefig('2val_acc.pdf')
plt.show()

plt.plot(bin_edgesSGDval, cdfSGDval,linestyle='--',label='SGD')
plt.plot(bin_edgesADAMval, cdfADAMval,linestyle='--',label='Adam')
plt.plot(bin_edges0005val, cdf0005val,linestyle='--',label='RMSprop')
plt.ylabel("CDF")
plt.xlabel("metres")
plt.grid(True)
plt.title('Validation CDF')
legend = plt.legend(loc='best')
plt.savefig('2val_cdf.pdf')
plt.show()


plt.plot(bin_edgesSGD, cdfSGD,linestyle='--',label='SGD')
plt.plot(bin_edgesADAM, cdfADAM,linestyle='--',label='Adam')
plt.plot(bin_edges0005, cdf0005,linestyle='--',label='RMSprop')
plt.ylabel("CDF")
plt.xlabel("metres")
plt.grid(True)
plt.title('Test CDF')
legend = plt.legend(loc='best')
plt.savefig('2test_cdf.pdf')
plt.show()
#################################################################################


model1round = load_model('TS_SGD1000LR0.005model.h5')
locPrediction1round = model1round.predict(SensorTrain1000,batch_size=batch_size)
bin_edges1round, cdf1round = CDF(testpath,time_step,locPrediction1round)
locPrediction1roundval = model1round.predict(Sensor_val,batch_size=batch_size)
bin_edges1roundval, cdf1roundval = CDF(valpath,time_step,locPrediction1roundval)
with open('TS1000LR0.005_1rounds_traditional_history.json', 'r') as f:
    history1round = json.load(f)


model8round = load_model('TS1000LR0.005_8rounds_traditional_model.h5')
locPrediction8round = model8round.predict(SensorTrain1000,batch_size=batch_size)
bin_edges8round, cdf8round = CDF(testpath,time_step,locPrediction8round)
locPrediction8roundval = model8round.predict(Sensor_val,batch_size=batch_size)
bin_edges8roundval, cdf8roundval = CDF(valpath,time_step,locPrediction8roundval)
with open('TS1000LR0.005_8rounds_traditional_history.json', 'r') as f:
    history8round = json.load(f)


plt.plot(history1round['val_loss'],label='1round')
plt.plot(history8round['val_loss'],label='8rounds')

#plt.ylim((0,1))
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title('Validation Loss')
legend = plt.legend(loc='best')
plt.savefig('3val_loss.pdf')
plt.show()

plt.plot(history1round['val_acc'],label='1round')
plt.plot(history8round['val_acc'],label='8rounds')


#plt.ylim((0,1))
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.title('Validation Accuracy')
legend = plt.legend(loc='best')
plt.savefig('3val_acc.pdf')
plt.show()


plt.plot(bin_edges1roundval, cdf1roundval,linestyle='--',label='1round')
plt.plot(bin_edges8roundval, cdf8roundval,linestyle='--',label='8rounds')
plt.ylabel("CDF")
plt.xlabel("metres")
plt.grid(True)
plt.title('Validation CDF')
legend = plt.legend(loc='best')
plt.savefig('3val_cdf.pdf')
plt.show()


plt.plot(bin_edges1round, cdf1round,linestyle='--',label='1round')
plt.plot(bin_edges8round, cdf8round,linestyle='--',label='8rounds')
plt.ylabel("CDF")
plt.xlabel("metres")
plt.grid(True)
plt.title('Test CDF')
legend = plt.legend(loc='best')
plt.savefig('3test_cdf.pdf')
plt.show()
#################################################################################



modelOL300 = load_model('TS1000LR0.005overlap300_model.h5')
locPredictionOL300 = modelOL300.predict(SensorTrainOL300,batch_size=batch_size)
bin_edgesOL300, cdfOL300 = CDF_OV(overlappath300,time_step,locPredictionOL300)
locPredictionOL300val = modelOL300.predict(Sensor_OC300_val,batch_size=batch_size)
bin_edgesOL300val, cdfOL300val = CDF_OV(valpathOC300,time_step,locPredictionOL300val)
with open('TS1000LR0.005overlap300_history.json', 'r') as f:
    historyOL300 = json.load(f)

modelOL500 = load_model('TS1000LR0.005overlap500_model.h5')
locPredictionOL500 = modelOL500.predict(SensorTrainOL500,batch_size=batch_size)
bin_edgesOL500, cdfOL500 = CDF_OV(overlappath500,time_step,locPredictionOL500)
locPredictionOL500val = modelOL500.predict(Sensor_OC500_val,batch_size=batch_size)
bin_edgesOL500val, cdfOL500val = CDF_OV(valpathOC500,time_step,locPredictionOL500val)
with open('TS1000LR0.005overlap500_history.json', 'r') as f:
    historyOL500 = json.load(f)

modelOL900 = load_model('TS1000LR0.005overlap900_model.h5')
locPredictionOL900 = modelOL900.predict(SensorTrainOL900,batch_size=batch_size)
bin_edgesOL900, cdfOL900 = CDF_OV(overlappath900,time_step,locPredictionOL900)
locPredictionOL900val = modelOL900.predict(Sensor_OC900_val,batch_size=batch_size)
bin_edgesOL900val, cdfOL900val = CDF_OV(valpathOC900,time_step,locPredictionOL900val)
with open('TS1000LR0.005overlap900_history.json', 'r') as f:
    historyOL900 = json.load(f)



plt.plot(historyOL300['val_loss'],label='Overlap: 30%')
plt.plot(historyOL500['val_loss'],label='Overlap: 50%')
plt.plot(historyOL900['val_loss'],label='Overlap: 90%')

#plt.ylim((0,1))
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title('Validation Loss')
legend = plt.legend(loc='best')
plt.savefig('4val_loss.pdf')
plt.show()

plt.plot(historyOL300['val_acc'],label='Overlap: 30%')
plt.plot(historyOL500['val_acc'],label='Overlap: 50%')
plt.plot(historyOL900['val_acc'],label='Overlap: 90%')


#plt.ylim((0,1))
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.title('Validation Accuracy')
legend = plt.legend(loc='best')
plt.savefig('4val_acc.pdf')
plt.show()

plt.plot(bin_edgesOL300val, cdfOL300val,linestyle='--',label='Overlap: 30%')
plt.plot(bin_edgesOL500val, cdfOL500val,linestyle='--',label='Overlap: 50%')
plt.plot(bin_edgesOL900val, cdfOL900val,linestyle='--',label='Overlap: 90%')
plt.ylabel("CDF")
plt.xlabel("metres")
plt.grid(True)
plt.title('Validation CDF')
legend = plt.legend(loc='best')
plt.savefig('4val_cdf.pdf')
plt.show()


plt.plot(bin_edgesOL300, cdfOL300,linestyle='--',label='Overlap: 30%')
plt.plot(bin_edgesOL500, cdfOL500,linestyle='--',label='Overlap: 50%')
plt.plot(bin_edgesOL900, cdfOL900,linestyle='--',label='Overlap: 90%')
plt.ylabel("CDF")
plt.xlabel("metres")
plt.grid(True)
plt.title('Test CDF')
legend = plt.legend(loc='best')
plt.savefig('4test_cdf.pdf')
plt.show()
#################################################################################


modeldownsampleOL900 = load_model('TS1000LR0.005downsampled_overlap900_model.h5')
locPredictiondownsampleOL900 = modeldownsampleOL900.predict(SensorTrainDS,batch_size=batch_size)
bin_edgesdownsampleOL900, cdfdownsampleOL900 = CDF_OV(overlappath900,time_step,locPredictiondownsampleOL900)
locPredictiondownsampleOL900val = modeldownsampleOL900.predict(SensorTrainDS_val,batch_size=batch_size)
bin_edgesdownsampleOL900val, cdfdownsampleOL900val = CDF_OV(valpathOC900,time_step,locPredictiondownsampleOL900val)
with open('TS1000LR0.005downsampled_overlap500_history.json', 'r') as f:
    historydownsampleOL900 = json.load(f)
################################################################################# Add OnePlus Test
OnePluspath900='1_timestep1000_overlap900Oneplus.csv'
OnePlusOL900, OnePlus_locationOL900 = overlapping(OnePluspath900,3, time_step)
OnePlusDS = SimpleDownsampling(OnePlusOL900, downsample_num)
OnePlus = modeldownsampleOL900.predict(OnePlusDS,batch_size=batch_size)
bin_OnePlus, cdfOnePlus = CDF_OV(OnePluspath900,time_step,OnePlus)
#################################################################################

plt.plot(historydownsampleOL900['val_loss'],label='Down Sampled')
plt.plot(historyOL900['val_loss'],label='Original')
#plt.ylim((0,1))
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title('Validation Loss')
legend = plt.legend(loc='best')
plt.savefig('5val_loss.pdf')
plt.show()


plt.plot(historydownsampleOL900['val_acc'],label='Down Sampled')
plt.plot(historyOL900['val_acc'],label='Original')
#plt.ylim((0,1))
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.title('Validation Accuracy')
legend = plt.legend(loc='best')
plt.savefig('5val_acc.pdf')
plt.show()



plt.plot(bin_edgesdownsampleOL900val, cdfdownsampleOL900val,'--',label='Down Sampled')
plt.plot(bin_edgesOL900val, cdfOL900val,linestyle='--',label='Original')
plt.ylabel("CDF")
plt.xlabel("metres")
plt.grid(True)
plt.title('Validation CDF')
legend = plt.legend(loc='best')
plt.savefig('5val_cdf.pdf')
plt.show()


plt.plot(bin_edgesdownsampleOL900, cdfdownsampleOL900,linestyle='--',label='Down Sampled')
plt.plot(bin_edgesOL900, cdfOL900,linestyle='--',label='Original')
plt.ylabel("CDF")
plt.xlabel("metres")
plt.grid(True)
plt.title('Test CDF')
legend = plt.legend(loc='best')
plt.savefig('5test_cdf.pdf')
plt.show()
#################################################################################




#################################################################################


model_pca100 = load_model('TS1000LR0.005pca100_model.h5')
locPrediction_pca100 = model_pca100.predict(SensorTrainPCA_100,batch_size=batch_size)
bin_edges_pca100, cdf_pca100 = CDF(testpath,time_step,locPrediction_pca100)
locPrediction_pca100val = model_pca100.predict(SensorTrainPCA_100_val,batch_size=batch_size)
bin_edges_pca100val, cdf_pca100val = CDF_OV(valpathOC900,time_step,locPrediction_pca100val)
with open('TS1000LR0.005pca100_history.json', 'r') as f:
    history_pca100 = json.load(f)


model_pca10 = load_model('TS1000LR0.005pca10_model.h5')
locPrediction_pca10 = model_pca10.predict(SensorTrainPCA_10,batch_size=batch_size)
bin_edges_pca10, cdf_pca10 = CDF(testpath,time_step,locPrediction_pca10)
locPrediction_pca10val = model_pca10.predict(SensorTrainPCA_10_val,batch_size=batch_size)
bin_edges_pca10val, cdf_pca10val = CDF_OV(valpathOC900,time_step,locPrediction_pca10val)
with open('TS1000LR0.005pca10_history.json', 'r') as f:
    history_pca10 = json.load(f)


model_svd100 = load_model('TS1000LR0.005svd100_model.h5')
locPrediction_svd100 = model_svd100.predict(SensorTrainSVD_100,batch_size=batch_size)
bin_edges_svd100, cdf_svd100 = CDF(testpath,time_step,locPrediction_svd100)
locPrediction_svd100val = model_svd100.predict(SensorTrainSVD_100_val,batch_size=batch_size)
bin_edges_svd100val, cdf_svd100val = CDF_OV(valpathOC900,time_step,locPrediction_svd100val)
with open('TS1000LR0.005svd100_history.json', 'r') as f:
    history_svd100 = json.load(f)


model_svd10 = load_model('TS1000LR0.005svd10_model.h5')
locPrediction_svd10 = model_svd10.predict(SensorTrainSVD_10,batch_size=batch_size)
bin_edges_svd10, cdf_svd10 = CDF(testpath,time_step,locPrediction_svd10)
locPrediction_svd10val = model_svd10.predict(SensorTrainSVD_10_val,batch_size=batch_size)
bin_edges_svd10val, cdf_svd10val = CDF_OV(valpathOC900,time_step,locPrediction_svd10val)
with open('TS1000LR0.005svd10_history.json', 'r') as f:
    history_svd10 = json.load(f)


plt.plot(history_pca100['val_loss'],label='PCA: 100')
plt.plot(history_pca10['val_loss'],label='PCA: 10')
plt.plot(history_svd100['val_loss'],label='SVD: 100')
plt.plot(history_svd10['val_loss'],label='SVD: 10')

#plt.ylim((0,1))
plt.ylabel("loss")
plt.xlabel("epoch")
plt.title('Validation Loss')
legend = plt.legend(loc='best')
plt.savefig('6val_loss.pdf')
plt.show()

plt.plot(history_pca100['val_acc'],label='PCA: 100')
plt.plot(history_pca10['val_acc'],label='PCA: 10')
plt.plot(history_svd100['val_acc'],label='SVD: 100')
plt.plot(history_svd10['val_acc'],label='SVD: 10')


#plt.ylim((0,1))
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.title('Validation Accuracy')
legend = plt.legend(loc='best')
plt.savefig('6val_acc.pdf')
plt.show()


plt.plot(bin_edges_pca100val, cdf_pca100val,linestyle='--',label='PCA: 100')
plt.plot(bin_edges_pca10val, cdf_pca10val,linestyle='--',label='PCA: 10')
plt.plot(bin_edges_svd100val, cdf_svd100val,linestyle='--',label='SVD: 100')
plt.plot(bin_edges_svd10val, cdf_svd10val,linestyle='--',label='SVD: 10')
plt.ylabel("CDF")
plt.xlabel("metres")
plt.grid(True)
plt.title('Validation CDF')
legend = plt.legend(loc='best')
plt.savefig('6val_cdf.pdf')
plt.show()


plt.plot(bin_edges_pca100, cdf_pca100,linestyle='--',label='PCA: 100')
plt.plot(bin_edges_pca10, cdf_pca10,linestyle='--',label='PCA: 10')
plt.plot(bin_edges_svd100, cdf_svd100,linestyle='--',label='SVD: 100')
plt.plot(bin_edges_svd10, cdf_svd10,linestyle='--',label='SVD: 10')
plt.ylabel("CDF")
plt.xlabel("metres")
plt.grid(True)
plt.title('Test CDF')
legend = plt.legend(loc='best')
plt.savefig('6test_cdf.pdf')
plt.show()
#################################################################################



plt.plot(bin_edgesdownsampleOL900, cdfdownsampleOL900,linestyle='--',label='LR0.005 RMSprop 8rounds Overlap: 90% Down Sampled')
plt.plot(bin_edges_pca100, cdf_pca100,linestyle='--',label='PCA: 100')
plt.plot(bin_edges_pca10, cdf_pca10,linestyle='--',label='PCA: 10')
plt.plot(bin_edges_svd100, cdf_svd100,linestyle='--',label='SVD: 100')
plt.plot(bin_edges_svd10, cdf_svd10,linestyle='--',label='SVD: 10')
plt.plot(bin_edgesOL900, cdfOL900,linestyle='--',label='LR0.005 RMSprop 8rounds Overlap: 90%')
plt.plot(bin_edgesOL500, cdfOL500,linestyle='--',label='LR0.005 RMSprop 8rounds Overlap: 50%')
plt.plot(bin_edgesOL300, cdfOL300,linestyle='--',label='LR0.005 RMSprop 8rounds Overlap: 30%')
plt.plot(bin_edges8round, cdf8round,linestyle='--',label='LR0.005 RMSprop 8rounds')
plt.plot(bin_edges1round, cdf1round,linestyle='--',label='LR0.005 RMSprop 1round')

plt.plot(bin_edgesADAM, cdfADAM,linestyle='--',label='LR0.005 Adam')
plt.plot(bin_edgesSGD, cdfSGD,linestyle='--',label='LR0.005 SGD')

plt.plot(bin_edges0005, cdf0005,linestyle='--',label='LR0.005 RMSprop')
plt.plot(bin_edges00005, cdf00005,linestyle='--',label='LR0.0005 RMSprop')

plt.ylabel("CDF")
plt.xlabel("metres")
#plt.xlim((0,50))
plt.grid(True)
plt.title('Test CDF')
legend = plt.legend(loc="best")
plt.savefig('Overall_test_cdf.pdf')
plt.show()

#################################################################################

plt.plot(bin_edgesdownsampleOL900, cdfdownsampleOL900,linestyle='--',label='Smartisan')
plt.plot(bin_OnePlus, cdfOnePlus,linestyle='--',label='OnePlus')
plt.ylabel("CDF")
plt.xlabel("metres")
#plt.xlim((0,50))
plt.grid(True)
plt.title('Smartisan_VS_OnePlus Test CDF')
legend = plt.legend(loc="best")
plt.savefig('Smartisan_VS_OnePlus_cdf.pdf')
plt.show()

#################################################################################
Smartisan=locPredictiondownsampleOL900
OnePlus
OnePlus_locationOL900


aveSmartisan = get_ave_prediction(Smartisan, average_num)
fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax1.plot(OnePlus_locationOL900[:,0],OnePlus_locationOL900[:,1])
ax1.plot(Smartisan[:,0],Smartisan[:,1])
ax1.set_title('Raw Prediction')

ax2 = fig.add_subplot(2,2,2)
ax2.plot(OnePlus_locationOL900[:,0],OnePlus_locationOL900[:,1])
ax2.plot(aveSmartisan[:,0],aveSmartisan[:,1])
ax2.set_title('Filterd Route')
# Save the full figure...
fig.savefig('SmartisanTest.pdf')

#################################################################################
aveOnePlus = get_ave_prediction(OnePlus, average_num)
fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax1.plot(OnePlus_locationOL900[:,0],OnePlus_locationOL900[:,1])
ax1.plot(OnePlus[:,0],OnePlus[:,1])
ax1.set_title('Raw Prediction')

ax2 = fig.add_subplot(2,2,2)
ax2.plot(OnePlus_locationOL900[:,0],OnePlus_locationOL900[:,1])
ax2.plot(aveOnePlus[:,0],aveOnePlus[:,1])
ax2.set_title('Filterd Route')

# Save the full figure...
fig.savefig('OnePlusTest.pdf')
#################################################################################
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.plot(OnePlus_locationOL900[:,0],OnePlus_locationOL900[:,1])
ax1.plot(Smartisan[:,0],Smartisan[:,1])
ax1.set_title('Raw Prediction')

ax2 = fig.add_subplot(2,2,2)
ax2.plot(OnePlus_locationOL900[:,0],OnePlus_locationOL900[:,1])
ax2.plot(OnePlus[:,0],OnePlus[:,1])
ax2.set_title('Raw Prediction')
# Save the full figure...
fig.savefig('Raw_Comparison.pdf')
#################################################################################
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.plot(OnePlus_locationOL900[:,0],OnePlus_locationOL900[:,1])
ax1.plot(aveSmartisan[:,0],aveSmartisan[:,1])
ax1.set_title('Filterd Route')

ax2 = fig.add_subplot(2,2,2)
ax2.plot(OnePlus_locationOL900[:,0],OnePlus_locationOL900[:,1])
ax2.plot(aveOnePlus[:,0],aveOnePlus[:,1])
ax2.set_title('Filterd Route')
# Save the full figure...
fig.savefig('Raw_Comparison.pdf')
#################################################################################
#time_step=100
#testpath = str(time_step)+'.csv'
#SensorTrain100, location100 = dataprocessing(testpath, time_step)
#model100 = load_model(str(time_step)+'model.h5')
#locPrediction100 = model100.predict(SensorTrain100,batch_size=batch_size)
#bin_edges100, cdf100 = CDF(testpath,time_step,locPrediction100)
#with open(str(time_step)+'history.json', 'r') as f:
#    history100 = json.load(f)
#    
#time_step=1000
#testpath = str(time_step)+'.csv'
#SensorTrain1000, location1000 = dataprocessing(testpath, time_step)
#model1000 = load_model(str(time_step)+'model.h5')
#locPrediction1000 = model1000.predict(SensorTrain1000,batch_size=batch_size)
#bin_edges1000, cdf1000 = CDF(testpath,time_step,locPrediction1000)
#with open(str(time_step)+'history.json', 'r') as f:
#    history1000 = json.load(f)
#    
#
#time_step=2000
#testpath = str(time_step)+'.csv'
#SensorTrain2000, location2000 = dataprocessing(testpath, time_step)
#model2000 = load_model(str(time_step)+'model.h5')
#locPrediction2000 = model2000.predict(SensorTrain2000,batch_size=batch_size)
#bin_edges2000, cdf2000 = CDF(testpath,time_step,locPrediction2000)
#with open(str(time_step)+'history.json', 'r') as f:
#    history2000 = json.load(f)
#
#plt.plot(bin_edges10, cdf10,linestyle='--',label='10ms')
#plt.plot(bin_edges100, cdf100,linestyle='--',label='100ms')
#plt.plot(bin_edges1000, cdf1000,linestyle='--',label='1000ms')
#plt.plot(bin_edges2000, cdf2000,linestyle='--',label='2000ms')
#plt.ylabel("CDF")
#plt.xlabel("metres")
#plt.grid(True)
#legend = plt.legend(loc='best')
#plt.savefig('cdf.pdf')
#plt.show()
#
#plt.plot(history10['val_loss'][0:10],label='10ms')
#plt.plot(history100['val_loss'][0:10],label='100ms')
#plt.plot(history1000['val_loss'][0:10],label='1000ms')
#plt.plot(history2000['val_loss'][0:10],label='2000ms')
#plt.ylim((0,1))
#plt.ylabel("loss")
#plt.xlabel("epoch")
#legend = plt.legend(loc='best')
#plt.savefig('val_loss.pdf')
#plt.show()
#
#plt.plot(history10['val_acc'],label='10ms')
#plt.plot(history100['val_acc'],label='100ms')
#plt.plot(history1000['val_acc'],label='1000ms')
#plt.plot(history2000['val_acc'],label='2000ms')
##plt.ylim((0,1))
#plt.ylabel("loss")
#plt.xlabel("epoch")
#legend = plt.legend(loc='best')
#plt.savefig('val_acc.pdf')
#plt.show()
#
#
#aveLocPrediction10 = get_ave_prediction(locPrediction10, average_num)
#fig = plt.figure()
#
#ax1 = fig.add_subplot(2,2,1)
#ax1.plot(location10[:,0],location10[:,1])
#ax1.plot(locPrediction10[:,0],locPrediction10[:,1])
#ax1.set_title('raw prediction')
#
#ax2 = fig.add_subplot(2,2,2)
#ax2.plot(location10[:,0],location10[:,1])
#ax2.plot(aveLocPrediction10[:,0],aveLocPrediction10[:,1])
#ax2.set_title('ave_'+str(average_num)+'_prediction')
## Save the full figure...
#fig.savefig('time_step_10.pdf')
#
#
#aveLocPrediction100 = get_ave_prediction(locPrediction100, average_num)
#fig = plt.figure()
#
#ax1 = fig.add_subplot(2,2,1)
#ax1.plot(location100[:,0],location100[:,1])
#ax1.plot(locPrediction100[:,0],locPrediction100[:,1])
#ax1.set_title('raw prediction')
#
#ax2 = fig.add_subplot(2,2,2)
#ax2.plot(location100[:,0],location100[:,1])
#ax2.plot(aveLocPrediction100[:,0],aveLocPrediction100[:,1])
#ax2.set_title('ave_'+str(average_num)+'_prediction')
## Save the full figure...
#fig.savefig('time_step_100.pdf')
#
#aveLocPrediction1000 = get_ave_prediction(locPrediction1000, average_num)
#fig = plt.figure()
#
#ax1 = fig.add_subplot(2,2,1)
#ax1.plot(location1000[:,0],location1000[:,1])
#ax1.plot(locPrediction1000[:,0],locPrediction1000[:,1])
#ax1.set_title('raw prediction')
#
#ax2 = fig.add_subplot(2,2,2)
#ax2.plot(location1000[:,0],location1000[:,1])
#ax2.plot(aveLocPrediction1000[:,0],aveLocPrediction1000[:,1])
#ax2.set_title('ave_'+str(average_num)+'_prediction')
## Save the full figure...
#fig.savefig('time_step_1000.pdf')
#
#
#
#
