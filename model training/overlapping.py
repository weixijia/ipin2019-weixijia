#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: weixijia
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
import json
import tensorflow as tf
import functions
from keras.models import Sequential
from keras.layers import Dense, concatenate, LSTM, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, Callback, TensorBoard
from functions import PCA_compress, SVD_compress,SimpleDownsampling, overlapping, LossHistory, BatchTensorBoard, moving_average, load_file, normolization,dataprocessing_overlap, dataprocessing, get_ave_prediction, dataprocessing_stateful
# fix random seed for reproducibility
np.random.seed(7)
time_step=1000
epoch=100
batch_size=100
LR=0.005
average_num=100
DownSample_num=100
compress_num=100



SensorTrain1, location1 = overlapping('1_timestep1000_overlap900.csv',3, time_step)
SensorTrain2, location2 = overlapping('2_timestep1000_overlap900.csv',3, time_step)
SensorTrain3, location3 = overlapping('3_timestep1000_overlap900.csv',3, time_step)
SensorTrain4, location4 = overlapping('4_timestep1000_overlap900.csv',3, time_step)
SensorTrain5, location5 = overlapping('5_timestep1000_overlap900.csv',3, time_step)
SensorTrain6, location6 = overlapping('6_timestep1000_overlap900.csv',3, time_step)
SensorTrain7, location7 = overlapping('7_timestep1000_overlap900.csv',3, time_step)
SensorTrain8, location8 = overlapping('8_timestep1000_overlap900.csv',3, time_step)
SensorTrain9, location9 = overlapping('9_timestep1000_overlap900.csv',3, time_step)
SensorTrain10, location10 = overlapping('10_timestep1000_overlap900.csv',3, time_step)
SensorTrain11, location11 = overlapping('11_timestep1000_overlap900.csv',3, time_step)
SensorTrain12, location12 = overlapping('12_timestep1000_overlap900.csv',3, time_step)
SensorTrain13, location13 = overlapping('13_timestep1000_overlap900.csv',3, time_step)
SensorTrain14, location14 = overlapping('14_timestep1000_overlap900.csv',3, time_step)


SensorTrain=np.concatenate((SensorTrain1,SensorTrain2,SensorTrain3,SensorTrain4, SensorTrain5,SensorTrain6,SensorTrain7,SensorTrain8,SensorTrain9),axis=0)
location=np.concatenate((location1,location2,location3,location4,location5,location6,location7,location8,location9),axis=0)


Sensor_val=np.concatenate((SensorTrain10,SensorTrain11,SensorTrain12,SensorTrain13),axis=0)
loc_val=np.concatenate((location10,location11,location12,location13),axis=0)

model_2d = Sequential()
model_2d.add(LSTM(
    input_shape=(SensorTrain.shape[1], SensorTrain.shape[2]),
    units=128,
))

model_2d.add(Dense(2))

model_2d.compile(optimizer=RMSprop(LR),
                 loss='mse',metrics=['acc'])

history = model_2d.fit(SensorTrain, location,
                       validation_data=(Sensor_val,loc_val),
                       epochs=epoch, batch_size=batch_size, verbose=1,
                       #shuffle=False,
                       callbacks=[TensorBoard(log_dir='Tensorboard/svd100'),
                                  #EarlyStopping(monitor='val_loss', patience=40, verbose=1, mode='min')
                                  ]
                       )
locPrediction = model_2d.predict(SensorTrain14,batch_size=batch_size)
aveLocPrediction = get_ave_prediction(locPrediction, average_num)

# Make an example plot with two subplots...
fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax1.plot(location[:,0],location[:,1])
ax1.plot(locPrediction[:,0],locPrediction[:,1])
ax1.set_title('raw prediction')

ax2 = fig.add_subplot(2,2,2)
ax2.plot(location[:,0],location[:,1])
ax2.plot(aveLocPrediction[:,0],aveLocPrediction[:,1])
ax2.set_title('ave_'+str(average_num)+'_prediction')
# Save the full figure...
fig.savefig('overlap_time_step='+str(time_step)+'overlappedmodel.pdf')

model_2d.save('TS'+str(time_step)+'LR'+str(LR)+'overlappedmodel.h5')
print("Saved model to disk")



with open('TS'+str(time_step)+'LR'+str(LR)+'overlappedmodel.json', 'w') as fp:
    json.dump(history.history, fp)
    print("Saved history to disk")




