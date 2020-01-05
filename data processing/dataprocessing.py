#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 23:25:23 2018

@author: weixijia
"""
from processing import raw_file_converter, data_generator, overlap_generator
"define fu_num, if fu_num is 1, convert raw file from xml log file; if fu_num is 2, generate machine learning file with ground truth label; if fu_num is 3, genearte overlapping dataset.overlapping rate could be adjusted by changing cover_range and over_lapping" 
"replace all Smartisan to Oneplus if you want to generate data based on Oneplus log file"
fn_num=3; "select mode"
time_step=1000; "set time step"
cover_range=1000; "set cover range, or let's say time window"
over_lapping=900; "choose overlapping rate, which means how many datapoints are repeated in next coming sample"

for file_num in range (1,15):
    
    rawfilepath='Smartisan/'+str(file_num)+'.xml'
    timerfilepath='Smartisan/'+str(file_num)+'.csv'
    overlap_data='Smartisan/'+str(file_num)+'_timestep'+str(cover_range)+'_overlap'+str(over_lapping)+'.csv'
    
    rawdata='Smartisan/Converted/'+str(file_num)+'_converted.csv'
    sensor_data='Smartisan/'+str(file_num)+'_timestep'+str(time_step)+'.csv'
    overlap_data='Smartisan/'+str(file_num)+'_timestep'+str(cover_range)+'_overlap'+str(over_lapping)+'.csv'
    
    if fn_num==1:
        print(file_num,'file is now converting from xml log file')
        raw_file_converter(rawfilepath,rawdata)
        
    elif fn_num==2:
        print(file_num,'file is now generating as machine learning dataset with ground truth label')
        data_generator(time_step, sensor_data, rawdata, timerfilepath)
        
    elif fn_num== 3:
        print('file',file_num,'is now used to genearte overlapping dataset, overlapping rate could be adjusted by changing cover_range and over_lapping') 
        overlap_generator(cover_range,over_lapping, overlap_data, rawdata,timerfilepath)
           
    print(file_num,'file is finished!')