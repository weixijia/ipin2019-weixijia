import xml.etree.ElementTree as ET
import math
import numpy as numpy
import numpy as np
import pandas as pd

def raw_file_converter(rawfilepath,rawdata):

    tree=ET.parse(rawfilepath)
    root=tree.getroot()
    read_limit = len(root)
    
    data=[]
    for i in range(read_limit):
        if root[i].tag=='a':
            data.append(root[i])  
        elif root[i].tag=='g':
            data.append(root[i])
        elif root[i].tag=='m':
            data.append(root[i])
    root=data
    read_limit=len(root)
    
    if len(root) < 1:
        print('Xml file is empty.')
        exit(-1)
    a_st = {}
    g_st = {}
    m_st = {}
    max_st = 0
    
    for i in range(read_limit):  
        st = int((float(root[i].attrib['st'])-float(root[0].attrib['st']))/1e6)
        if root[i].tag == 'a':
            a_st[st] = i
            if st > max_st:
                max_st = st
        elif root[i].tag == 'g':
            g_st[st] = i
            if st > max_st:
                max_st = st
        elif root[i].tag == 'm':
            m_st[st] = i
            if st > max_st:
                max_st = st
                
    st=[]
    ax=[]
    ay=[]
    az=[]
    gx=[]
    gy=[]
    gz=[]
    mx=[]
    my=[]
    mz=[]
    
    for i in range(max_st+1):
        st = numpy.append(st, i, axis=None)
        if i in a_st:
            ax = numpy.append(ax, float(root[a_st[i]].attrib['x']), axis=None)
            ay = numpy.append(ay, float(root[a_st[i]].attrib['y']), axis=None)
            az = numpy.append(az, float(root[a_st[i]].attrib['z']), axis=None)
        else:
            ax = numpy.append(ax, numpy.NaN, axis=None)
            ay = numpy.append(ay, numpy.NaN, axis=None)
            az = numpy.append(az, numpy.NaN, axis=None)
            
        if i in g_st:
            gx = numpy.append(gx, float(root[g_st[i]].attrib['x']), axis=None)
            gy = numpy.append(gy, float(root[g_st[i]].attrib['y']), axis=None)
            gz = numpy.append(gz, float(root[g_st[i]].attrib['z']), axis=None)
        else:
            gx = numpy.append(gx, numpy.NaN, axis=None)
            gy = numpy.append(gy, numpy.NaN, axis=None)
            gz = numpy.append(gz, numpy.NaN, axis=None)
            
        if i in m_st:
            mx = numpy.append(mx, float(root[m_st[i]].attrib['x']), axis=None)
            my = numpy.append(my, float(root[m_st[i]].attrib['y']), axis=None)
            mz = numpy.append(mz, float(root[m_st[i]].attrib['z']), axis=None)
        else:
            mx = numpy.append(mx, numpy.NaN, axis=None)
            my = numpy.append(my, numpy.NaN, axis=None)
            mz = numpy.append(mz, numpy.NaN, axis=None)
    
    df = pd.DataFrame(data=st,columns=['st'])
    df['ax'] = ax
    df['ay'] = ay
    df['az'] = az
    df['gx'] = gx
    df['gy'] = gy
    df['gz'] = gz
    df['mx'] = mx
    df['my'] = my
    df['mz'] = mz
    
    df=df.drop_duplicates(subset='st', keep='first', inplace=False)
    
    df['AccTotal'] = numpy.sqrt(df['ax']**2+df['ay']**2+df['az']**2)
    df['GyrTotal'] = numpy.sqrt(df['gx']**2+df['gy']**2+df['gz']**2)
    df['MagTotal'] = numpy.sqrt(df['mx']**2+df['my']**2+df['mz']**2)
    
    df=df.interpolate(method='linear', axis=0, limit=None, inplace=False, limit_direction='forward', limit_area=None, downcast=None)
    
    for i in range(1000):
        if not math.isnan(df['gx'][i]):
            df['gx'][0]=df['gx'][i]
            df['gy'][0]=df['gy'][i]
            df['gz'][0]=df['gz'][i]
            df['GyrTotal'][0]=df['GyrTotal'][i]
            break
    for j in range(1000):
        if not math.isnan(df['ax'][j]):
            df['ax'][0]=df['ax'][j]
            df['ay'][0]=df['ay'][j]
            df['az'][0]=df['az'][j]
            df['AccTotal'][0]=df['AccTotal'][j]
            break
    for k in range(1000):
        if not math.isnan(df['mx'][k]):
            df['mx'][0]=df['mx'][k]
            df['my'][0]=df['my'][k]
            df['mz'][0]=df['mz'][k]
            df['MagTotal'][0]=df['MagTotal'][k]
            break
    
    df=df.interpolate(method='linear', axis=0, limit=None, inplace=False, limit_direction='forward', limit_area=None, downcast=None)
    df.to_csv(rawdata)
    #return df

def data_generator(time_step, sensor_data, rawdata, timerfilepath):
    df = pd.read_csv(rawdata)
    point = pd.read_csv(timerfilepath)
    point['Time']=point['Time']*1000
    
    df['lat']=numpy.NaN
    df['lng']=numpy.NaN
    
    sequcnce=0
    s=0.0
    sl=0.0
    l_s=0
    for i in range (len(df)):
        if sequcnce >len(point)-1:
                break
        if df['st'][i]==point['Time'][sequcnce]:
            df['lat'][i]=point['lat'][sequcnce]
            df['lng'][i]=point['Lng'][sequcnce]
            diff=(point['lat'][sequcnce] - s)/(i-l_s)
            difflng=(point['Lng'][sequcnce] - sl)/(i-l_s)
            counter=1
            sum=s
            suml=sl
            for j in range (l_s+1,i):
                if counter%time_step==0:
                    sum=sum+diff*time_step
                    suml=suml+difflng*time_step
                df['lat'][j]=sum
                df['lng'][j]=suml
                counter=counter+1
            
            s=point['lat'][sequcnce]
            sl=point['Lng'][sequcnce]
            sequcnce=sequcnce+1
            l_s=i
    
    df=df.drop(df[df.st < point['Time'][0]].index)
    df=df.drop(df[df.st > point['Time'][len(point)-1]].index)
    
    df.to_csv(sensor_data)
    
def overlap_generator(cover_range,over_lapping, overlap_data, rawdata,timerfilepath):
    dftest=pd.read_csv(rawdata)
    Acct=[None] * cover_range
    Gyrt=[None] * cover_range
    Magt=[None] * cover_range
    Ax=[None] * cover_range
    Ay=[None] * cover_range
    Az=[None] * cover_range
    Gx=[None] * cover_range
    Gy=[None] * cover_range
    Gz=[None] * cover_range
    Mx=[None] * cover_range
    My=[None] * cover_range
    Mz=[None] * cover_range
    time=[None] * 1
    drop_num=abs((len(dftest)+over_lapping)//cover_range*cover_range-len(dftest))
    

    if drop_num==0:
        dftest=dftest
    else:
        dftest=dftest[:-drop_num]
        
    for i in range (0, len(dftest), cover_range-over_lapping): 
        slide_window_acc=dftest['AccTotal'][i:i+cover_range]
        slide_window_gyr=dftest['GyrTotal'][i:i+cover_range]
        slide_window_mag=dftest['MagTotal'][i:i+cover_range]
        slide_window_ax=dftest['ax'][i:i+cover_range]
        slide_window_ay=dftest['ay'][i:i+cover_range]
        slide_window_az=dftest['az'][i:i+cover_range]
        slide_window_gx=dftest['gx'][i:i+cover_range]
        slide_window_gy=dftest['gy'][i:i+cover_range]
        slide_window_gz=dftest['gz'][i:i+cover_range]
        slide_window_mx=dftest['mx'][i:i+cover_range]
        slide_window_my=dftest['my'][i:i+cover_range]
        slide_window_mz=dftest['mz'][i:i+cover_range]
        cur_time=i
        if not slide_window_acc.shape[0]==cover_range:
            break
        Acct = numpy.row_stack((Acct,slide_window_acc))
        Gyrt = numpy.row_stack((Gyrt,slide_window_gyr))
        Magt = numpy.row_stack((Magt,slide_window_mag))
        Ax = numpy.row_stack((Ax,slide_window_ax))
        Ay = numpy.row_stack((Ay,slide_window_ay))
        Az = numpy.row_stack((Az,slide_window_az))
        Gx = numpy.row_stack((Gx,slide_window_gx))
        Gy = numpy.row_stack((Gy,slide_window_gy))
        Gz = numpy.row_stack((Gz,slide_window_gz))
        Mx = numpy.row_stack((Mx,slide_window_mx))
        My = numpy.row_stack((My,slide_window_my))
        Mz = numpy.row_stack((Mz,slide_window_mz))
        time = numpy.row_stack((time,cur_time))
    
    abc=np.concatenate((Acct,Gyrt,Magt,Ax,Ay,Az,Gx,Gy,Gz,Mx,My,Mz),axis=1)
    abc=np.delete(abc, 0, 0)#drop the first row which contains all None value used for generating empty array to store data
    
    abc=pd.DataFrame(abc)
    point = pd.read_csv(timerfilepath)
    point['Time']=point['Time']*1000
    
    abc['lat']=numpy.NaN
    abc['lng']=numpy.NaN
    abc.insert(loc=0, column='st', value=time[1:])
    
    sequcnce=0
    for i in range(len(abc)):
        if sequcnce >len(point)-1:
                break
        if abc['st'][i]==point['Time'][sequcnce]:
            abc['lat'][i]=point['lat'][sequcnce]
            abc['lng'][i]=point['Lng'][sequcnce]
            sequcnce=sequcnce+1
            
    if sequcnce!=5:
        for i in range (len(point)):
            index=round(point['Time'][i]/(cover_range-over_lapping))
            abc['lat'][index]=point['lat'][i]
            abc['lng'][index]=point['Lng'][i]
        
    abc=abc.interpolate(method='linear', axis=0, limit=None, inplace=False, limit_direction='forward', limit_area=None, downcast=None)
    abc=abc.drop(abc[abc.st < point['Time'][0]].index)
    abc=abc.drop(abc[abc.st > point['Time'][len(point)-1]].index)
    
    abc.reset_index(drop=True, inplace=True)
    last_row_data_lat=abc['lat'][len(abc)-2]
    last_row_data_lng=abc.lng[len(abc)-2]
    abc.lat = abc.lat.shift(1)
    abc.lng = abc.lng.shift(1)
    abc.lat[0]=last_row_data_lat
    abc.lng[0]=last_row_data_lng
    abc.to_csv(overlap_data)
