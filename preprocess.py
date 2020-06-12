import dropbox
import numpy as np
from scipy.fft import fft,ifft
import math
from scipy.stats.stats import pearsonr
import eeglib
from scipy.stats import kurtosis, skew
import pandas as pd
from csv import writer
import sys

start = sys.argv[1]
end = sys.argv[2]
#function to return index for base normalization
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


#save the file from dropbox as musedata.csv
file_name="musedata.csv"
dropbox_path='/Apps/Muse/'
dbx=dropbox.Dropbox('un69i0ohq-AAAAAAAAAEsXY6Qwa8EbFoBLnypnIY2vteShzZ514CjVcSCI41KI0P')
entries = dbx.files_list_folder(dropbox_path).entries
for entry in entries:
    if isinstance(entry, dropbox.files.FileMetadata):
        dbx.files_download_to_file(file_name, entry.path_lower)


#preprocessing of data in muse.csv
time=list()
timediff=list()
df=pd.read_csv("musedata.csv")
df=df[['TimeStamp','RAW_TP10','RAW_AF7','RAW_AF8','RAW_TP9']]
df=df.dropna()
[m,n]=df.shape
df['TimeStamp']= pd.to_datetime(df['TimeStamp'])


#extracting time from timestamps
for i in range(m):
    a=df.iloc[i,0]
    number = str(a.second) + "."+str(a.microsecond/1000000).split(".")[1]
    time.append(float(number))

timediff = [time[n]-time[n-1] for n in range(1,len(time))]
timediff=np.asarray(timediff)
time=np.asarray(time)
timediff[timediff<0]=1+timediff[timediff<0]
time=np.cumsum(timediff)
#u, indices = np.unique(time, return_index=True)
#arr=np.arange(1,len(time),1)
#idx= np.setdiff1d(arr,indices)
#for i in range(len(idx)):
#    time[idx[i]]=time[idx[i]]+(i*0.0000001)
df.drop(df.tail(1).index,inplace=True)
df['time']=time


#continuous wavelet transform
#wavelet parameters
num_frex = 30
min_freq =  8
max_freq = 27
srate=256
frex = np.linspace(min_freq,max_freq,num_frex)
time =np.arange( -1.5,1.5,1/srate)
half_wave = round((len(time)-1)/2)
#FFT parameters
nKern = len(time)
nData = m
nConv = nKern+nData-1
#initialize output time-frequency data
baseline_window = [ 1.25, 1.5];
baseidx=np.zeros(2)
baseidx[0] = find_nearest(df['time'], baseline_window[0])
baseidx[1] = find_nearest(df['time'], baseline_window[1])
baseidx=baseidx.astype(int)
tf = [[[0]*m]*len(frex)]*4
tf=np.asarray(tf,dtype=float)
channels=['RAW_TP10','RAW_AF7','RAW_AF8','RAW_TP9']
for cyclei in range(0,4):
  dataX = fft(df[channels[cyclei]].to_numpy(),nConv)
  for i in range(1,len(frex)):
        s = 8 / (2 * math.pi * frex[i])
        cmw = np.multiply(np.exp(np.multiply(2 * complex(0,1) * math.pi * frex[i], time)), np.exp(np.divide(-time ** 2,(2 * s ** 2))))
        cmwX = fft(cmw, nConv)
        cmwX = np.divide(cmwX, max(cmwX))
        as1 =ifft(np.multiply(cmwX, dataX), nConv)
        as1 = as1[half_wave:len(as1)-half_wave+1]
        as1=np.reshape(as1, m);
        mag=np.absolute(as1)** 2
        tf[cyclei, i, :] = np.absolute(as1) ** 2;



pts=np.where(np.logical_and(df['time']>=float(start), df['time']<=float(end)))
pts=np.asarray(pts)
[m1,n1]=pts.shape
tf=tf[:,:,pts]
tf=np.reshape(tf,(4,len(frex),n1))

falpha=np.where(np.logical_and(frex>=8, frex<=13))
falpha=np.asarray(falpha)
tfalpha=[[0]*n1]*4
tfalpha=np.array(tfalpha,dtype=float)
for i in range(4):
    tfalpha[i]=np.mean(tf[i,falpha,:],axis=1)


fbeta=np.where(np.logical_and(frex>=13, frex<=27))
fbeta=np.asarray(fbeta)
tfbeta=[[0]*n1]*4
tfbeta=np.array(tfbeta,dtype=float)
for i in range(4):
    tfbeta[i]=np.mean(tf[i,fbeta,:],axis=1)

'''
lines 114-180 extract 78 features from the 
EEG data
'''
features=np.zeros(shape=86)
mobilityalpha=np.zeros(shape=4)
mobilitybeta=np.zeros(shape=4)
j=0

for i in range(4):
    features[j]=np.mean(tfalpha[i,:])
    j=j+1

for i in range(4):
    features[j]=np.mean(tfbeta[i,:])
    j=j+1

for i in range(4):
    features[j]=np.var(tfalpha[i,:])
    j=j+1

for i in range(4):
    features[j]=np.var(tfbeta[i,:])
    j=j+1


features[j]=features[0]+features[2]-(features[1]+features[3])
j=j+1;
features[j]=features[4]+features[6]-(features[5]+features[7])
j=j+1;

for i in range(4):
    for l in range(4):
        if i>l:
            features[j]=pearsonr(np.transpose(tfbeta[i,:]),np.transpose(tfbeta[l,:]))[0]
            j=j+1
            features[j]=pearsonr(np.transpose(tfalpha[i,:]),np.transpose(tfalpha[l,:]))[0]
            j=j+1
            features[j]=pearsonr(np.transpose(tfbeta[i,:]),np.transpose(tfalpha[l,:]))[0]
            j=j+1
            features[j]=pearsonr(np.transpose(tfbeta[l,:]),np.transpose(tfalpha[i,:]))[0]
            j=j+1
        if i==l:
            features[j]=pearsonr(np.transpose(tfbeta[i,:]),np.transpose(tfalpha[l,:]))[0]
            j=j+1


for i in range(4):
    features[j]=eeglib.features.hjorthMobility(tfalpha[i,:])
    j=j+1

for i in range(4):
    features[j] = eeglib.features.hjorthMobility(tfbeta[i, :])
    j = j + 1

for i in range(4):
    features[j] = eeglib.features.hjorthComplexity(tfalpha[i, :])
    j=j+1

for i in range(4):
    features[j] = eeglib.features.hjorthComplexity(tfbeta[i, :])
    j = j + 1

for i in range(4):
    features[j]=skew(tfbeta[i, :])
    j=j+1

for i in range(4):
    features[j] = skew(tfalpha[i, :])
    j=j+1

for i in range(4):
    features[j] = kurtosis(tfalpha[i, :])
    j=j+1


for i in range(4):
    features[j]=kurtosis(tfbeta[i, :])
    j=j+1


for i in range(4):
    features[j] = -np.sum(np.multiply(tfalpha[i, :], np.log(tfalpha[i, :])))
    j = j + 1


for i in range(4):
    features[j]=-np.sum(np.multiply(tfbeta[i, :],np.log(tfbeta[i, :])))
    j=j+1



with open(r'test.csv', 'a+', newline='') as write_obj:
    csv_writer = writer(write_obj)
    csv_writer.writerow(features)






