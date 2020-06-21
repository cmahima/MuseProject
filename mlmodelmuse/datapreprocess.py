import numpy as np
import scipy.io
import scipy.signal as sig
import pywt
import os.path
import pickle
import pandas as pd 
import matplotlib.pyplot as plt

df=pd.read_csv("/Users/mahima/research/df1blue0.csv")

df1=df[['RAW_AF7','RAW_TP10','RAW_TP9','RAW_AF8']]
#n=(np.transpose(df1['RAW_AF7'].to_numpy()))
df1=df1.dropna()

def multch_wavelet(data, wavelet='db4', mode=pywt.MODES.zpd, numLevels=4):
    coeffs = list()
    for i, channel in enumerate(data):
        #print(channel)
        coeffs.append(pywt.wavedec(channel, wavelet, mode, numLevels))
    return coeffs
#n=(np.transpose(df1['RAW_AF7'].to_numpy()))

def artfct_chk(data, threshold=0, preSize=10, postSize=20):
    flags = np.abs(data) > threshold
    rows, cols = np.nonzero(flags)
    indices = zip(rows, cols)
    for ind in indices:
        pre = max(1, ind[1] - preSize)
        post = min(data.shape[1], ind[1] + postSize)
        flags[ind[0], pre:post] = True
    return flags

f=artfct_chk(np.transpose(df1.to_numpy()))
#print(f)





def baseline_correct(data, low_fr=7, high_fr=13, order=100):
    for i, ch in enumerate(data):
        #print(ch)
        meanData = ch.mean()
        ch = ch - meanData
        # Band-pass filter between low_fr and high_fr
        h = sig.firwin(order, [low_fr, high_fr], nyq=128, pass_zero=False)
        data[i, :] = sig.filtfilt(h, 1, ch)
    return data

'''
d=(df1.to_numpy()[:,3])
baseline=baseline_correct(np.transpose(df1.to_numpy()))
m=multch_wavelet(np.transpose(df1.to_numpy()))
print(len(m[0]))
time=np.linspace(1, len(m[0][0]), num=len(m[0][0]))
time1=np.linspace(1,1020, num=1020)
plt.plot(time,m[0][0])
k=m[0][0]
n=k
plt.plot(time1,d)
plt.show()
#(make_spectro(m,baseline))
'''