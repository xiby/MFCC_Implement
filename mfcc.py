import wave
from pyaudio import PyAudio,paInt16
import numpy as np
import matplotlib.pyplot as plt
import pylab
import math

#############################
#########录音的一些参数########
############################

framerate=8000
NUM_SAMPLES=2000
channels=2
sampwidth=2
TIME=2

##########################

def readfile(filename):
    wf=wave.open(filename,'rb')
    return wf

def closefile(wf):
    wf.close()

def save_wave_file(filename,data,ch):
    wf=wave.open(filename,'wb')
    wf.setnchannels(ch)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)
    wf.writeframes(b"".join(data))
    wf.close()

def record():
    pa=PyAudio()
    stream=pa.open(format=paInt16,channels=2,
            rate=framerate,input=True,
            frames_per_buffer=NUM_SAMPLES)
    my_buf=[]
    count=0
    while count<TIME*20:
        string_audio_data=stream.read(NUM_SAMPLES)
        my_buf.append(string_audio_data)
        count=count+1
    print('.')
    save_wave_file('./datasets/01.wav',my_buf,2)
    stream.close()

def divide_frame(signal,frameLen):
    '''
    功能说明：
        对信号进行分帧，直接利用numpy模块中的reshape方法
    参数说明：
        signal  要进行分帧的声音信号
        frameLen  帧长，单位为毫秒
    返回值说明：
        返回一个分好帧的array
    '''
    len=np.size(signal)
    if len%frameLen!=0:
        padding=frameLen-len%frameLen
    else:
        padding=0
    ret=np.lib.pad(signal,(0,padding),'constant')
    ret=np.reshape(ret,(-1,frameLen))
    return ret

def add_hammingWindow(data):
    '''
    功能说明：
        为信号的每一个窗口添加海明窗，使得信号平稳
    返回值说明：
        返回每一帧加上海明窗后的帧信号'''
    row=np.size(data,0)         #信号帧数
    column=np.size(data,1)      #每一帧包含信号数
    hamminged=np.zeros((row,column))
    for i in range(row):
        for j in range(column):
            hamminged[i][j]=(0.54-0.46*math.cos(2*math.pi*j/column))*data[i][j]
    return hamminged

def preprocess(wave_data,alpha):
    '''进行预处理，主要为去除噪声，alpha为参数，一般取值为0.98'''
    len=np.size(wave_data)
    y=np.ones(len)
    y[0]=0
    for i in range(1,len):
        y[i]=wave_data[i]-alpha*wave_data[i-1]
    return y

def doFFT(data,NFFT):
    '''
    功能说明：
        对每一帧信号进行傅里叶变换，并对结果取实部的绝对值
    参数说明：
        data为所有信号帧
        NFFT为傅里叶变换序列长度，若序列长度不够长，则在其之后补0
    返回值说明：
        返回每一帧信号的傅里叶变换序列
    '''
    row,column=(np.size(data,0),np.size(data,1))
    retFFT=np.zeros((row,NFFT))
    for i in range(row):
        retFFT[i]=np.abs(np.fft.rfft(data[i],NFFT))
    return retFFT


if __name__=='__main__':
    sound=readfile('./datasets/OSR_us_000_0010_8k.wav')
    nframes=sound.getnframes()
    # print(nframes)
    framerate=sound.getframerate()
    str_data=sound.readframes(nframes)
    closefile(sound)
    wave_data=np.fromstring(str_data,dtype=np.short)
    wave_data.shape=-1,1
    wave_data=wave_data.T
    time=np.arange(0,nframes)*(1.0/framerate)
    pylab.subplot(411)
    pylab.plot(time,wave_data[0])
    new_wave_data=preprocess(wave_data[0],0.98)
    pylab.subplot(412)
    pylab.plot(time,new_wave_data,c='r')
    pylab.subplot(413)
    subWindows=divide_frame(new_wave_data,int(framerate/50))
    pylab.plot(subWindows[50],c='g')
    pylab.subplot(414)
    hamminged=add_hammingWindow(subWindows)
    pylab.plot(hamminged[50],c='g')
    pylab.show()
    pylab.subplot(411)
    pylab.plot(hamminged[50])
    pylab.subplot(412)
    transformed=np.fft.fft(hamminged[50])
    pylab.plot(transformed)
    pylab.subplot(413)
    FFT=doFFT(hamminged,256)
    pylab.plot(FFT[50])
    pylab.show()
