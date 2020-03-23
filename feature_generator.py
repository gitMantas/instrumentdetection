import numpy as np
import pandas as pd
import os
import librosa
import soundfile as sf
import json
import math
import pickle
from pathlib import Path

def create_features(data, param_file):
    
    modelstore = Path.cwd() / 'models'
    if Path(modelstore / param_file).is_file():

        with open(modelstore / param_file, 'r') as file:
            parameter = json.load(file)

    else:
        print('No such Parameter set:', param_file )
        
        return 0,0
    
    
    desc = parameter['Description']
    Transformation = parameter['Transformation']
    srs = parameter['sampling_rate']
    hl = parameter['hop_length']
    fft = parameter['fft_window']
    n_mel = parameter['no_mel_bin']
    norm = parameter['loudness_normalization']
    shape = parameter['Input_Dim']
    fmin = parameter['fmin']
    fmax = parameter['fmax']
    
    
    if Transformation == 'MEL_linear':
        print(desc)
        print('Hop_length: ', hl)
        print('Sampling Rate:', srs)
        print('Fast Fourier Window:', fft)
        print('Number of MEL Bins:', n_mel)
        print('Shape of Feature: ', shape)
        print('Minimum Frequency: ', fmin )
        print('Maximum Frequency: ', fmax )
        
        features = MEL_linear(data, srs, hl, fft, n_mel, fmin, fmax)
        features = Scale_0_1(features)
        features = np.expand_dims(features, axis=3) #This adds a channel dimension of 1
        
    if Transformation == 'MFCC':
        n_mfcc = parameter['no_mfcc']
        dcttype = parameter['dct_type']
        print(desc)
        print('Hop_length: ', hl)
        print('Sampling Rate:', srs)
        print('Fast Fourier Window:', fft)
        print('Number of MEL Bins:', n_mel)
        print('Number of Cepstral Coefficients: ', n_mfcc)
        print('Typer of discrete cosinus transform: ', dcttype)
        print('Shape of Feature: ', shape)
        print('Minimum Frequency: ', fmin )
        print('Maximum Frequency: ', fmax )
        
        features = MEL_Cepstrum_Coeff(data, srs, hl, fft, n_mel, fmin, fmax, n_mfcc, dcttype)
        features = Scale_0_1(features)
        features = np.expand_dims(features, axis=3) #This adds a channel dimension of 1

    
    if Transformation == 'FFT_Complex':
        print(desc)
        print('Hop_length: ', hl)
        print('Sampling Rate:', srs)
        print('Fast Fourier Window:', fft)
        print('Shape of Feature: ', shape)
       
        features = fft_complex(data, srs, hl, fft)
        features = complex_2_channels(features)
        features = Scale_1_1(features)

        
    if Transformation == 'FFT_Absolut':
        print(desc)
        print('Hop_length: ', hl)
        print('Sampling Rate:', srs)
        print('Fast Fourier Window:', fft)
        print('Shape of Feature: ', shape)
       
        features = fft_complex(data, srs, hl, fft)
        features = np.abs(features)**2 #Calculaing the Power
        features = Scale_1_1(features)
        features = np.expand_dims(features, axis=3) #This adds a channel dimension of 1

        
        
    if Transformation == 'Time':
        print(desc)
        print('Hop_length: ', hl)
        print('Sampling Rate:', srs)
        print('Shape of Feature: ', shape)
       
        features = time_only(data, srs)
        features = np.expand_dims(features, axis=2) #This adds a channel dimension of 1
        
             
    labels = np.array(data['labels'])    
    return features, labels



def MEL_linear(data, srs, hl, fft, n_mel, fmin, fmax):
    '''Calculate the MEL Spectogramm with given Parameters, returns numpy array'''
    sample_vector = np.array(data['raw_sounds'])
    result = []
    
    for sample in sample_vector:
        ft = librosa.stft(sample, hop_length=hl, n_fft = fft, window='hann') #Calculate Fast Fourier Transform
        D = np.abs(ft)**2 #Calculaing the Power
        mels = librosa.feature.melspectrogram(S=D, sr=srs, n_mels=n_mel, fmin=fmin, fmax=fmax) # Calculate MEL Spectogramm
        result.append(mels)
        
    return np.array(result)


def MEL_Cepstrum_Coeff(data, srs, hl, fft, n_mel, fmin, fmax, nmfcc, dcttype):
    '''Calculate the MEL Spectogramm with given Parameters, returns numpy array'''
    sample_vector = np.array(data['raw_sounds'])
    result = []
    
    for sample in sample_vector:
        ft = librosa.stft(sample, hop_length=hl, n_fft = fft, window='hann') #Calculate Fast Fourier Transform
        D = np.abs(ft)**2 #Calculaing the Power
        mels = librosa.feature.melspectrogram(S=D, sr=srs, n_mels=n_mel, fmin=fmin, fmax=fmax) # Calculate MEL Spectogramm
        mels_log = librosa.power_to_db(mels) #Translate Power onto log scale
        mfccs = librosa.feature.mfcc(S=mels_log, sr=srs, n_mfcc=nmfcc, dct_type=dcttype) #Do PCA / COS Transform
        result.append(mfccs)
        
    return np.array(result)



def fft_complex(data, srs, hl, fft):
    '''Calculate complex fft Transformation in 2 channels'''
    sample_vector = np.array(data['raw_sounds'])
    result = []
    
    for sample in sample_vector:
        ft = librosa.stft(sample, hop_length=hl, n_fft = fft, window='hann') #Calculate Fast Fourier Transform
        result.append(ft)
        
    return np.array(result)    
    

def complex_2_channels(data):
    '''Split a complex feature matrix in real and imaginary part and return as 2 channels'''
    x = data.real
    y = data.imag
    z = np.stack((x, y), axis=3)
    
    return z


def time_only(data, srs_neu):
    sample_vector = np.array(data['raw_sounds'])
    rate_vector = np.array(data['sample_rate'])

    result = []
    
    for sample , sr in zip(sample_vector, rate_vector):
        y = librosa.resample(sample, sr, srs_neu)
        result.append(y)
        
    return np.array(result) 



def Scale_0_1(features):
    ''' Too simple MinMax Scaler'''
    maximum = np.max(features)
    minimum = np.min(features)
    features = (features - minimum) / (maximum - minimum)
   
    return features    


def Scale_1_1(features):
    ''' Too simple MinMax Scaler'''
    maximum = np.max(features)
    minimum = np.min(features)
    if maximum >= abs(minimum):
        scale = maximum
    else:
        scale = abs(minimum)
    features = features / scale
    
    return features    