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
        labels = np.array(data['labels'])
    
    if Transformation == 'FFT_Complex_1':
        print(desc)
        print('Hop_length: ', hl)
        print('Sampling Rate:', srs)
        print('Fast Fourier Window:', fft)
        print('Shape of Feature: ', shape)
       
        features = fft_complex(data, srs, hl, fft)
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



def fft_complex(data, srs, hl, fft):
    '''Calculate complex fft Transformation in 2 channels'''
    sample_vector = np.array(data['raw_sounds'])
    result = []
    
    for sample in sample_vector:
        ft = librosa.stft(sample, hop_length=hl, n_fft = fft, window='hann') #Calculate Fast Fourier Transform
        result.append(ft)
        
    return np.array(result)    
    


def Scale_0_1(features):
    ''' Too simple MinMax Scaler'''
    maximum = np.max(features)
    minimum = np.min(features)
    features = (features - minimum) / (maximum - minimum)
    
    return features    