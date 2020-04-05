import numpy as np
import pandas as pd
import os
import librosa
import soundfile as sf
import json
import math
import pickle
from pathlib import Path

def create_features(data, param_file, suffix=''):
    
    featurestore = Path.cwd() / 'features'
    if not os.path.exists(featurestore):
        os.makedirs(featurestore)
    

    feature_file = param_file[:-4] + '_features' + suffix + '.npy'
    label_file = param_file[:-4] + '_label' + suffix + '.npy'
    
    
    if Path(featurestore / feature_file).is_file():
        features = np.load(featurestore / feature_file)
        labels = np.load(featurestore / label_file)
        
        print(param_file)
        print('Features already calculated, read from disc, ignore parameter file')
        
        return features, labels #Function is dropping out here, if features are already stored on disc
        
    if Path(featurestore / param_file).is_file():

        with open(featurestore / param_file, 'r') as file:
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
    if 'content' in parameter:
        content = parameter['content']
    else:
        content = 'all'
    
    
    
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
        
        
    if Transformation == 'MEL_dB':
        print(desc)
        print('Hop_length: ', hl)
        print('Sampling Rate:', srs)
        print('Fast Fourier Window:', fft)
        print('Number of MEL Bins:', n_mel)
        print('Shape of Feature: ', shape)
        print('Minimum Frequency: ', fmin )
        print('Maximum Frequency: ', fmax )
        print('Content: ', content)
        
        S = MEL_linear(data, srs, hl, fft, n_mel, fmin, fmax, content=content)
        features  = librosa.power_to_db(S, ref=np.max)/(-80) #Scale since minimum value is-80, maximum 0
        features = np.expand_dims(features, axis=3) #This adds a channel dimension of 1    

        
        
    if Transformation == 'MEL_dB_complex':
        print(desc)
        print('Hop_length: ', hl)
        print('Sampling Rate:', srs)
        print('Fast Fourier Window:', fft)
        print('Number of MEL Bins:', n_mel)
        print('Shape of Feature: ', shape)
        print('Minimum Frequency: ', fmin )
        print('Maximum Frequency: ', fmax )
        
        S = MEL_linear(data, srs, hl, fft, n_mel, fmin, fmax, compl=True)
        features_x  = librosa.power_to_db(S[0], ref=np.max)/(-80)
        features_y  = librosa.power_to_db(S[1], ref=np.max)/(-80)
        features = np.stack((features_x, features_y), axis=3)
        #features = np.expand_dims(features, axis=3) #This adds a channel dimension of 1    
        
    if Transformation == 'MEL_dB_decompose':
        margin = parameter['margin']
        print(desc)
        print('Hop_length: ', hl)
        print('Sampling Rate:', srs)
        print('Fast Fourier Window:', fft)
        print('Number of MEL Bins:', n_mel)
        print('Shape of Feature: ', shape)
        print('Minimum Frequency: ', fmin )
        print('Maximum Frequency: ', fmax )
        print('Content: ', content)
        print('Margin: ', margin)
        
        if content == 'decomposed':
            S = MEL_decompose(data, srs, hl, fft, n_mel, fmin, fmax, margin, content)
            features_x  = librosa.power_to_db(S[0], ref=np.max)/(-80)
            features_y  = librosa.power_to_db(S[1], ref=np.max)/(-80)
            features = np.stack((features_x, features_y), axis=3)
            
        if content == 'all':
            S = MEL_decompose(data, srs, hl, fft, n_mel, fmin, fmax, margin, content)
            features_x  = librosa.power_to_db(S[0], ref=np.max)/(-80)
            features_y  = librosa.power_to_db(S[1], ref=np.max)/(-80)
            features_z  = librosa.power_to_db(S[2], ref=np.max)/(-80)
            features = np.stack((features_x, features_y, features_z), axis=3)
        #features = np.expand_dims(features, axis=3) #This adds a channel dimension of 1            
        

        
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
        
    if Transformation == 'chroma':
        n_chroma = parameter['no_chroma']
        tuning = parameter['chroma_tune']
        flatten = parameter['flatten']
        print(desc)
        print('Hop_length: ', hl)
        print('Sampling Rate:', srs)
        print('Fast Fourier Window:', fft)
        print('Number of Chroma Bins:', n_chroma)
        print('A440 Tuning: ', tuning)
        print('Shape of Feature: ', shape)
        print('Minimum Frequency: ', fmin )
        print('Maximum Frequency: ', fmax )
        print('Flatten: ', flatten )
        
        features = chromagram(data, srs, hl, fft, n_chroma, tuning, flatten)
        #features = Scale_0_1(features)
        if flatten:
            features = np.expand_dims(features, axis=2) #This adds a channel dimension of 1
        if not flatten:
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
        if norm == 'linear':
            features = Scale_0_1(features)
        elif norm == 'dB':
            features  = librosa.power_to_db(features, ref=np.max) / (-80) #Scale since minimum value is-80, maximum 0
        features = np.expand_dims(features, axis=3) #This adds a channel dimension of 1

        
        
    if Transformation == 'Time':
        print(desc)
        print('Hop_length: ', hl)
        print('Sampling Rate:', srs)
        print('Shape of Feature: ', shape)
       
        features = time_only(data, srs)
        features = np.expand_dims(features, axis=2) #This adds a channel dimension of 1
        
             
    labels = np.array(data['labels'])
    
    np.save(featurestore / feature_file, features)
    np.save(featurestore / label_file, labels)
    
    
    return features, labels



def MEL_linear(data, srs, hl, fft, n_mel, fmin, fmax, compl=False, content='all'):
    '''Calculate the MEL Spectogramm with given Parameters, returns numpy array'''
    sample_vector = np.array(data['raw_sounds'])
    result = []
    result_x = []
    result_y = []
    
    for sample in sample_vector:
        ft = librosa.stft(sample, hop_length=hl, n_fft = fft, window='hann') #Calculate Fast Fourier Transform
        
        if content == 'harmonic':
            ft, D_percussive = librosa.decompose.hpss(ft)
            
        if content == 'percuss':
            D_harmonic, ft = librosa.decompose.hpss(ft)
            
                
        if not compl:
            D = np.abs(ft)**2 #Calculaing the Power
            mels = librosa.feature.melspectrogram(S=D, sr=srs, n_mels=n_mel, fmin=fmin, fmax=fmax) # Calculate MEL Spectogramm
            result.append(mels)
        if compl:
            x = np.abs(ft.real)
            y = np.abs(ft.imag)
            mels_x = librosa.feature.melspectrogram(S=x, sr=srs, n_mels=n_mel, fmin=fmin, fmax=fmax)
            mels_y = librosa.feature.melspectrogram(S=y, sr=srs, n_mels=n_mel, fmin=fmin, fmax=fmax)
            result_x.append(mels_x)
            result_y.append(mels_y)
            result = [result_x, result_y]
    return np.array(result)


def MEL_decompose(data, srs, hl, fft, n_mel, fmin, fmax, margin, content='all'):
    '''Calculate the MEL Spectogramm with given Parameters, returns numpy array'''
    sample_vector = np.array(data['raw_sounds'])
    result_W = []
    result_x = []
    result_y = []
    result_z = []
    
    for sample in sample_vector:
        ft = librosa.stft(sample, hop_length=hl, n_fft = fft, window='hann') #Calculate Fast Fourier Transform
        
        if content == 'decomposed':
            D_harmonic, D_percussive = librosa.decompose.hpss(ft, margin)
            D_h = np.abs(D_harmonic)**2
            D_p = np.abs(D_percussive)**2
            mels_h = librosa.feature.melspectrogram(S=D_h, sr=srs, n_mels=n_mel, fmin=fmin, fmax=fmax)
            mels_p = librosa.feature.melspectrogram(S=D_p, sr=srs, n_mels=n_mel, fmin=fmin, fmax=fmax)
            result_x.append(mels_h)
            result_y.append(mels_p)
            result = [result_x, result_y]
            
        if content == 'all':
            D_harmonic, D_percussive = librosa.decompose.hpss(ft, margin)
            D_h = np.abs(D_harmonic)**2
            D_p = np.abs(D_percussive)**2
            D = np.abs(ft)**2
            mels_h = librosa.feature.melspectrogram(S=D_h, sr=srs, n_mels=n_mel, fmin=fmin, fmax=fmax)
            mels_p = librosa.feature.melspectrogram(S=D_p, sr=srs, n_mels=n_mel, fmin=fmin, fmax=fmax)
            mels = librosa.feature.melspectrogram(S=D, sr=srs, n_mels=n_mel, fmin=fmin, fmax=fmax)
            result_x.append(mels_h)
            result_y.append(mels_p)
            result_z.append(mels)
            result = [result_x, result_y, result_z]
            
            
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

def chromagram(data, srs, hl, fft, nchroma, tune, flatten):
    '''Calculate complex fft Transformation in 2 channels'''
    sample_vector = np.array(data['raw_sounds'])
    result = []
    
    for sample in sample_vector:
        ft = librosa.stft(sample, hop_length=hl, n_fft = fft, window='hann') #Calculate Fast Fourier Transform
        D = np.abs(ft)
        chromagram = librosa.feature.chroma_stft(S=D, sr=srs, n_chroma=nchroma, tuning=tune)
        if flatten:
            chromagram = chromagram.flatten()
        result.append(chromagram)
        
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


def create_feature(data, param_file):
    featurestore = Path.cwd() / 'features'
    if not os.path.exists(featurestore):
        os.makedirs(featurestore)
        
    if Path(featurestore / param_file).is_file():

        with open(featurestore / param_file, 'r') as file:
            parameter = json.load(file)

    else:
        print('No such Parameter set:', param_file )
        
        return 0,0
    
    raw_sounds = []
    sr = []
    raw_sounds.append(data)
    raw_sounds.append(data)
    sr.append(44100)
    sr.append(44100)
    
    data = pd.DataFrame(list(zip(raw_sounds, sr)), columns = ['raw_sounds', 'sample_rate'])    
    
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
    if 'content' in parameter:
        content = parameter['content']
    else:
        content = 'all'
    
    
    
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
        #features = np.expand_dims(features, axis=3) #This adds a channel dimension of 1
        
        
    if Transformation == 'MEL_dB':
        print(desc)
        print('Hop_length: ', hl)
        print('Sampling Rate:', srs)
        print('Fast Fourier Window:', fft)
        print('Number of MEL Bins:', n_mel)
        print('Shape of Feature: ', shape)
        print('Minimum Frequency: ', fmin )
        print('Maximum Frequency: ', fmax )
        print('Content: ', content)
        
        S = MEL_linear(data, srs, hl, fft, n_mel, fmin, fmax, content=content)
        features  = librosa.power_to_db(S, ref=np.max)/(-80) #Scale since minimum value is-80, maximum 0
        #features = np.expand_dims(features, axis=3) #This adds a channel dimension of 1    

        
        
    if Transformation == 'MEL_dB_complex':
        print(desc)
        print('Hop_length: ', hl)
        print('Sampling Rate:', srs)
        print('Fast Fourier Window:', fft)
        print('Number of MEL Bins:', n_mel)
        print('Shape of Feature: ', shape)
        print('Minimum Frequency: ', fmin )
        print('Maximum Frequency: ', fmax )
        
        S = MEL_linear(data, srs, hl, fft, n_mel, fmin, fmax, compl=True)
        features_x  = librosa.power_to_db(S[0], ref=np.max)/(-80)
        features_y  = librosa.power_to_db(S[1], ref=np.max)/(-80)
        features = np.stack((features_x, features_y), axis=3)
        #features = np.expand_dims(features, axis=3) #This adds a channel dimension of 1    
        
    if Transformation == 'MEL_dB_decompose':
        margin = parameter['margin']
        print(desc)
        print('Hop_length: ', hl)
        print('Sampling Rate:', srs)
        print('Fast Fourier Window:', fft)
        print('Number of MEL Bins:', n_mel)
        print('Shape of Feature: ', shape)
        print('Minimum Frequency: ', fmin )
        print('Maximum Frequency: ', fmax )
        print('Content: ', content)
        print('Margin: ', margin)
        
        if content == 'decomposed':
            S = MEL_decompose(data, srs, hl, fft, n_mel, fmin, fmax, margin, content)
            features_x  = librosa.power_to_db(S[0], ref=np.max)/(-80)
            features_y  = librosa.power_to_db(S[1], ref=np.max)/(-80)
            features = np.stack((features_x, features_y), axis=3)
            
        if content == 'all':
            S = MEL_decompose(data, srs, hl, fft, n_mel, fmin, fmax, margin, content)
            features_x  = librosa.power_to_db(S[0], ref=np.max)/(-80)
            features_y  = librosa.power_to_db(S[1], ref=np.max)/(-80)
            features_z  = librosa.power_to_db(S[2], ref=np.max)/(-80)
            features = np.stack((features_x, features_y, features_z), axis=3)
        #features = np.expand_dims(features, axis=3) #This adds a channel dimension of 1            
        

        
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
        #features = np.expand_dims(features, axis=3) #This adds a channel dimension of 1
        
    if Transformation == 'chroma':
        n_chroma = parameter['no_chroma']
        tuning = parameter['chroma_tune']
        flatten = parameter['flatten']
        print(desc)
        print('Hop_length: ', hl)
        print('Sampling Rate:', srs)
        print('Fast Fourier Window:', fft)
        print('Number of Chroma Bins:', n_chroma)
        print('A440 Tuning: ', tuning)
        print('Shape of Feature: ', shape)
        print('Minimum Frequency: ', fmin )
        print('Maximum Frequency: ', fmax )
        print('Flatten: ', flatten )
        
        features = chromagram(data, srs, hl, fft, n_chroma, tuning, flatten)
        #features = Scale_0_1(features)
#         if flatten:
#             features = np.expand_dims(features, axis=2) #This adds a channel dimension of 1
#         if not flatten:
#             features = np.expand_dims(features, axis=3) #This adds a channel dimension of 1


    
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
        if norm == 'linear':
            features = Scale_0_1(features)
        elif norm == 'dB':
            features  = librosa.power_to_db(features, ref=np.max) / (-80) #Scale since minimum value is-80, maximum 0
#         features = np.expand_dims(features, axis=3) #This adds a channel dimension of 1

        
        
    if Transformation == 'Time':
        print(desc)
        print('Hop_length: ', hl)
        print('Sampling Rate:', srs)
        print('Shape of Feature: ', shape)
       
        features = time_only(data, srs)
#         features = np.expand_dims(features, axis=2) #This adds a channel dimension of 1
        
             
    
    return features[0]

