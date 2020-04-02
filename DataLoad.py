import pandas as pd
import numpy as np
import os, pathlib
import librosa
from sklearn.model_selection import train_test_split

def load_to_dataframe():
    IRMAS_TRAINING = 'data/IRMAS-TrainingData'
    base_path = pathlib.Path(IRMAS_TRAINING)
    classes, paths = [], []
    for p in base_path.glob('*/*'):
        relative_path = p.relative_to(base_path)
        classes.append(str(relative_path.parent))
        paths.append(p)
    df = pd.DataFrame({"tags": classes, "wav_path": paths}).sample(frac=1)#.reset_index(drop=True)#, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Test if files are loading
    successful, corrupted = [], []
    len_x, sample_rates = [], []
    raw_samples, sample_rates = [], []
    for p in df.wav_path:
        try:
            x, sr = librosa.load(p, sr=None)
            successful.append(p)
            len_x.append(len(x))
            raw_samples.append(x)
            sample_rates.append(sr)
        except:
            corrupted.append(p)
            
    assert len(successful) == len(raw_samples)
    
    # Generate instrument dict
    class_list = set(df.tags)
    class_list = sorted(list(class_list))
    class_dict =  { i: class_list[i] for i in range(0, len(class_list))}
    inverted_dict = dict(map(reversed, class_dict.items()))
    # Add labels column
    labels = []
    for row in df.itertuples():
        label_number = inverted_dict[row[1]]
        labels.append(label_number)
    df['labels'] = labels
    
    # Create a copy of a raw table
    df_raw = df.copy()
    df_raw['raw_sounds'] = raw_samples
    df_raw['sample_rate'] = sample_rates
    # 
    return (df, df_raw, class_dict)

def filter_instruments(df, instrument_list={'cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi'}):
    # Takes an instrument list and returns a filtered dataframe 
    df = df[df['tags'].isin(instrument_list)].copy()
    return df

def load_to_dataframe_train_test_split():
    # Return a dataframe split in train and test
    IRMAS_TRAINING = 'data/IRMAS-TrainingData'
    base_path = pathlib.Path(IRMAS_TRAINING)
    classes, paths = [], []
    for p in base_path.glob('*/*'):
        relative_path = p.relative_to(base_path)
        classes.append(str(relative_path.parent))
        paths.append(p)
    df = pd.DataFrame({"tags": classes, "wav_path": paths}).sample(frac=1)#.reset_index(drop=True)#, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Test if files are loading
    successful, corrupted = [], []
    len_x, sample_rates = [], []
    raw_samples, sample_rates = [], []
    for p in df.wav_path:
        try:
            x, sr = librosa.load(p, sr=None)
            successful.append(p)
            len_x.append(len(x))
            raw_samples.append(x)
            sample_rates.append(sr)
        except:
            corrupted.append(p)
            
    assert len(successful) == len(raw_samples)
    
    # Generate instrument dict
    class_list = set(df.tags)
    class_list = sorted(list(class_list))
    class_dict =  { i: class_list[i] for i in range(0, len(class_list))}
    inverted_dict = dict(map(reversed, class_dict.items()))
    # Add labels column
    labels = []
    for row in df.itertuples():
        label_number = inverted_dict[row[1]]
        labels.append(label_number)
      
    df['labels'] = labels    
    df['raw_sounds'] = raw_samples
    df['sample_rate'] = sample_rates
    #
    df_train, df_test = train_test_split(df, test_size=0.3)
    #
    return (df_train, df_test, class_dict, inverted_dict)