import pandas as pd
import numpy as np
import os, pathlib
import librosa

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
    df_raw = df.copy()
    df_raw['raw_sounds'] = raw_samples
    df_raw['sample_rate'] = sample_rates
    # Generate instrument dict
    class_list = set(df.tags)
    class_list = list(c_list)
    class_dict =  { i: c_list[i] for i in range(0, len(c_list))}
    # 
    return (df, df_raw, class_dict)

def filter_instruments(df, instrument_list={'cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi'}):
    # Takes an instrument list and returns a filtered dataframe 
    df = df[df['tags'].isin(instrument_list)].copy()
    return df