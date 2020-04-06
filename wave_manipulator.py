import numpy as np
import pandas as pd
import os
import librosa
from pathlib import Path
import datetime
import glob
import shutil
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift



def create_pathfile(p):
    '''Create a pandas frame with labels from directories and the path of the file, lots of hardcoding'''
    now = datetime.datetime.now()
    time = now.strftime("%H:%M:%S")
    print(time, ': Creating Pandas Frame with path and labels')
    
    pathes = []
    level_1 = []
    level_2 = []
    level_3 = []
    level_4 = []

    for filename in glob.iglob(p, recursive=True):
        if os.path.isfile(filename):
            if filename[-4:] == '.aif' or filename[-5:] == '.aiff':
                pathes.append(filename)
                folder = filename.split(os.sep)
                level_1.append(folder[5])
                level_2.append(folder[6].split()[0])
                if folder[6].split()[0] == 'Piano':
                    level_3.append('Piano')
                    level_4.append('unused')
                elif folder[6].split()[0] == 'Guitar':
                    level_3.append('Guitar')
                    level_4.append('unused')
                else:
                    level_3.append(folder[7])
                    if folder[8][-4:] == '.aif' or folder[8][-5:] == '.aiff':
                        level_4.append('unused')
                    else:
                        level_4.append(folder[8].split('.')[0])

    
    pathfile = pd.DataFrame(list(zip(pathes, level_1, level_2, level_3, level_4)), columns = ['path', 'source', 'type', 'instrument', 'style'])    
    return pathfile

def create_rawfiles(pathfile, p):
    
    now = datetime.datetime.now()
    time = now.strftime("%H:%M:%S")
    print(time, ': Creating Pandas Frames with raw data and write to Disc')
    instruments = list(set(pathfile['instrument']))
    
    for instrument in instruments:
        recording = []
        sample_rate = []
        samples = []
        length = []
        maximum_db = []
        level_1 = []
        level_2 = []
        level_3 = []
        level_4 = []

        filtered = pathfile.loc[pathfile['instrument'] == instrument]
        # print(instrument)
        for index, row in filtered.iterrows():
            x, sr = librosa.load(row['path'], sr=None)
            recording.append(x)
            sample_rate.append(sr)
            maximum_db.append(np.max(x))
            level_1.append(row['source'])
            level_2.append(row['type'])
            level_3.append(row['instrument'])
            level_4.append(row['style'])
            length.append(len(x) / sr)
            samples.append(len(x))
        output = pd.DataFrame(list(zip(recording, length, samples, sample_rate, maximum_db, level_1, level_2, level_3, level_4)), 
                              columns = ['raw_sounds', 'length', 'sample_count', 'sample_rate', 'maximum_db', 'source', 'type', 'instrument', 'style'])

        output.to_pickle(p / (instrument + '.pkl'))
        
    file_list = [p/(file + '.pkl') for file in instruments] #Create instrument list to split the amount of data into smaller frames
        
    return file_list

    
    
    
def resample(file_list, p, sr):
    
    '''Resample if required and drop to disc resampled data '''
    
    now = datetime.datetime.now()
    time = now.strftime("%H:%M:%S")
    print(time, ': Resample all recordings to target sampling rate if required')
    
    for file in file_list:

        data = pd.read_pickle(file)
        downsample = data.loc[data['sample_rate'] != sr]
        print(file, ': Resampling records:', len(downsample.index))
        resampled = []
        indexes = []
        length = []
        srs = []
        samples = []

        for index, row in downsample.iterrows():
            sample = row['raw_sounds']
            sro = row['sample_rate']
            y = librosa.resample(sample, sro, sr)
            l = len(y) / sr
            resampled.append(y)
            indexes.append(index)
            length.append(l)
            samples.append(len(y))
            srs.append(sr)
            
        output = pd.DataFrame(list(zip(resampled, srs, indexes, length, samples)),
                              columns = ['raw_sounds', 'sample_rate', 'index', 'length', 'sample_count']).set_index('index')
        data.update(output) # Join resampled recordings to raw frame
        data.to_pickle(p / file.name)
        
    file_list = [p/(file.name) for file in file_list] 
    
    return file_list
    


def trim_silence(file_list, p, Ignore, top_db):
    
    '''Cut leading and trailing silence from file'''
    now = datetime.datetime.now()
    time = now.strftime("%H:%M:%S")
    print(time, ': Trim leading and trailing silence')
    
    for file in file_list:
        data = pd.read_pickle(file)
        pad = data.loc[data['sample_count'] > Ignore]
        
        cropped = []
        indexes = []
        samples = []

        for index, row in pad.iterrows():
            sample = row['raw_sounds']
            y, area = librosa.effects.trim(sample, top_db)
            cropped.append(y)
            indexes.append(index)
            samples.append(len(y))

        output = pd.DataFrame(list(zip(cropped, indexes, samples)), columns = ['raw_sounds', 'index', 'sample_count']).set_index('index')

        data.update(output) # Join resampled recordings to raw frame
        data.to_pickle(p / file.name)
        
    file_list = [p/(file.name) for file in file_list] 
    
    return file_list



def increase_loudness(file_list, p, Ignore, factor, noise):
    '''Increase maximum Amplitude by factor for too small values'''
    now = datetime.datetime.now()
    time = now.strftime("%H:%M:%S")
    #print(time, ': Increase loudness for too silent values')
    
    for file in file_list:
        data = pd.read_pickle(file)
        pad = data.loc[data['sample_count'] > Ignore]
        
        boosted = []
        indexes = []
        maximum_db = []
        
        for index, row in pad.iterrows():
            sample = row['raw_sounds']
            maximum = np.max(sample)
            minimum = np.min(sample)
            if maximum < abs(minimum):
                maximum = abs(minimum)
            if maximum > noise:
                scale = factor / maximum
            else:
                scale = 0
                #print(file.name, index, maximum, scale)
            sample = sample * scale
            boosted.append(sample)
            indexes.append(index)
            maximum_db.append(np.max(sample))

        output = pd.DataFrame(list(zip(boosted, maximum_db, indexes)), columns = ['raw_sounds', 'maximum_db', 'index']).set_index('index')
        
        data.update(output) # Join resampled recordings to raw frame
        data.to_pickle(p / file.name)
        
    file_list = [p/(file.name) for file in file_list] 
    
    return file_list



def to_wav(file_list, p, sr=44100):
    '''Helper Function to create various wave files on demand with additional naming'''
    shutil.rmtree(p, ignore_errors=True)
    os.makedirs(p)
    for file in file_list:
        pad = pd.read_pickle(file)

        for index, row in pad.iterrows():
            play = row['raw_sounds']
            folder = file.name[:-4]
            if not os.path.exists(p / folder):
                os.makedirs(p / folder)
            if 'name' in row.index:
                librosa.output.write_wav(p/folder/(row['name'] + '.wav'), play, sr)
            else:
                librosa.output.write_wav(p/folder /(str(index) + '.wav'), play, sr)
                
def slice_recording(skipfiles, file_list, p, Ignore, min_loud, top_db, no_samples):
    
    now = datetime.datetime.now()
    time = now.strftime("%H:%M:%S")
    print(time, ': Slice longer recordings into single recordings')

    for file in file_list:

        data = pd.read_pickle(file)
        raw_sounds = []
        sample_count = []
        sample_rate = []
        source = []
        types = []
        instrument = []
        style = []
        length = []
        maximum_db = []
        name = []
        n = 0
            
        if file.name not in skipfiles:
            #print(file.name)
            for index, row in data.iterrows():
                n = 0
                sample = row['raw_sounds']
                if len(sample) > Ignore:
                    intervals =  librosa.effects.split(sample, top_db)
                    if 'name' in row.index:
                        old_name = row['name'].split('_')
                        filename = old_name[0]
                        fileno = int(old_name[1])
                    for inter in intervals:
                        if inter[1] - inter[0] >= no_samples:
                            interval = sample[inter[0]:inter[1]]
                            if np.max(interval) > min_loud:
                                raw_sounds.append(interval)
                                sample_count.append(len(interval))
                                sample_rate.append(row['sample_rate'])
                                source.append(row['source'])
                                types.append(row['type'])
                                instrument.append(row['instrument'])
                                style.append(row['style'])
                                length.append(len(interval) / row['sample_rate'])
                                maximum_db.append(np.max(interval))
                                if 'name' not in row.index:
                                    name.append(str(index) + '_' + str(n))
                                    n += 1
                                else:
                                    name.append(filename + '_' + str(fileno))
                                    fileno += 1
    
                else:
                    if np.max(sample) > min_loud:
                        raw_sounds.append(sample)
                        sample_count.append(row['sample_count'])
                        sample_rate.append(row['sample_rate'])
                        source.append(row['source'])
                        types.append(row['type'])
                        instrument.append(row['instrument'])
                        style.append(row['style'])
                        length.append(len(sample) / row['sample_rate'])
                        maximum_db.append(np.max(sample))
                        if 'name' in row.index:
                            name.append(row['name'])
                        else:
                            name.append(str(index) + '_' + str(n))


                            
        else:
            for index, row in data.iterrows():
                sample = row['raw_sounds']
                if np.max(sample) > min_loud:
                    raw_sounds.append(sample)
                    sample_count.append(row['sample_count'])
                    sample_rate.append(row['sample_rate'])
                    source.append(row['source'])
                    types.append(row['type'])
                    instrument.append(row['instrument'])
                    style.append(row['style'])
                    length.append(len(sample) / row['sample_rate'])
                    maximum_db.append(np.max(sample))
                    if 'name' in row.index:
                        name.append(row['name'])
                    else:
                        name.append(str(index) + '_' + str(n))

        output = pd.DataFrame(list(zip(name, raw_sounds, maximum_db, length, sample_count, sample_rate, source, types, instrument, style)),
                              columns = ['name', 'raw_sounds', 'maximum_db', 'length', 'sample_count', 'sample_rate', 'source', 'type', 'instrument', 'style'])  
        output.to_pickle(p / file.name)
        
    file_list = [p/(file.name) for file in file_list] 
    
    return file_list



def create_rawfiles2(pathfile, p):
    
#     now = datetime.datetime.now()
#     time = now.strftime("%H:%M:%S")
#     print(time, ': Creating Pandas Frames with raw data and write to Disc')
    instruments = list(set(pathfile['instrument']))
    
    for instrument in instruments:
        recording = []
        sample_rate = []
        samples = []
        length = []
        maximum_db = []
        level_1 = []
        level_2 = []
        level_3 = []
        level_4 = []
        name = []
        n = 0
        filtered = pathfile.loc[pathfile['instrument'] == instrument]
        # print(instrument)
        for index, row in filtered.iterrows():
            n += 1
            x, sr = librosa.load(row['path'], sr=None)
            recording.append(x)
            sample_rate.append(sr)
            maximum_db.append(np.max(x))
            level_1.append(row['source'])
            level_2.append(row['type'])
            level_3.append(row['instrument'])
            level_4.append(row['style'])
            length.append(len(x) / sr)
            samples.append(len(x))
            name.append(str(n))
        output = pd.DataFrame(list(zip(name, recording, length, samples, sample_rate,
                                       maximum_db, level_1, level_2, level_3, level_4)), 
                                      columns = ['name', 'raw_sounds', 'length', 'sample_count',
                                                 'sample_rate', 'maximum_db', 'source', 'type', 'instrument', 'style'])

        output.to_pickle(p / (instrument + '.pkl'))
        
def slice_recording2(data, Ignore, min_loud, top_db, no_samples):
    
    
    raw_sounds = []
    sample_count = []
    sample_rate = []
    source = []
    types = []
    instrument = []
    style = []
    length = []
    maximum_db = []
    name = []
    
    for index, row in data.iterrows():
        sample = row['raw_sounds']
        if len(sample) > Ignore:

            intervals =  librosa.effects.split(sample, top_db)
            for inter in intervals:
                if inter[1] - inter[0] >= no_samples:
                    interval = sample[inter[0]:inter[1]]
                    if interval.shape[0] > 0:
                        if np.max(interval) > min_loud:
                            raw_sounds.append(interval)
                            sample_count.append(len(interval))
                            sample_rate.append(row['sample_rate'])
                            source.append(row['source'])
                            types.append(row['type'])
                            instrument.append(row['instrument'])
                            style.append(row['style'])
                            length.append(len(interval) / row['sample_rate'])
                            maximum_db.append(np.max(interval))
                            name.append(row['name'])
                            
                            
        else:
            if np.max(sample) > min_loud:
                raw_sounds.append(sample)
                sample_count.append(row['sample_count'])
                sample_rate.append(row['sample_rate'])
                source.append(row['source'])
                types.append(row['type'])
                instrument.append(row['instrument'])
                style.append(row['style'])
                length.append(len(sample) / row['sample_rate'])
                maximum_db.append(np.max(sample))
                name.append(row['name'])
                        
        output = pd.DataFrame(list(zip(name, raw_sounds, maximum_db, length, sample_count, sample_rate,
                                               source, types, instrument, style)),
                              columns = ['name', 'raw_sounds', 'maximum_db', 'length', 'sample_count', 'sample_rate',
                                         'source', 'type', 'instrument', 'style'])
    
    return output

def to_wav2(files, p, sr=44100):
    '''Helper Function to create various wave files on demand with additional naming'''
    data1 = pd.read_pickle(files[0])
    names1 = set(data1['name'])
    folder = files[0].name[:-4] + '_Cut'
    shutil.rmtree(p / folder, ignore_errors=True)
    os.makedirs(p / folder)
    for name in names1:
        pad = data1.loc[data1['name'] == name]
        n = 1
        for index, row in pad.iterrows():
            play = row['raw_sounds']
            librosa.output.write_wav(p/folder/(row['name'] + '_' + str(n) + '.wav'), play, sr)
            n += 1
        
    if len(files) == 2:
        data2 = pd.read_pickle(files[1])
        names2 = set(data2['name'])
        missing = names2 - names1
        
        folder2 = files[1].name[:-4] + '_Original'
        shutil.rmtree(p / folder2, ignore_errors=True)
        os.makedirs(p / folder2)
        
        folder3 = files[1].name[:-4] + '_Missing'
        shutil.rmtree(p / folder3, ignore_errors=True)
        os.makedirs(p / folder3)
        
        for name in names2:
            pad = data2.loc[data2['name'] == name]
            n = 1
            for index, row in pad.iterrows():
                play = row['raw_sounds']
                librosa.output.write_wav(p/folder2/(row['name'] + '_' + str(n) + '.wav'), play, sr)
                n += 1
        for name in missing:
            pad = data2.loc[data2['name'] == name]
            n = 1
            for index, row in pad.iterrows():
                play = row['raw_sounds']
                librosa.output.write_wav(p/folder3/(row['name'] + '_' + str(n) + '.wav'), play, sr)
                n += 1
            
def resample2(data, sr):
    
    '''Resample if required and drop to disc resampled data '''
    
#     now = datetime.datetime.now()
#     time = now.strftime("%H:%M:%S")
#     print(time, ': Resample all recordings to target sampling rate if required')
    
    downsample = data.loc[data['sample_rate'] != sr]
    #print(file, ': Resampling records:', len(downsample.index))
    resampled = []
    indexes = []
    length = []
    srs = []
    samples = []

    for index, row in downsample.iterrows():
        sample = row['raw_sounds']
        sro = row['sample_rate']
        y = librosa.resample(sample, sro, sr)
        l = len(y) / sr
        resampled.append(y)
        indexes.append(index)
        length.append(l)
        samples.append(len(y))
        srs.append(sr)
            
    output = pd.DataFrame(list(zip(resampled, srs, indexes, length, samples)),
                          columns = ['raw_sounds', 'sample_rate', 'index', 'length', 'sample_count']).set_index('index')
    data.update(output) # Join resampled recordings to raw frame
    #    data.to_pickle(p / file.name)
        
    #file_list = [p/(file.name) for file in file_list] 
    
    return data


def trim_silence2(data, Ignore, top_db):
    
    '''Cut leading and trailing silence from file'''
#     now = datetime.datetime.now()
#     time = now.strftime("%H:%M:%S")
#     print(time, ': Trim leading and trailing silence')
    

    pad = data.loc[data['sample_count'] > Ignore]
        
    cropped = []
    indexes = []
    samples = []

    for index, row in pad.iterrows():
        sample = row['raw_sounds']
        y, area = librosa.effects.trim(sample, top_db)
        cropped.append(y)
        indexes.append(index)
        samples.append(len(y))

        output = pd.DataFrame(list(zip(cropped, indexes, samples)),
                              columns = ['raw_sounds', 'index', 'sample_count']).set_index('index')

        data.update(output) # Join resampled recordings to raw frame

        
    return data

def to_wav3(data, p, sr=44100):
    '''Helper Function to create various wave files on demand with additional naming'''
    
    direcs = set(data['directories'])
    instrument = data.iloc[0].instrument
    
    for direc in direcs:
        if not os.path.exists(p / direc):
            os.makedirs(p / direc) 

    names = set(data['name'])
    for name in names:
        pad = data.loc[data['name'] == name]
        n = 1
        for index, row in pad.iterrows():
            play = row['raw_sounds']
            librosa.output.write_wav(p/ row['directories'] / (instrument + '_' + row['name'] + '_' + str(n) + '.wav'), play, sr)
            n += 1

def delete_trash(data, length, top_db):

    data = data.drop(data[(data.maximum_db < top_db) & (data.sample_count < length)].index)

    return data

def cut_end(data, length):
    
    cropped = []
    indexes = []
    samples = []
    
    pad = data.loc[data.sample_count >= length]
    
    for index, row in pad.iterrows():
        sample = row['raw_sounds']
        y = librosa.util.fix_length(sample, length)
        
        cropped.append(y)
        indexes.append(index)
        samples.append(len(y))
        
    output = pd.DataFrame(list(zip(cropped, indexes, samples)),
                              columns = ['raw_sounds', 'index', 'sample_count']).set_index('index')

    data.update(output) # Join resampled recordings to raw frame

        
    return data

def clean_edges(data, length):
    
    cropped = []
    indexes = []
    samples = []
        
    for index, row in data.iterrows():
        sample = row['raw_sounds']
        
        z = librosa.zero_crossings(sample)
        crossings = np.nonzero(z)
        begin = crossings[0][0]
        end = crossings[0][-1]
        y = sample[begin : end + 1]
        y = librosa.util.fix_length(y, length)
        
        cropped.append(y)
        indexes.append(index)
        samples.append(len(y))
        
    output = pd.DataFrame(list(zip(cropped, indexes, samples)),
                              columns = ['raw_sounds', 'index', 'sample_count']).set_index('index')

    data.update(output) # Join resampled recordings to raw frame

        
    return data

def pad_end(data, length):
    
    pad = data.loc[data.sample_count < length]
    
    cropped = []
    indexes = []
    samples = []
        
    for index, row in pad.iterrows():
        sample = row['raw_sounds']
        l = len(sample)
        
        z = librosa.zero_crossings(sample)
        crossings = np.nonzero(z)
        begin = crossings[0][0]
        end = crossings[0][-1]
        y = sample[begin : end + 1]
        
        cropped.append(y)
        indexes.append(index)
        samples.append(len(y))
        
    output = pd.DataFrame(list(zip(cropped, indexes, samples)),
                              columns = ['raw_sounds', 'index', 'sample_count']).set_index('index')

    data.update(output) # Join resampled recordings to raw frame

        
    return data

def create_dataframe(p, structure):
    raw_sounds = []
    sra = []
    wav_path = []
    names = []
    instruments = []
    length = []
    labels = []
    
    for index, row in structure.iterrows():
        p_read = p / row.Directory
        name = row.Label
        print(p_read)
        #instrument = row.tags
        label = row.Label_int
        for file in os.scandir(p_read):
            if file.name[-4:] == '.wav':
                x, sr = librosa.load(file, sr=None)
                raw_sounds.append(x)
                sra.append(sr)
                wav_path.append(p_read / file.name)
                #instruments.append(instrument)
                names.append(name)
                length.append(len(x))
                labels.append(label)
                
    output = pd.DataFrame(list(zip(wav_path, raw_sounds, sra, names, length, labels)),
                              columns = ['wav_path', 'raw_sounds', 'sample_rate', 'names', 'no_samples', 'labels'])
                
                
    return output
            
def analyze(data):

    tags = set(data.names)

    count = []
    labels = []
    instruments = []
    for tag in tags:
        pad = data.loc[data.names == tag]
        count.append(len(pad))
        labels.append(pad.iloc[0]['labels'])
        instruments.append(tag)

    analyze = pd.DataFrame(list(zip(labels, instruments, count)), columns = ['Label', 'Instrument', 'Nos']) 
    Label_dict = dict(zip(instruments, labels))
    
    return analyze, Label_dict   

def sound_augmenter(data, augmentation, size=1):
    
    sr = augmentation['sr']
    augmenter = augmentation['augmenter']
    loudad = augmentation['loudad']
    min_loud = augmentation['min_loud']
    max_loud = augmentation['max_loud']
    new_sounds = []
    maximum = []
    
    if isinstance(data, pd.DataFrame):
        pad = data.sample(frac=size)

        for index, row in pad.iterrows():
            sample = row['raw_sounds']
            new_sound = augmenter(samples=sample, sample_rate=sr)
            if loudad:
                new_sound = new_sound / np.max(sample) * np.random.uniform(min_loud, max_loud)

            new_sounds.append(new_sound)
        pad['raw_sounds'] = new_sounds
        augmented = pad
        
    if isinstance(data, np.ndarray):
        new_sound = augmenter(samples=data, sample_rate=sr)
        if loudad:
            new_sound = new_sound / np.max(data) * np.random.uniform(min_loud, max_loud)


        augmented = new_sound
        
    return new_sound

def slicer(audio, sr, hoplength):
    l_predict = sr * 3 -1
    hop = int(sr * hoplength)
    frames = int((len(audio) - l_predict) / hop)
    inter = []
    srs = []
    for frame in range(frames):
        slices = audio[frame * hop : (frame * hop) + l_predict]
        inter.append(slices)
        srs.append(sr)
    out = pd.DataFrame(list(zip(inter, srs)), columns = ['raw_sounds', 'sample_rate'])  
    return out