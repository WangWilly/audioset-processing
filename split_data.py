import os
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
from shutil import copyfile

np.random.seed(87)


def split_data(dir, name, type):
    files = librosa.util.find_files(dir)
    output_dir = 'audio_splited/'
    os.makedirs(output_dir, exist_ok=True)
    np.random.shuffle(files)
    
    os.makedirs('./{}'.format(type), exist_ok=True)
    
    train = []
    validate = []
    test = []

    train_num = int(len(files)*0.7)
    validate_num = int(len(files)*0.15)

    for idx, f in enumerate(files):
        if idx <= train_num:
            train.append(f)
        elif train_num < idx <= train_num + validate_num:
            validate.append(f)
        else:
            test.append(f)
    
    data = {
        'train': train, 
        'validate': validate, 
        'test': test
    }
    np.save(os.path.join(type, 'filepath_{}.npy'.format(name)), data)
    
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    for f in tqdm(train):
        copyfile(f, os.path.join(output_dir, 'train', '{}'.format(Path(f).name)))

    os.makedirs(os.path.join(output_dir, 'validate'), exist_ok=True)
    for f in tqdm(validate):
        copyfile(f, os.path.join(output_dir, 'validate', '{}'.format(Path(f).name)))

    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    for f in tqdm(test):
        copyfile(f, os.path.join(output_dir, 'test', '{}'.format(Path(f).name)))
    

if __name__ == '__main__':
    split_data('output/', 'audioset', 'noise')