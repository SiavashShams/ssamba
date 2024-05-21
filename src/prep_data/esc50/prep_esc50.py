import numpy as np
import json
import os
import zipfile
import wget
import torchaudio

#

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

# downlooad esc50
# dataset provided in https://github.com/karolpiczak/ESC-50
file_count = 0
if os.path.exists('/engram/naplab/shared/ESC-50-master') == False:
    esc50_url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
    wget.download(esc50_url, out='/engram/naplab/shared/')
    with zipfile.ZipFile('/engram/naplab/shared/ESC-50-master.zip', 'r') as zip_ref:
        zip_ref.extractall('/engram/naplab/shared/')
    os.remove('/engram/naplab/shared/ESC-50-master.zip')

    # convert the audio to 16kHz
    base_dir = '/engram/naplab/shared/ESC-50-master/'
    os.mkdir('/engram/naplab/shared/ESC-50-master/audio_16k/')
    audio_list = get_immediate_files('/engram/naplab/shared/ESC-50-master/audio')
    output_dir = '/engram/naplab/shared/ESC-50-master/audio_16k'  
    for audio in audio_list:
        #print('sox ' + base_dir + '/audio/' + audio + ' -r 16000 ' + base_dir + '/audio_16k/' + audio)
        #os.system('sox ' + base_dir + '/audio/' + audio + ' -r 16000 ' + base_dir + '/audio_16k/' + audio)
        input_path = os.path.join(base_dir, 'audio', audio)
        output_path = os.path.join(output_dir, audio)
        waveform, sample_rate = torchaudio.load(input_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
    
        torchaudio.save(output_path, waveform, 16000)
        file_count += 1
        if file_count % 2000 == 0:
            print(f"Processed {file_count} files.")
            
if os.path.exists('/engram/naplab/shared/ESC-50-master/audio_16k') == False:
    base_dir = '/engram/naplab/shared/ESC-50-master/'
    os.mkdir('/engram/naplab/shared/ESC-50-master/audio_16k/')
    audio_list = get_immediate_files('/engram/naplab/shared/ESC-50-master/audio')
    output_dir = '/engram/naplab/shared/ESC-50-master/audio_16k'  
    for audio in audio_list:
        #print('sox ' + base_dir + '/audio/' + audio + ' -r 16000 ' + base_dir + '/audio_16k/' + audio)
        #os.system('sox ' + base_dir + '/audio/' + audio + ' -r 16000 ' + base_dir + '/audio_16k/' + audio)
        input_path = os.path.join(base_dir, 'audio', audio)
        output_path = os.path.join(output_dir, audio)
        waveform, sample_rate = torchaudio.load(input_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
    
        torchaudio.save(output_path, waveform, 16000)
        file_count += 1
        if file_count % 2000 == 0:
            print(f"Processed {file_count} files.")

output_dir = '/engram/naplab/shared/ESC-50-master/audio_16k'  

label_set = np.loadtxt('./esc_class_labels_indices.csv', delimiter=',', dtype='str')
label_map = {}
for i in range(1, len(label_set)):
    label_map[eval(label_set[i][2])] = label_set[i][0]
print(label_map)

# fix bug: generate an empty directory to save json files
if os.path.exists('/engram/naplab/shared/datafiles') == False:
    os.mkdir('/engram/naplab/shared/datafiles')

for fold in [1,2,3,4,5]:
    base_path = os.path.abspath(os.getcwd()) + "/data/ESC-50-master/audio_16k/"
    meta = np.loadtxt('/engram/naplab/shared/ESC-50-master/meta/esc50.csv', delimiter=',', dtype='str', skiprows=1)
    train_wav_list = []
    eval_wav_list = []
    for entry in meta:
        cur_label = label_map[entry[3]]
        cur_path = entry[0]
        cur_fold = int(entry[1])
        cur_dict = {"wav": os.path.join(output_dir, cur_path), "labels": '/m/07rwj' + cur_label.zfill(2)}
        if cur_fold == fold:
            eval_wav_list.append(cur_dict)
        else:
            train_wav_list.append(cur_dict)
    print(f'fold {fold}: {len(train_wav_list)} training samples, {len(eval_wav_list)} test samples')

    train_file = f'/engram/naplab/shared/datafiles/esc_train_data_{fold}.json'
    eval_file = f'/engram/naplab/shared/datafiles/esc_eval_data_{fold}.json'
    with open(train_file, 'w') as f:
        json.dump({'data': train_wav_list}, f, indent=1)
    with open(eval_file, 'w') as f:
        json.dump({'data': eval_wav_list}, f, indent=1)

print('Finished ESC-50 Preparation')