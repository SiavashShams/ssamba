import os
import pandas as pd
import torchaudio
from pydub import AudioSegment
import numpy as np
import logging

# Configure logging
logging.basicConfig(filename='audio_loading.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

# Path to the metadata file
metadata_file = '/engram/naplab/shared/UrbanSound8K/metadata/UrbanSound8K.csv'

audio_base_path = '/engram/naplab/shared/UrbanSound8K/audio'

desired_length = 60000

# Load the metadata
metadata = pd.read_csv(metadata_file)
segment_duration = 1000  # 1 second in milliseconds

def concatenate_audios(fold, audio_base_path, desired_length, segment_duration):
    fold_path = os.path.join(audio_base_path, f'fold{fold}')
    fold_metadata = metadata[metadata['fold'] == fold]

    # Shuffle the audio files to ensure varied classes
    fold_metadata = fold_metadata.sample(frac=1).reset_index(drop=True)
    
    concatenated_segments = []
    concatenated_labels = []
    current_segment = AudioSegment.empty()
    current_length = 0
    current_labels = []
    padded_count = 0
    
    for index, row in fold_metadata.iterrows():
        audio_file = row['slice_file_name']
        label = row['classID']  
        audio_path = os.path.join(fold_path, audio_file)
        
        if not os.path.exists(audio_path):
            logging.error(f"File {audio_path} does not exist.")
            continue

        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert stereo to mono by averaging the channels if there are 2 channels
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16000 Hz if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Convert waveform to bytes for AudioSegment
        waveform_bytes = (waveform.numpy() * 32767).astype(np.int16).tobytes()
        audio_segment = AudioSegment(
            data=waveform_bytes,
            sample_width=2,  # 2 bytes for int16
            frame_rate=16000,
            channels=1
        )
        
        # Round up audio segment to the next full second if it's shorter
        if len(audio_segment) % segment_duration != 0:
            pad_length = segment_duration - (len(audio_segment) % segment_duration)
            audio_segment += AudioSegment.silent(duration=pad_length)
            padded_count += 1

        current_segment += audio_segment
        current_length += len(audio_segment)
        current_labels.extend([label] * ((len(audio_segment) + segment_duration - 1) // segment_duration))

        while current_length >= desired_length:
            concatenated_segments.append(current_segment[:desired_length])
            concatenated_labels.append(current_labels[:desired_length // segment_duration])
            current_segment = current_segment[desired_length:]
            current_length = len(current_segment)
            current_labels = current_labels[desired_length // segment_duration:]

    # If there's any remaining segment, pad it to the desired length and add it
    if current_length > 0:
        pad_length = desired_length - current_length
        current_segment += AudioSegment.silent(duration=pad_length)
        current_labels.extend([current_labels[-1]] * (pad_length // segment_duration))
        concatenated_segments.append(current_segment[:desired_length])
        concatenated_labels.append(current_labels[:desired_length // segment_duration])
    
    print(f'Fold {fold} - {padded_count} files padded to the nearest second.')
    
    return concatenated_segments, concatenated_labels

# Main loop to process each fold
for fold in range(1, 11):
    print(f'Processing fold {fold}...')
    concatenated_segments, concatenated_labels = concatenate_audios(fold, audio_base_path, desired_length, segment_duration)
    
    # Save the concatenated segments
    output_fold_path = os.path.join(audio_base_path, f'concatenated_fold{fold}')
    os.makedirs(output_fold_path, exist_ok=True)
    
    metadata_records = []
    for i, (segment, labels) in enumerate(zip(concatenated_segments, concatenated_labels)):
        output_path = os.path.join(output_fold_path, f'concatenated_{i}.wav')
        segment.export(output_path, format='wav')
        metadata_records.append({
            'filename': f'concatenated_{i}.wav',
            'labels': ','.join(map(str, labels))
        })
    
    # Save the metadata for the concatenated segments
    metadata_df = pd.DataFrame(metadata_records)
    metadata_df.to_csv(os.path.join(output_fold_path, 'concatenated_metadata.csv'), index=False)
    
    print(f'Fold {fold} processed and saved.')

print('All folds processed and saved.')
