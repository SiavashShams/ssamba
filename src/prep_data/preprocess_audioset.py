"""
Author: Siavash Shams
Date: 03/29/2024
"""

import os
import torchaudio
import logging
from pathlib import Path

def preprocess_audio_files(directory):
    file_count = 0
    error_files = []

    # Setup logging for files that cannot be opened
    logging.basicConfig(filename='preprocess_errors.log', level=logging.ERROR, format='%(asctime)s %(message)s')

    # Iterate through all files in the directory
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.flac'):
                filepath = os.path.join(subdir, file)
                try:
                    # Load the audio file
                    waveform, sample_rate = torchaudio.load(filepath)
                    
                    # Convert stereo to mono by averaging the channels if there are 2 channels
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    
                    # Resample to 16000 Hz if necessary
                    if sample_rate != 16000:
                        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                        waveform = resampler(waveform)
                    
                    # Save the processed file, overwrite the original file or save to a new path as needed
                    torchaudio.save(filepath, waveform, 16000)
                    
                    file_count += 1
                    if file_count % 2000 == 0:
                        print(f"Processed {file_count} files.")
                except Exception as e:
                    error_files.append(filepath)
                    logging.error(f"Error processing file {filepath}: {e}")
    
    # Report files that couldn't be processed
    if error_files:
        print(f"Finished with errors. Check 'preprocess_errors.log' for details on files that could not be processed.")
    else:
        print(f"Successfully processed all files.")

# Directories to preprocess
directories = [
    "/engram/naplab/shared/audioset/audio/bal_train",
    "/engram/naplab/shared/audioset/audio/eval",
    "/engram/naplab/shared/audioset/audio/unbal_train",
]

for directory in directories:
    print(f"Processing {directory}...")
    preprocess_audio_files(directory)
    print(f"Finished processing {directory}.")
