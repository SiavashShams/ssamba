"""
Author: Siavash Shams
Date: 03/29/2024
"""

import os
import torchaudio
from pathlib import Path

def process_audio_file(input_filepath, output_directory, processed_files_log, error_files_log):
    try:
        waveform, sample_rate = torchaudio.load(input_filepath)
        original_waveform_shape = waveform.shape
        original_sample_rate = sample_rate

        # Convert stereo to mono by averaging the channels if there are 2 channels
        if waveform.shape[0] == 2:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 16000 Hz if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # Check if any changes have been made to the original file
        if waveform.shape != original_waveform_shape or sample_rate != original_sample_rate:
            filename = os.path.basename(input_filepath)
            output_filepath = os.path.join(output_directory, filename)
            torchaudio.save(output_filepath, waveform, sample_rate)
            processed_files_log.append(input_filepath)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error processing file {input_filepath}: {e}")
        error_files_log.append(input_filepath)
        return False

def preprocess_librispeech(input_directories, output_base_directory):
    total_processed = 0
    total_skipped = 0
    processed_files_log = []
    error_files_log = []

    for input_directory in input_directories:
        output_directory = os.path.join(output_base_directory, os.path.basename(input_directory))
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        
        for subdir, _, files in os.walk(input_directory):
            for file in files:
                if file.endswith('.flac'):
                    input_filepath = os.path.join(subdir, file)
                    processed = process_audio_file(input_filepath, output_directory, processed_files_log, error_files_log)
                    
                    if processed:
                        total_processed += 1
                    else:
                        total_skipped += 1

                    if total_processed % 2000 == 0:
                        print(f"Processed {total_processed} files.")

    # Log error files and processed files
    with open("libri_error_files_log.txt", "w") as ef:
        for error_file in error_files_log:
            ef.write(f"{error_file}\n")
    with open("processed_files_log.txt", "w") as pf:
        for processed_file in processed_files_log:
            pf.write(f"{processed_file}\n")

    print(f"Total processed files: {total_processed}")
    print(f"Total skipped files: {total_skipped}")
    print(f"Error files logged in error_files_log.txt")
    print(f"Processed files logged in processed_files_log.txt")

# Directories to preprocess
input_directories = [
    "/engram/naplab/shared/Librispeech/LibriSpeech/train-clean-100",
    "/engram/naplab/shared/Librispeech/LibriSpeech/train-clean-360",
    "/engram/naplab/shared/Librispeech/LibriSpeech/train-other-500",
]

# Base directory to save processed files
output_base_directory = "/engram/naplab/shared/Librispeech/ProcessedLibriSpeech"

preprocess_librispeech(input_directories, output_base_directory)
