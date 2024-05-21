"""
Author: Siavash Shams
Date: 03/29/2024
"""

import os
import subprocess

# URLs for the LibriSpeech datasets
urls = [
    "http://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "http://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "http://www.openslr.org/resources/12/train-other-500.tar.gz"
]

# Target directory for downloads and extraction
target_dir = "/engram/naplab/shared/Librispeech"

# Ensure the target directory exists
os.makedirs(target_dir, exist_ok=True)

# Navigate to the target directory
os.chdir(target_dir)

for url in urls:
    # Extract the filename from the URL
    filename = url.split("/")[-1]
    
    # Download the file using wget
    print(f"Downloading {filename}...")
    subprocess.run(["wget", "-c", url])
    
    # Extract the downloaded file
    print(f"Extracting {filename}...")
    subprocess.run(["tar", "-xzf", filename])
    
    # Remove the downloaded .tar.gz file to save space
    print(f"Removing {filename}...")
    os.remove(filename)

print("All files downloaded and extracted successfully.")

