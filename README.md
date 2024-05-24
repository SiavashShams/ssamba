# SSAMBA: Self-Supervised Audio Mamba

⚠️ **Under Construction** ⚠️

We will add recipes for fine-tuning on more datasets later. 🛠️ Stay tuned!!!


## Introduction
This repository contains the official implementation (in PyTorch) of the the paper SSAMBA: Self-Supervised Audio Representation Learning with Mamba State Space Model. SSAMBA is an advanced audio representation learning model designed to leverage self-supervised learning techniques using the Mamba State Space Model. This project builds on the success of the Self-Supervised Audio Spectrogram Transformer (SSAST) and introduces novel methodologies to further enhance performance and efficiency on various audio tasks. 

## Installation

To install the necessary dependencies, you can use the following commands:

```bash
git clone https://github.com/SiavashShams/ssamba.git
cd ssamba
pip install -r requirements.txt
```

## Architecture

![architecture](figures/ssamba.png)

## Efficiency Comparison
SSAMBA is approximately 92.7\% faster in batch inference speed and 95.4\% more memory-efficient than SSAST for the tiny model size with an input token size of 22k.
<p align="center">
  <img src="figures/inference_time_b4.png" alt="Models Inference Speed" width="45%" />
  <img src="figures/gpu_memory_b4.png" alt="Models GPU Memory" width="45%" />
</p>

## Pretraining

We pretrained SSAMBA with various sizes (base, small, tiny) for patches (250, 300, and 400) on a mixture of unlabeled audios from AudioSet and LibriSpeech. You can find these weights in the "Pretrained Model Weights" section below. However, if you want to pretrain the model from scratch, follow this recipe:

1. **Navigate to the Directory**: Change to the directory containing the pretraining scripts. You can do this by running the following command in your terminal:
    ```bash
    cd ssamba/src/pretrain
    ```

2. **Adjust the Script**: Edit the `run_mask_patch_amba.sh` script to update the paths to your data files, Mamba encoder configurations, and any other necessary hyperparameters. Make sure that all paths and settings accurately reflect your local environment and the specifics of the dataset you are using.

3. **Run the Script**: After making the necessary adjustments, execute the script to start the pretraining process. You can run the script directly from the terminal with the following command:
    ```bash
    ./run_mask_patch_amba.sh
    ```

## Pretrained Model Weights

The pretrained model weights for our SSAMBA model in sizes (base, small, and tiny) for different number of masked patches (400, 300, 250) can be found at:

[Pretrained Model Weights](https://drive.google.com/drive/u/1/folders/1E1gf5SxdSByDJ16_WQvzTKn8lIoYtZiX)

## Finetuning

### Audioset_20k and ESC-50

To finetune the pretrained SSAMBA on the balanced Audioset or ESC-50 datasets, follow these steps:

1. **Navigate to the finetuning directory:**
   - For Audioset:
     ```bash
     cd src/finetune/audioset
     ```
   - For ESC-50:
     ```bash
     cd src/finetune/esc50
     ```

2. **Adjust the paths and hyperparameters:**
   Edit `run_as_amba.sh` and `run_esc_patch_amba.sh`. Adjust the paths and hyperparameters as needed for your dataset.

3. **Configure SLURM job submission (if using SLURM):**
   Add the models you want to finetune to `submit_jobs.sh`:
   ```bash
   #!/bin/bash

   # Array of pre-trained models
   declare -a models=("ssamba_tiny_400")

   # Submit a job for each model
   for model in "${models[@]}"; do
       sbatch run_as_amba.sh $model
   done
   ```

4. **Run the job submission script:**
   Execute the `submit_jobs.sh` script in the terminal to start the finetuning process:
   ```bash
   ./submit_jobs.sh
   ```

Make sure to monitor the jobs and adjust any parameters as needed to suit your specific requirements and hardware configuration.


## License
The license for borrowed code can be found in [LICENSE](https://github.com/SiavashShams/ssamba/blob/main/LICENSE) file. 
We acknowledge the wonderful work of [SSAST](https://arxiv.org/abs/2110.09784), and [Vision Mamba](https://arxiv.org/abs/2401.09417). 

## Citing
If you find this work helpful, please consider giving us a star 🌟 and citing:

```bibtex
@article{shams2024ssamba,
      title={SSAMBA: Self-Supervised Audio Representation Learning with Mamba State Space Model},
      author={Siavash Shams and Sukru Samet Dindar and Xilin Jiang and Nima Mesgarani},
      year={2024},
      eprint={2405.11831},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      journal={arXiv preprint arXiv:2405.11831}
}

```
