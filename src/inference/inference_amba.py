import time

import sys

from models import ASTModel, AMBAModel
import sys
import os, csv, argparse, wget
import torch, torchaudio, timm
import numpy as np
from torch.cuda.amp import autocast
import IPython
import time
import argparse

parser = argparse.ArgumentParser(description='Run inference tests for different model sizes.')
parser.add_argument('--model_size', type=str, default='base', choices=['base', 'small', 'tiny'],
                    help='Model size to use for inference tests')
args = parser.parse_args()

model_size = args.model_size
csv_file_name = f'inference_times_{model_size}_amba_batch2.csv'

pretrain_path=f"/engram/naplab/shared/ssamba/models/amba_{model_size}_300.pth"

if model_size == 'base':
    embed_dim = 768
elif model_size == 'small':
    embed_dim = 384
elif model_size == 'tiny':
    embed_dim = 192
    
batch_size = 2
# Make the prediction
with open(csv_file_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Input Size', 'Run Number', 'Inference Time (seconds)', 'Memory Used (MB)'])
    
    # Test with different input sizes
    for size in range(1024, 151680, 512):  # Adjust range and step as needed
        for run in range(5):              
            vision_mamba_config = {
            'img_size': (128, size),
            'patch_size': 16,
            'stride': 16,
            'embed_dim': embed_dim,
            'depth': 24,
            'channels': 1,
            'num_classes': 1000,
            'drop_rate': 0.,
            'drop_path_rate': 0.1,
            'norm_epsilon': 1e-5,
            'rms_norm': False,
            'residual_in_fp32': False,
            'fused_add_norm': False,
            'if_rope': False,
            'if_rope_residual': False,
            'bimamba_type': "v2",
            'if_bidirectional': True,
            'final_pool_type': 'none',
            'if_abs_pos_embed': True,
            'if_bimamba': False,
            'if_cls_token': True,
            'if_devide_out': True,
            'use_double_cls_token': False,
            'use_middle_cls_token': False,
            }

            audio_model = AMBAModel(label_dim=50, fshape=16, tshape=16, fstride=16, tstride=16,
                           input_fdim=128, input_tdim=size, model_size='base', pretrain_stage=False, load_pretrained_mdl_path=pretrain_path, vision_mamba_config=vision_mamba_config)
            if not isinstance(audio_model, torch.nn.DataParallel):
                audio_model = torch.nn.DataParallel(audio_model).to('cuda')
            # (batch, tdim, fdim)
            feats = torch.randn(batch_size, size, 128).to('cuda')  # Random data simulating the features

            # Start timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            
            torch.cuda.reset_peak_memory_stats()
            memory_before = torch.cuda.memory_allocated() / (1024 ** 2)  

            # Perform inference
            with torch.no_grad():
                output = audio_model(x=feats, task="ft_avgtok")
                output = torch.sigmoid(output)
                
            end_event.record()
            torch.cuda.synchronize()
            # End timing
            inference_time = start_event.elapsed_time(end_event) / 1000.0  
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  
            
            
            writer.writerow([size, run + 1, inference_time, peak_memory])
            print(f'Run {run + 1}: Input Size: {size}, Inference Time: {inference_time:.4f} seconds, Memory Usage: {peak_memory:.2f} MB')
            del feats, output
            torch.cuda.empty_cache()  

            
            
print(f'Inference times recorded in {csv_file_name}.')