"""This module contains simple helper functions"""

from __future__ import print_function
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import torch.distributed as dist
import os


def tensor2im(input_image, imtype=np.uint8):
    
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor): 
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  
        if image_numpy.shape[0] == 1:  
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  
    else:  
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name="network"):
    
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def init_ddp():
    
    is_ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1

    if is_ddp:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")
    print(f"Initialized with device {device}")
    return device



def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    
    x = x.astype(np.float64)
    if shp:
        print("shape,", x.shape)
    if val:
        x = x.flatten()
        print("mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f" % (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    
    Path(path).mkdir(parents=True, exist_ok=True)
