import numpy as np
from scipy.ndimage.interpolation import zoom
import torch
import random

def get_dataset(args, split, image_transforms = None, target_transforms = None, augment = False,inputRes = None, video_mode = True, use_prev_mask = False):


    if args.dataset =='davis2017':
        from .davis2017 import DAVISLoader as MyChosenDataset
    elif args.dataset == 'youtube':
        from .youtubeVOS import YoutubeVOSLoader as MyChosenDataset
    
    


    dataset = MyChosenDataset(args,
                            split = split,
                            transform = image_transforms,
                            target_transform = target_transforms,
                            augment = augment,
                            resize = args.resize,
                            inputRes = inputRes,
                            video_mode = video_mode,
                            use_prev_mask = use_prev_mask)
    return dataset
    
def sequence_palette():

    # RGB to int conversion

    palette = {(  0,   0,   0) : 0 ,
             (0,   255,   0) : 1 ,
             (  255, 0,   0) : 2 ,
             (0, 0,   255) : 3 ,
             (  255,   0, 255) : 4 ,
             (0,   255, 255) : 5 ,
             (  255, 128, 0) : 6 ,
             (102, 0, 102) : 7 ,
             ( 51,   153,   255) : 8 ,
             (153,   153,   255) : 9 ,
             ( 153, 153,   0) : 10,
             (178, 102,   255) : 11,
             ( 204,   0, 204) : 12,
             (0,   102, 0) : 13,
             ( 102, 0, 0) : 14,
             (51, 0, 0) : 15,
             (  0,  64,   0) : 16,
             (128,  64,   0) : 17,
             (  0, 192,   0) : 18,
             (128, 192,   0) : 19,
             (  0,  64, 128) : 20,
             (224,  224, 192) : 21 }

    return palette
