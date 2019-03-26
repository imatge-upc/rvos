import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
import os.path as osp
import glob
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import toimage
from scipy.optimize import linear_sum_assignment
import math
import argparse
import json
from utils.utils import make_dir
import numpy as np
from PIL import Image
import collections
import lmdb
from misc.config import cfg
from easydict import EasyDict as edict
import yaml
PALETTE = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0, 128]


def jaccard_simple(annotation,segmentation):

    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
    Return:
        jaccard (float): region similarity
 """

    annotation   = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)

    if np.isclose(np.sum(annotation),0) and np.isclose(np.sum(segmentation),0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / \
                np.sum((annotation | segmentation),dtype=np.float32)

def save_seq_results(model_name,seq_name,prev_mask):
    
    results_dir = os.path.join('../models', model_name, 'masks_sep_2assess_val_davis', seq_name)
    submission_dir = os.path.join('../models', model_name, 'Annotations-davis', seq_name)
    lmdb_env_seq = lmdb.open(osp.join(cfg.PATH.DATA, 'lmdb_seq'))
    make_dir(submission_dir)
    prev_assignment = []
    images_dir = os.path.join('../../databases/DAVIS2017/JPEGImages/480p/', seq_name)
    image_names = os.listdir(images_dir)
    image_names.sort()
    starting_img_name = image_names[0]
    starting_frame = int(starting_img_name[:-4])
    key_db = osp.basename(seq_name)
    with lmdb_env_seq.begin() as txn:
        _files_vec = txn.get(key_db.encode()).decode().split('|')
        _files = [osp.splitext(f)[0] for f in _files_vec]
    
    frame_names = _files

    frame_names.sort()
    frame_idx = 0
    obj_ids_sorted_increasing_jaccard = []
    
    for frame_name in frame_names:
        
        if frame_idx == 0:
            
            annotation = np.array(Image.open('../../databases/DAVIS2017/Annotations/480p/' + seq_name + '/' + frame_name + '.png'))
            instance_ids = sorted(np.unique(annotation))
            instance_ids = instance_ids if instance_ids[0] else instance_ids[1:]
            if len(instance_ids) > 0:
                instance_ids = instance_ids[:-1] if instance_ids[-1]==255 else instance_ids

            res_im = Image.fromarray(annotation, mode="P")
            res_im.putpalette(PALETTE)
            res_im.save(submission_dir + '/' + frame_name + '.png')
            
            #compute assignment between predictions from first frame and ground truth from first frame
            if prev_mask == True:
                num_preds = len(instance_ids)
            else:
                num_preds = 10
            cost = np.ones((len(instance_ids),num_preds))
            for obj_id in instance_ids:
                annotation_obj = np.zeros(annotation.shape)
                annotation_obj[annotation==obj_id] = 1
                for pred_id in range(num_preds):
                    if prev_mask == True:
                        pred_mask = imread(results_dir + '/' + frame_name + '_instance_%02d.png' %pred_id)
                    else:
                        pred_mask = imread(results_dir + '/' + '%05d_instance_%02d.png' %(starting_frame + frame_idx,pred_id))
                    pred_mask_resized = imresize(pred_mask, annotation.shape, interp='nearest')
                    cost[obj_id-1,pred_id] = 1- jaccard_simple(annotation_obj, pred_mask_resized)
            
            row_ind, col_ind = linear_sum_assignment(cost)

            prev_assignment = col_ind

            cost_objs = {}
            for obj_id in instance_ids:
                cost_objs[obj_id] = cost[obj_id-1,prev_assignment[obj_id-1]]
            obj_ids_sorted_increasing_jaccard = sorted(cost_objs.items(), key=lambda kv: kv[1], reverse = True)  

        else:
            
            pred_mask_resized = np.zeros(annotation.shape, dtype=np.uint8)
                            
            for obj_id, jaccard_val in obj_ids_sorted_increasing_jaccard:
                instance_assigned_id = prev_assignment[obj_id-1]
                if prev_mask == True:
                    pred_mask = imread(results_dir + '/' + frame_name + '_instance_%02d.png' %instance_assigned_id)
                else:
                    pred_mask = imread(results_dir + '/' + '%05d_instance_%02d.png' %(starting_frame + frame_idx,instance_assigned_id))

                pred_mask_resized_aux = imresize(pred_mask, annotation.shape, interp='nearest')
                pred_mask_resized[pred_mask_resized_aux==255]=obj_id

            res_im = Image.fromarray(pred_mask_resized, mode="P")
            res_im.putpalette(PALETTE)
            res_im.save(submission_dir + '/' + frame_name + '.png')
                

        frame_idx = frame_idx + 1                               
    

        
if __name__ == "__main__":
     
    parser = argparse.ArgumentParser(description='Plot visual results.')
    parser.add_argument('-model_name', dest='model_name', default='model')
    parser.add_argument('--prev_mask', dest='prev_mask', action='store_true')
    parser.set_defaults(prev_mask=False)
    args = parser.parse_args()
    
    with open('./dataloader/db_info.yaml','r') as f:
        sequences = edict(yaml.load(f)).sequences
    
    sequences = filter(lambda s:s.set == 'test-dev',sequences)

    submission_base_dir = os.path.join('../models', args.model_name, 'Annotations-davis')
    make_dir(submission_base_dir)

    
    for seq_name in sequences:
        save_seq_results(args.model_name,seq_name.name, args.prev_mask)
