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
    
    results_dir = os.path.join('../models', model_name, 'masks_sep_2assess_val', seq_name)
    submission_dir = os.path.join('../models', model_name, 'Annotations', seq_name)
    make_dir(submission_dir)
    json_data = open('../../databases/YouTubeVOS/val/meta.json')
    data = json.load(json_data)
    seq_data = data['videos'][seq_name]['objects']
    prev_obj_ids = []
    prev_assignment = []
    images_dir = os.path.join('../../databases/YouTubeVOS/val/JPEGImages/', seq_name)
    image_names = os.listdir(images_dir)
    image_names.sort()
    starting_img_name = image_names[0]
    starting_frame = int(starting_img_name[:-4])
    frame_names = []
    min_obj_id = 5
    max_obj_id = 1
    for obj_id in seq_data.keys():
        if int(obj_id) < min_obj_id:
            min_obj_id = int(obj_id)
        if int(obj_id) > max_obj_id:
            max_obj_id = int(obj_id)
        for frame_name in seq_data[obj_id]['frames']:
            if frame_name not in frame_names:
                frame_names.append(frame_name)
    

    frame_names.sort()
    frame_idx = 0
    obj_ids_sorted_increasing_jaccard = []
    
    for frame_name in frame_names:
        obj_ids = []
        for obj_id in seq_data.keys():
            if frame_name in seq_data[obj_id]['frames']:
                obj_ids.append(int(obj_id))

        if frame_idx == 0:
            
            annotation = np.array(Image.open('../../databases/YouTubeVOS/val/Annotations/' + seq_name + '/' + frame_name + '.png'))

            res_im = Image.fromarray(annotation, mode="P")
            res_im.putpalette(PALETTE)
            res_im.save(submission_dir + '/' + frame_name + '.png')
            
            #compute assignment between predictions from first frame and ground truth from first frame
            if prev_mask == True:
                num_preds = max_obj_id-min_obj_id+1
            else:
                num_preds = 10
            cost = np.ones((max_obj_id-min_obj_id+1,num_preds))
            for obj_id in obj_ids:
                annotation_obj = np.zeros(annotation.shape)
                annotation_obj[annotation==obj_id] = 1
                for pred_id in range(num_preds):
                    if prev_mask == True:
                        pred_mask = imread(results_dir + '/' + frame_name + '_instance_%02d.png' %pred_id)
                    else:
                        pred_mask = imread(results_dir + '/' + '%05d_instance_%02d.png' %(starting_frame + frame_idx,pred_id))
                    pred_mask_resized = imresize(pred_mask, annotation.shape, interp='nearest')
                    cost[obj_id-min_obj_id,pred_id] = 1- jaccard_simple(annotation_obj, pred_mask_resized)
            
            row_ind, col_ind = linear_sum_assignment(cost)
            prev_assignment = col_ind
            prev_obj_ids = obj_ids

            cost_objs = {}
            for obj_id in obj_ids:
                cost_objs[obj_id] = cost[obj_id-min_obj_id,prev_assignment[obj_id-min_obj_id]]
            obj_ids_sorted_increasing_jaccard = sorted(cost_objs.items(), key=lambda kv: kv[1], reverse = True)  

        else:
            
            new_elems = [];
            for obj_id in obj_ids:
                if obj_id not in prev_obj_ids:
                    new_elems.append(obj_id)
            
            pred_mask_resized = np.zeros(annotation.shape, dtype=np.uint8)
            
            if len(new_elems)==0:
                
                for obj_id, jaccard_val in obj_ids_sorted_increasing_jaccard:
                    instance_assigned_id = prev_assignment[obj_id-min_obj_id]
                    if prev_mask == True:
                        pred_mask = imread(results_dir + '/' + frame_name + '_instance_%02d.png' %instance_assigned_id)
                    else:
                        pred_mask = imread(results_dir + '/' + '%05d_instance_%02d.png' %(starting_frame + frame_idx,instance_assigned_id))

                    pred_mask_resized_aux = imresize(pred_mask, annotation.shape, interp='nearest')
                    pred_mask_resized[pred_mask_resized_aux==255]=obj_id

                res_im = Image.fromarray(pred_mask_resized, mode="P")
                res_im.putpalette(PALETTE)
                res_im.save(submission_dir + '/' + frame_name + '.png')
                
            else:
                
                prev_cost_objs = cost_objs
                cost_objs = {}

                annotation = np.array(Image.open('../../databases/YouTubeVOS/val/Annotations/' + seq_name + '/' + frame_name + '.png'))

                if prev_mask == True:
                    num_preds = max_obj_id - min_obj_id + 1
                else:
                    num_preds = 10
                cost = np.ones((max_obj_id-min_obj_id+1,num_preds))
                for obj_id in obj_ids:
                    if obj_id in prev_obj_ids:
                        cost[obj_id-min_obj_id,prev_assignment[obj_id-min_obj_id]] = 0
                    else:
                        annotation_obj = np.zeros(annotation.shape)
                        annotation_obj[annotation==obj_id] = 1
                        for pred_id in range(num_preds):
                            if prev_mask == True:
                                pred_mask = imread(results_dir + '/' + frame_name + '_instance_%02d.png' %pred_id)
                            else:
                                pred_mask = imread(results_dir + '/' + '%05d_instance_%02d.png' %(starting_frame + frame_idx,pred_id))
                            pred_mask_resized = imresize(pred_mask, annotation.shape, interp='nearest')
                            cost[obj_id-min_obj_id,pred_id] = 1- jaccard_simple(annotation_obj, pred_mask_resized)
                
                row_ind, col_ind = linear_sum_assignment(cost)
                prev_assignment = col_ind
                
                for obj_id in obj_ids:
                    if obj_id in prev_obj_ids:
                        cost_objs[obj_id] = prev_cost_objs[obj_id]
                    else:
                        cost_objs[obj_id] = cost[obj_id-min_obj_id,prev_assignment[obj_id-min_obj_id]]

                obj_ids_sorted_increasing_jaccard = sorted(cost_objs.items(), key=lambda kv: kv[1], reverse = True)

                pred_mask_resized = np.zeros(annotation.shape, dtype=np.uint8)

                for obj_id, jaccard_val in obj_ids_sorted_increasing_jaccard:
                    if obj_id in prev_obj_ids:

                        instance_assigned_id = prev_assignment[obj_id-min_obj_id]
                        if prev_mask == True:
                            pred_mask = imread(results_dir + '/' + frame_name + '_instance_%02d.png' %instance_assigned_id)
                        else:
                            pred_mask = imread(results_dir + '/' + '%05d_instance_%02d.png' %(starting_frame + frame_idx,instance_assigned_id))
                        pred_mask_resized_aux = imresize(pred_mask, annotation.shape, interp='nearest')
                        pred_mask_resized[pred_mask_resized_aux==255]=obj_id
                        
                for obj_id in obj_ids:
                    if obj_id not in prev_obj_ids:
                        annotation_obj = np.zeros(annotation.shape)
                        annotation_obj[annotation==obj_id] = 255
                        pred_mask_resized[annotation_obj==255]=obj_id
                
                res_im = Image.fromarray(pred_mask_resized, mode="P")
                res_im.putpalette(PALETTE)
                res_im.save(submission_dir + '/' + frame_name + '.png')
                prev_obj_ids = obj_ids

        frame_idx = frame_idx + 1                               
    

        
if __name__ == "__main__":
     
    parser = argparse.ArgumentParser(description='Plot visual results.')
    parser.add_argument('-model_name', dest='model_name', default='model')
    parser.add_argument('--prev_mask', dest='prev_mask', action='store_true')
    parser.set_defaults(prev_mask=False)
    args = parser.parse_args()
    
    json_data = open('../../databases/YouTubeVOS/val/meta.json')
    data = json.load(json_data)
    sequences = data['videos'].keys()
    
    submission_base_dir = os.path.join('../models', args.model_name, 'Annotations')
    make_dir(submission_base_dir)
    
    for seq_name in sequences:
        save_seq_results(args.model_name,seq_name, args.prev_mask)
