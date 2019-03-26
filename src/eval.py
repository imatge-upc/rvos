import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from args import get_parser
from utils.utils import batch_to_var, batch_to_var_test, make_dir, outs_perms_to_cpu, load_checkpoint, check_parallel
from modules.model import RSIS, FeatureExtractor
from test import test
from dataloader.dataset_utils import sequence_palette
from PIL import Image
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize
from scipy.misc import toimage
#import scipy
from dataloader.dataset_utils import get_dataset
import torch
import numpy as np
from torchvision import transforms
import torch.utils.data as data
import sys, os
import json



class Evaluate():

    def __init__(self,args):

        self.split = args.eval_split
        self.dataset = args.dataset
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        image_transforms = transforms.Compose([to_tensor,normalize])
        
        if args.dataset == 'davis2017':
            dataset = get_dataset(args,
                                split=self.split,
                                image_transforms=image_transforms,
                                target_transforms=None,
                                augment=args.augment and self.split == 'train',
                                inputRes = (240,427),
                                video_mode = True,
                                use_prev_mask = False)
        else: #args.dataset == 'youtube'
            dataset = get_dataset(args,
                                split=self.split,
                                image_transforms=image_transforms,
                                target_transforms=None,
                                augment=args.augment and self.split == 'train',
                                inputRes = (256, 448),
                                video_mode = True,
                                use_prev_mask = False)

        self.loader = data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=args.num_workers,
                                         drop_last=False)

        self.args = args

        print(args.model_name)
        encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, load_args = load_checkpoint(args.model_name,args.use_gpu)
        load_args.use_gpu = args.use_gpu
        self.encoder = FeatureExtractor(load_args)
        self.decoder = RSIS(load_args)

        print(load_args)

        if args.ngpus > 1 and args.use_gpu:
            self.decoder = torch.nn.DataParallel(self.decoder,device_ids=range(args.ngpus))
            self.encoder = torch.nn.DataParallel(self.encoder,device_ids=range(args.ngpus))

        encoder_dict, decoder_dict = check_parallel(encoder_dict,decoder_dict)
        self.encoder.load_state_dict(encoder_dict)
        to_be_deleted_dec = []
        for k in decoder_dict.keys():
            if 'fc_stop' in k:
                to_be_deleted_dec.append(k)
        for k in to_be_deleted_dec:
            del decoder_dict[k]
        self.decoder.load_state_dict(decoder_dict)

        if args.use_gpu:
            self.encoder.cuda()
            self.decoder.cuda()

        self.encoder.eval()
        self.decoder.eval()
        if load_args.length_clip == 1:
            self.video_mode = False
            print('video mode not activated')
        else:
            self.video_mode = True
            print('video mode activated')
 

    def run_eval(self):
        print ("Dataset is %s"%(self.dataset))
        print ("Split is %s"%(self.split))

        if args.overlay_masks:

            colors = []
            palette = sequence_palette()
            inv_palette = {}
            for k, v in palette.items():
                inv_palette[v] = k
            num_colors = len(inv_palette.keys())
            for id_color in range(num_colors):
                if id_color == 0 or id_color == 21:
                    continue
                c = inv_palette[id_color]
                colors.append(c)

        if self.split == 'val':
            
            if args.dataset == 'youtube':

                masks_sep_dir = os.path.join('../models', args.model_name, 'masks_sep_2assess')
                make_dir(masks_sep_dir)
                if args.overlay_masks:
                    results_dir = os.path.join('../models', args.model_name, 'results')
                    make_dir(results_dir)
            
                json_data = open('../../databases/YouTubeVOS/train/train-val-meta.json')
                data = json.load(json_data)
                
            else:

                masks_sep_dir = os.path.join('../models', args.model_name, 'masks_sep_2assess-davis')
                make_dir(masks_sep_dir)
                if args.overlay_masks:
                    results_dir = os.path.join('../models', args.model_name, 'results-davis')
                    make_dir(results_dir)


            for batch_idx, (inputs, targets,seq_name,starting_frame) in enumerate(self.loader):
                
                prev_hidden_temporal_list = None
                max_ii = min(len(inputs),args.length_clip)

                base_dir_masks_sep = masks_sep_dir + '/' + seq_name[0] + '/'
                make_dir(base_dir_masks_sep)

                if args.overlay_masks:
                    base_dir = results_dir + '/' + seq_name[0] + '/'
                    make_dir(base_dir)
                
                for ii in range(max_ii):

                    #                x: input images (N consecutive frames from M different sequences)
                    #                y_mask: ground truth annotations (some of them are zeros to have a fixed length in number of object instances)
                    #                sw_mask: this mask indicates which masks from y_mask are valid
                    x, y_mask, sw_mask = batch_to_var(args, inputs[ii], targets[ii])

                    print(seq_name[0] + '/' + '%05d' % (starting_frame[0] + ii))
                    
                    #from one frame to the following frame the prev_hidden_temporal_list is updated.
                    outs, hidden_temporal_list = test(args, self.encoder, self.decoder, x, prev_hidden_temporal_list)

                    if args.dataset == 'youtube':
                        num_instances = len(data['videos'][seq_name[0]]['objects'])
                    else:
                        num_instances = int(torch.sum(sw_mask.data).data.cpu().numpy())

                    x_tmp = x.data.cpu().numpy()
                    height = x_tmp.shape[-2]
                    width = x_tmp.shape[-1]
                    for t in range(10):
                        mask_pred = (torch.squeeze(outs[0, t, :])).cpu().numpy()
                        mask_pred = np.reshape(mask_pred, (height, width))
                        indxs_instance = np.where(mask_pred > 0.5)
                        mask2assess = np.zeros((height, width))
                        mask2assess[indxs_instance] = 255
                        toimage(mask2assess, cmin=0, cmax=255).save(
                            base_dir_masks_sep + '%05d_instance_%02d.png' % (starting_frame[0] + ii, t))
                
                    if args.overlay_masks:

                        frame_img = x.data.cpu().numpy()[0,:,:,:].squeeze()
                        frame_img = np.transpose(frame_img, (1,2,0))
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        frame_img = std * frame_img + mean
                        frame_img = np.clip(frame_img, 0, 1)
                        plt.figure();plt.axis('off')
                        plt.figure();plt.axis('off')
                        plt.imshow(frame_img)

                        for t in range(num_instances):
                            
                            mask_pred = (torch.squeeze(outs[0][0,t,:])).cpu().numpy()
                            mask_pred = np.reshape(mask_pred, (height, width))
                            ax = plt.gca()
                            tmp_img = np.ones((mask_pred.shape[0], mask_pred.shape[1], 3))
                            color_mask = np.array(colors[t])/255.0
                            for i in range(3):
                                tmp_img[:,:,i] = color_mask[i]
                            ax.imshow(np.dstack( (tmp_img, mask_pred*0.7) ))
                            
                        figname = base_dir + 'frame_%02d.png' %(starting_frame[0]+ii)
                        plt.savefig(figname,bbox_inches='tight')
                        plt.close()

                    if self.video_mode:
                        prev_hidden_temporal_list = hidden_temporal_list
            
        else:
            
            if args.dataset == 'youtube':

                masks_sep_dir = os.path.join('../models', args.model_name, 'masks_sep_2assess_val')
                make_dir(masks_sep_dir)
                if args.overlay_masks:
                    results_dir = os.path.join('../models', args.model_name, 'results_val')
                    make_dir(results_dir)
                
                json_data = open('../../databases/YouTubeVOS/val/meta.json')
                data = json.load(json_data)
                
            else:

                masks_sep_dir = os.path.join('../models', args.model_name, 'masks_sep_2assess_val_davis')
                make_dir(masks_sep_dir)
                if args.overlay_masks:
                    results_dir = os.path.join('../models', args.model_name, 'results_val_davis')
                    make_dir(results_dir)
        
            for batch_idx, (inputs,seq_name,starting_frame) in enumerate(self.loader):
                
                prev_hidden_temporal_list = None
                max_ii = min(len(inputs),args.length_clip)
                
                for ii in range(max_ii):

                    #                x: input images (N consecutive frames from M different sequences)
                    x = batch_to_var_test(args, inputs[ii])

                    print(seq_name[0] + '/' +  '%05d' %(starting_frame[0] + ii))
                    
                    if ii==0:
                        
                        if args.dataset == 'youtube':
                            
                            num_instances = len(data['videos'][seq_name[0]]['objects'])
                        
                        else:
                        
                            annotation = Image.open('../../databases/DAVIS2017/Annotations/480p/' + seq_name[0] + '/00000.png')
                            instance_ids = sorted(np.unique(annotation))
                            instance_ids = instance_ids if instance_ids[0] else instance_ids[1:]
                            if len(instance_ids) > 0:
                                instance_ids = instance_ids[:-1] if instance_ids[-1]==255 else instance_ids
                            num_instances = len(instance_ids)
                    
                    #from one frame to the following frame the prev_hidden_temporal_list is updated.
                    outs, hidden_temporal_list = test(args, self.encoder, self.decoder, x, prev_hidden_temporal_list)

                    base_dir_masks_sep = masks_sep_dir + '/' + seq_name[0] + '/'
                    make_dir(base_dir_masks_sep)

                    if args.overlay_masks:
                        base_dir = results_dir + '/' + seq_name[0] + '/'
                        make_dir(base_dir)

                    x_tmp = x.data.cpu().numpy()
                    height = x_tmp.shape[-2]
                    width = x_tmp.shape[-1]
                    for t in range(10):
                        mask_pred = (torch.squeeze(outs[0, t, :])).cpu().numpy()
                        mask_pred = np.reshape(mask_pred, (height, width))
                        indxs_instance = np.where(mask_pred > 0.5)
                        mask2assess = np.zeros((height, width))
                        mask2assess[indxs_instance] = 255
                        toimage(mask2assess, cmin=0, cmax=255).save(
                            base_dir_masks_sep + '%05d_instance_%02d.png' % (starting_frame[0] + ii, t))

                    if args.overlay_masks:

                        frame_img = x.data.cpu().numpy()[0,:,:,:].squeeze()
                        frame_img = np.transpose(frame_img, (1,2,0))
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        frame_img = std * frame_img + mean
                        frame_img = np.clip(frame_img, 0, 1)
                        plt.figure();plt.axis('off')
                        plt.figure();plt.axis('off')
                        plt.imshow(frame_img)

                        for t in range(num_instances):
                            
                            mask_pred = (torch.squeeze(outs[0,t,:])).cpu().numpy()
                            mask_pred = np.reshape(mask_pred, (height, width))
                            ax = plt.gca()
                            tmp_img = np.ones((mask_pred.shape[0], mask_pred.shape[1], 3))
                            color_mask = np.array(colors[t])/255.0
                            for i in range(3):
                                tmp_img[:,:,i] = color_mask[i]
                            ax.imshow(np.dstack( (tmp_img, mask_pred*0.7) ))
                            
                        figname = base_dir + 'frame_%02d.png' %(starting_frame[0]+ii)
                        plt.savefig(figname,bbox_inches='tight')
                        plt.close()
                            
                    if self.video_mode:
                        prev_hidden_temporal_list = hidden_temporal_list


if __name__ == "__main__":
    
    parser = get_parser()
    args = parser.parse_args()
    
    gpu_id = args.gpu_id
    if args.use_gpu:
        torch.cuda.set_device(device=gpu_id)
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    if not args.log_term:
        print ("Eval logs will be saved to:", os.path.join('../models',args.model_name, 'eval.log'))
        sys.stdout = open(os.path.join('../models',args.model_name, 'eval.log'), 'w')

    E = Evaluate(args)
    E.run_eval()
