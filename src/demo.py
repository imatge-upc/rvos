# Demo file for RVOS

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from args import get_parser
from dataloader.dataset_utils import sequence_palette
from modules.model import RSIS, RSISMask, FeatureExtractor
from PIL import Image
from scipy.misc import imresize, toimage
from test import test_prev_mask, test
from torch.autograd import Variable
from torchvision import transforms
from utils.utils import batch_to_var_test, load_checkpoint, check_parallel


class Sequence:
    def __init__(self, args, seq_name):
        # Frames and annotation paths
        self.frames_path = args.frames_path
        self.init_mask_path = args.init_mask_path
        self.seq_name = seq_name
        # Frames information
        self.frames_list = None
        self.input_res = (240, 427)
        self.max_instances = args.maxseqlen  # Limit the max number of instances
        # Frame and annotation data
        self.imgs_data = []
        self.init_mask_data = None
        self.instance_ids = None
        # Frames normalization
        self.img_transforms = None
        self._generate_transform()
        # Initialize variables
        self._get_frames_list()
        self.load_frames()
        if not args.zero_shot:
            # Semi-supervised
            self.load_annot(args.use_gpu)
        if args.zero_shot:
            self.instance_ids = np.arange(0, 10)  # Get 10 instances for zero-shot

    def _get_frames_list(self):
        self.frames_list = sorted(os.listdir(self.frames_path))

    def load_frame(self, frame_path):
        img = Image.open(frame_path)
        if self.input_res is not None:
            img = imresize(img, self.input_res)
        if self.img_transforms is not None:
            img = self.img_transforms(img)

        return img

    def load_frames(self):
        for frame_name in self.frames_list:
            frame_path = os.path.join(self.frames_path, frame_name)
            img = self.load_frame(frame_path)
            self.imgs_data.append(img)

    def _generate_transform(self):
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.img_transforms = transforms.Compose([to_tensor, normalize])

    def load_annot(self, use_gpu):
        annot = Image.open(self.init_mask_path)
        if self.input_res is not None:
           annot = imresize(annot, self.input_res, interp='nearest')

        # Prepared for DAVIS-like annotations
        annot = np.expand_dims(annot, axis=0)
        annot = torch.from_numpy(annot)
        annot = annot.float()
        annot = annot.numpy().squeeze()
        annot = self.seg_from_annot(annot)

        prev_mask = annot
        prev_mask = np.expand_dims(prev_mask, axis=0)
        prev_mask = torch.from_numpy(prev_mask)
        y_mask = Variable(prev_mask.float(), requires_grad=False)
        if use_gpu:
            y_mask = y_mask.cuda()
        self.init_mask_data = y_mask

    def seg_from_annot(self, annot):
        instance_ids = sorted(np.unique(annot)[1:])

        h = annot.shape[0]
        w = annot.shape[1]

        total_num_instances = len(instance_ids)
        max_instance_id = 0
        if total_num_instances > 0:
            max_instance_id = int(np.max(instance_ids))
        num_instances = max(self.max_instances, max_instance_id)

        gt_seg = np.zeros((num_instances, h * w))

        for i in range(total_num_instances):
            id_instance = int(instance_ids[i])
            aux_mask = np.zeros((h, w))
            aux_mask[annot == id_instance] = 1
            gt_seg[id_instance - 1, :] = np.reshape(aux_mask, h * w)

        self.instance_ids = instance_ids
        gt_seg = gt_seg[:][:self.max_instances]

        return gt_seg


class Model:
    def __init__(self, args):
        # Define encoder and decoder
        self.encoder = None
        self.decoder = None
        # Mode
        self.video_mode = False
        # Load model
        self._init_model(args)

    def _init_model(self, args):
        print("Loading model: " + args.model_name)
        encoder_dict, decoder_dict, _, _, load_args = load_checkpoint(args.model_name, args.use_gpu)
        load_args.use_gpu = args.use_gpu
        self.encoder = FeatureExtractor(load_args)
        if args.zero_shot:
            self.decoder = RSIS(load_args)
        else:
            self.decoder = RSISMask(load_args)
        print(load_args)

        if args.ngpus > 1 and args.use_gpu:
            self.decoder = torch.nn.DataParallel(self.decoder, device_ids=range(args.ngpus))
            self.encoder = torch.nn.DataParallel(self.encoder, device_ids=range(args.ngpus))

        encoder_dict, decoder_dict = check_parallel(encoder_dict, decoder_dict)
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


class SaveResults:
    def __init__(self, seq_object, save_path):
        # Sequence info
        self.seq = seq_object
        self.save_path = save_path
        # Colors for overlay
        self.colors = self._init_colors()

    def save_result(self, x, net_outs, frame_name):
        x_tmp = x.data.cpu().numpy()
        height = x_tmp.shape[-2]
        width = x_tmp.shape[-1]

        for t in range(len(self.seq.instance_ids)):
            mask_pred = (torch.squeeze(net_outs[0, t, :])).cpu().numpy()
            mask_pred = np.reshape(mask_pred, (height, width))
            indxs_instance = np.where(mask_pred > 0.5)
            mask2assess = np.zeros((height, width))
            mask2assess[indxs_instance] = 255
            mask_save_path = os.path.join(self.save_path, frame_name + '_instance_%02d.png' % (t))

            toimage(mask2assess, cmin=0, cmax=255).save(mask_save_path)

    @staticmethod
    def _init_colors():
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

        return colors

    def save_result_overlay(self, x, net_outs, frame_name):
        x_tmp = x.data.cpu().numpy()
        height = x_tmp.shape[-2]
        width = x_tmp.shape[-1]

        frame_img = x.data.cpu().numpy()[0, :, :, :].squeeze()
        frame_img = np.transpose(frame_img, (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame_img = std * frame_img + mean
        frame_img = np.clip(frame_img, 0, 1)
        plt.figure();
        plt.axis('off')
        plt.figure();
        plt.axis('off')
        plt.imshow(frame_img)

        if self.seq.instance_ids is not None:
            for t in range(len(self.seq.instance_ids)):
                mask_pred = (torch.squeeze(net_outs[0, t, :])).cpu().numpy()
                mask_pred = np.reshape(mask_pred, (height, width))
                ax = plt.gca()
                tmp_img = np.ones((mask_pred.shape[0], mask_pred.shape[1], 3))
                color_mask = np.array(self.colors[t]) / 255.0
                for i in range(3):
                    tmp_img[:, :, i] = color_mask[i]
                ax.imshow(np.dstack((tmp_img, mask_pred * 0.7)))

        figname = os.path.join(self.save_path, frame_name + '.png')
        plt.savefig(figname, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.results_path is not None:
        masks_save_path = args.results_path
        seq_name = ''
    else:
        if args.frames_path[-1] == '/':
            args.frames_path = args.frames_path[:-1]
        seq_name = os.path.basename(args.frames_path)
        masks_save_path = os.path.join('../models', args.model_name, 'results')
        if not os.path.isdir(masks_save_path):
            os.mkdir(masks_save_path)
        masks_save_path = os.path.join(masks_save_path, seq_name)

    print('Results will be saved to: ' + masks_save_path)
    if not os.path.isdir(masks_save_path):
        os.mkdir(masks_save_path)

    gpu_id = args.gpu_id
    if args.use_gpu:
        torch.cuda.set_device(device=gpu_id)
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    # Sequence object
    seq = Sequence(args, seq_name)
    model = Model(args)
    results = SaveResults(seq, masks_save_path)

    prev_hidden_temporal_list = None
    prev_mask = seq.init_mask_data

    for ii, img in enumerate(seq.imgs_data):
        img = img.unsqueeze(0)
        x = batch_to_var_test(args, img)

        if not args.zero_shot:
            # One-shot
            outs, hidden_temporal_list = test_prev_mask(args, model.encoder, model.decoder, x,
                                                        prev_hidden_temporal_list, prev_mask)
        else:
            # Zero-shot
            outs, hidden_temporal_list = test(args, model.encoder, model.decoder, x, prev_hidden_temporal_list)

        frame_name = os.path.splitext(os.path.basename(seq.frames_list[ii]))[0]
        if args.overlay_masks:
            results.save_result_overlay(x, outs, frame_name)
        else:
            results.save_result(x, outs, frame_name)

        if model.video_mode:
            if not args.only_spatial:
                prev_hidden_temporal_list = hidden_temporal_list
            if ii > 0:
                prev_mask = outs
