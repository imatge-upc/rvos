import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='RIASS')
    
    parser.add_argument('-year', dest='year', default = '2017')
    parser.add_argument('-imsize',dest='imsize', default=480, type=int)
    parser.add_argument('-batch_size', dest='batch_size', default = 10, type=int)
    parser.add_argument('-num_workers', dest='num_workers', default = 1, type=int)
    parser.add_argument('-length_clip', dest='length_clip', default = 1, type=int)
    parser.add_argument('--single_object',dest='single_object', action='store_true')
    parser.set_defaults(single_object=False)
    parser.add_argument('--only_temporal',dest='only_temporal', action='store_true')
    parser.set_defaults(only_temporal=False)
    parser.add_argument('--only_spatial',dest='only_spatial', action='store_true')
    parser.set_defaults(only_spatial=False)

    ## TRAINING parameters ##
    parser.add_argument('--resume', dest='resume',action='store_true',
                        help=('whether to resume training an existing model '
                        '(the one with name model_name will be used)'))
    parser.set_defaults(resume=False)
    # set epoch_resume if you want flags --finetune_after and --update_encoder to be properly
    # activated (eg if you stop training for whatever reason at epoch 15, set epoch_resume to 15)
    parser.add_argument('-epoch_resume', dest='epoch_resume',default= 0,type=int,
                        help=('set epoch_resume if you want flags '
                        '--finetune_after and --update_encoder to be properly '
                        'activated (eg if you stop training for whatever reason '
                        'at epoch 15, set epoch_resume to 15)'))
    parser.add_argument('-seed', dest='seed',default = 123, type=int)
    parser.add_argument('-gpu_id', dest='gpu_id',default = 0, type=int)
    parser.add_argument('-lr', dest='lr', default = 1e-3,type=float)
    parser.add_argument('-lr_cnn', dest='lr_cnn', default = 1e-6,type=float)
    parser.add_argument('-optim_cnn', dest='optim_cnn', default = 'adam',
                        choices=['adam','sgd','rmsprop'])
    parser.add_argument('-momentum', dest='momentum', default =0.9,type=float)
    parser.add_argument('-weight_decay', dest='weight_decay', default = 1e-6, type=float)
    parser.add_argument('-weight_decay_cnn', dest='weight_decay_cnn', default = 1e-6, type=float)
    parser.add_argument('-optim', dest='optim', default = 'adam',
                        choices=['adam','sgd','rmsprop'])
    parser.add_argument('-maxseqlen', dest='maxseqlen', default = 10, type=int)
    parser.add_argument('-gt_maxseqlen', dest='gt_maxseqlen', default = 10, type=int)
    parser.add_argument('-best_val_loss', dest='best_val_loss', default = 1000, type=float)
    parser.add_argument('--crop', dest='crop', action='store_true')
    parser.set_defaults(crop=False)
    parser.add_argument('--smooth_curves',dest='smooth_curves', action='store_true')
    parser.set_defaults(smooth_curves=False)
    parser.add_argument('--overlay_masks', dest='overlay_masks', action='store_true')
    parser.set_defaults(overlay_masks=False)

    # base model fine tuning
    parser.add_argument('-finetune_after', dest='finetune_after', default = 0, type=int,
                        help=('epoch number to start finetuning. set -1 to not finetune.'
                        'there is a patience term that can allow starting to fine tune '
                        'earlier (does not apply if value is -1)'))
    parser.add_argument('--update_encoder', dest='update_encoder', action='store_true',
                        help='used in sync with finetune_after. no need to activate.')
    parser.set_defaults(update_encoder=False)

    parser.add_argument('--transfer',dest='transfer', action='store_true')
    parser.set_defaults(transfer=False)
    parser.add_argument('-transfer_from', dest='transfer_from', default = 'model')
    parser.add_argument('-min_delta', dest='min_delta', default=0.0, type=float)

    # stopping criterion
    parser.add_argument('-patience', dest='patience', default = 15, type=int,
                        help=('patience term to activate flags such as '
                        'use_class_loss, feed_prediction and update_encoder if '
                        'their matching vars are not -1'))
    parser.add_argument('-patience_stop', dest='patience_stop', default = 60, type=int,
                        help='patience to stop training.')
    parser.add_argument('-max_epoch', dest='max_epoch', default = 100, type=int)

    # visualization and logging
    parser.add_argument('-print_every', dest='print_every', default = 10, type=int)
    parser.add_argument('--log_term', dest='log_term', action='store_true',
                        help='if activated, will show logs in stdout instead of log file.')
    parser.set_defaults(log_term=False)
    parser.add_argument('--visdom', dest='visdom', action='store_true')
    parser.set_defaults(visdom=False)
    parser.add_argument('-port',dest='port',default=8097, type=int, help='visdom port')
    parser.add_argument('-server',dest='server',default='http://localhost', help='visdom server')

    # loss weights
    parser.add_argument('-iou_weight',dest='iou_weight',default=1.0, type=float)
    # augmentation
    parser.add_argument('--augment', dest='augment', action='store_true')
    parser.set_defaults(augment=False)
    parser.add_argument('-rotation', dest='rotation', default = 10, type=int)
    parser.add_argument('-translation', dest='translation', default = 0.1, type=float)
    parser.add_argument('-shear', dest='shear', default = 0.1, type=float)
    parser.add_argument('-zoom', dest='zoom', default = 0.7, type=float)

    # GPU
    parser.add_argument('--cpu', dest='use_gpu', action='store_false')
    parser.set_defaults(use_gpu=True)
    parser.add_argument('-ngpus', dest='ngpus', default=1,type=int)

    parser.add_argument('-base_model', dest='base_model', default = 'resnet101',
                        choices=['resnet101','resnet50','resnet34','vgg16'])
    parser.add_argument('-skip_mode', dest='skip_mode', default = 'concat',
                        choices=['sum','concat','mul','none'])
    parser.add_argument('-model_name', dest='model_name', default='model')
    parser.add_argument('-log_file', dest='log_file', default='train.log')
    parser.add_argument('-hidden_size', dest='hidden_size', default = 128, type=int)
    parser.add_argument('-kernel_size', dest='kernel_size', default = 3, type=int)
    parser.add_argument('-dropout', dest='dropout', default = 0.0, type=float)

    # dataset parameters
    parser.add_argument('--resize',dest='resize', action='store_true')
    parser.set_defaults(resize=False)
    parser.add_argument('-num_classes', dest='num_classes', default = 21, type=int)
    parser.add_argument('-dataset', dest='dataset', default = 'davis2017',choices=['davis2017', 'youtube'])
    parser.add_argument('-youtube_dir', dest='youtube_dir',
                        default='../../databases/YouTubeVOS/')

    # testing
    parser.add_argument('-eval_split',dest='eval_split', default='test')
    parser.add_argument('-mask_th',dest='mask_th', default=0.5, type=float)
    parser.add_argument('-max_dets',dest='max_dets', default=100, type=int)
    parser.add_argument('-min_size',dest='min_size', default=0.001, type=float)
    parser.add_argument('--display', dest='display', action='store_true')
    parser.add_argument('--no_display_text', dest='no_display_text', action='store_true')
    parser.set_defaults(display=False)
    parser.set_defaults(display_route=False)
    parser.set_defaults(no_display_text=False)
    parser.set_defaults(use_gt_masks=False)

    # demo
    parser.add_argument('-frames_path', dest='frames_path', default='../../databases/DAVIS2017/JPEGImages/480p/tennis-vest')
    parser.add_argument('-mask_path', dest='init_mask_path', default='../../databases/DAVIS2017/Annotations/480p/tennis-vest/00000.png')
    parser.add_argument('-results_path', dest='results_path', default=None)
    parser.add_argument('--zero_shot', dest='zero_shot', action='store_true')
    return parser

if __name__ =="__main__":

    parser = get_parser()
    args_dict = parser.parse_args()
