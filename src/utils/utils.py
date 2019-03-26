from torch.autograd import Variable
import torch
import os
import numpy as np
import pickle
from collections import OrderedDict

def make_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def check_parallel(encoder_dict,decoder_dict):
	# check if the model was trained using multiple gpus
    trained_parallel = False
    for k, v in encoder_dict.items():
        if k[:7] == "module.":
            trained_parallel = True
        break;
    if trained_parallel:
        # create new OrderedDict that does not contain "module."
        new_encoder_state_dict = OrderedDict()
        new_decoder_state_dict = OrderedDict()
        for k, v in encoder_dict.items():
            name = k[7:]  # remove "module."
            new_encoder_state_dict[name] = v
        for k, v in decoder_dict.items():
            name = k[7:]  # remove "module."
            new_decoder_state_dict[name] = v
        encoder_dict = new_encoder_state_dict
        decoder_dict = new_decoder_state_dict

    return encoder_dict, decoder_dict

def get_base_params(args, model):
    b = []
    if 'vgg' in args.base_model:
        b.append(model.base.features)
    else:
        b.append(model.base.conv1)
        b.append(model.base.bn1)
        b.append(model.base.layer1)
        b.append(model.base.layer2)
        b.append(model.base.layer3)
        b.append(model.base.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

def get_skip_params(model):
    b = []

    b.append(model.sk2.parameters())
    b.append(model.sk3.parameters())
    b.append(model.sk4.parameters())
    b.append(model.sk5.parameters())
    b.append(model.bn2.parameters())
    b.append(model.bn3.parameters())
    b.append(model.bn4.parameters())
    b.append(model.bn5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i

def merge_params(params):
    for j in range(len(params)):
        for i in params[j]:
            yield i

def get_optimizer(optim_name, lr, parameters, weight_decay = 0, momentum = 0.9):
    if optim_name == 'sgd':
        opt = torch.optim.SGD(filter(lambda p: p.requires_grad, parameters),
                                lr=lr, weight_decay = weight_decay,
                                momentum = momentum)
    elif optim_name =='adam':
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, parameters), lr=lr, weight_decay = weight_decay)
    elif optim_name =='rmsprop':
        opt = torch.optim.RMSprop(filter(lambda p: p.requires_grad, parameters), lr=lr, weight_decay = weight_decay)
    return opt

def save_checkpoint(args, encoder, decoder, enc_opt, dec_opt):
    torch.save(encoder.state_dict(), os.path.join('../models',args.model_name,'encoder.pt'))
    torch.save(decoder.state_dict(), os.path.join('../models',args.model_name,'decoder.pt'))
    torch.save(enc_opt.state_dict(), os.path.join('../models',args.model_name,'enc_opt.pt'))
    torch.save(dec_opt.state_dict(), os.path.join('../models',args.model_name,'dec_opt.pt'))
    # save parameters for future use
    pickle.dump(args, open(os.path.join('../models',args.model_name,'args.pkl'),'wb'))
    
def save_checkpoint_prev_mask(args, encoder, decoder, enc_opt, dec_opt):
    torch.save(encoder.state_dict(), os.path.join('../models',args.model_name + '_prev_mask','encoder.pt'))
    torch.save(decoder.state_dict(), os.path.join('../models',args.model_name + '_prev_mask','decoder.pt'))
    torch.save(enc_opt.state_dict(), os.path.join('../models',args.model_name + '_prev_mask','enc_opt.pt'))
    torch.save(dec_opt.state_dict(), os.path.join('../models',args.model_name + '_prev_mask','dec_opt.pt'))
    # save parameters for future use
    pickle.dump(args, open(os.path.join('../models',args.model_name + '_prev_mask','args.pkl'),'wb'))
    
def save_checkpoint_prev_inference_mask(args, encoder, decoder, enc_opt, dec_opt):
    torch.save(encoder.state_dict(), os.path.join('../models',args.model_name + '_prev_inference_mask','encoder.pt'))
    torch.save(decoder.state_dict(), os.path.join('../models',args.model_name + '_prev_inference_mask','decoder.pt'))
    torch.save(enc_opt.state_dict(), os.path.join('../models',args.model_name + '_prev_inference_mask','enc_opt.pt'))
    torch.save(dec_opt.state_dict(), os.path.join('../models',args.model_name + '_prev_inference_mask','dec_opt.pt'))
    # save parameters for future use
    pickle.dump(args, open(os.path.join('../models',args.model_name + '_prev_inference_mask','args.pkl'),'wb'))

def load_checkpoint(model_name,use_gpu=True):
    if use_gpu:
        encoder_dict = torch.load(os.path.join('../models',model_name,'encoder.pt'))
        decoder_dict = torch.load(os.path.join('../models',model_name,'decoder.pt'))
        enc_opt_dict = torch.load(os.path.join('../models',model_name,'enc_opt.pt'))
        dec_opt_dict = torch.load(os.path.join('../models',model_name,'dec_opt.pt'))
    else:
        encoder_dict = torch.load(os.path.join('../models',model_name,'encoder.pt'), map_location=lambda storage, location: storage)
        decoder_dict = torch.load(os.path.join('../models',model_name,'decoder.pt'), map_location=lambda storage, location: storage)
        enc_opt_dict = torch.load(os.path.join('../models',model_name,'enc_opt.pt'), map_location=lambda storage, location: storage)
        dec_opt_dict = torch.load(os.path.join('../models',model_name,'dec_opt.pt'), map_location=lambda storage, location: storage)
    # save parameters for future use
    args = pickle.load(open(os.path.join('../models',model_name,'args.pkl'),'rb'))

    return encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, args

def batch_to_var(args, inputs, targets):
    """
    Turns the output of DataLoader into data and ground truth to be fed
    during training
    """
    x = Variable(inputs,requires_grad=False)
    y_mask = Variable(targets[:,:,:-1].float(),requires_grad=False)
    sw_mask = Variable(targets[:,:,-1],requires_grad=False)

    if args.use_gpu:
        return x.cuda(), y_mask.cuda(), sw_mask.cuda()
    else:
        return x, y_mask, sw_mask
        
def batch_to_var_test(args, inputs):
    """
    Turns the output of DataLoader into data and ground truth to be fed
    during training
    """
    x = Variable(inputs,requires_grad=False)

    if args.use_gpu:
        return x.cuda()
    else:
        return x

def get_skip_dims(model_name):
    if model_name == 'resnet50' or model_name == 'resnet101':
        skip_dims_in = [2048,1024,512,256,64]
    elif model_name == 'resnet34':
        skip_dims_in = [512,256,128,64,64]
    elif model_name =='vgg16':
        skip_dims_in = [512,512,256,128,64]

    return skip_dims_in

def init_visdom(args,viz):

    # initialize visdom figures

    lot = viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1,4)).cpu(),
        opts=dict(
            xlabel='Iteration',
            ylabel='Loss',
            title='Training Losses',
            legend=['iou','total']
        )
    )

    elot = {}
    # epoch losses

    elot['iou'] = viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1,2)).cpu(),
        opts=dict(
            xlabel='Epoch',
            ylabel='Loss',
            title='sIoU Loss',
            legend = ['train','val']
        )
    )

    elot['total'] = viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1,2)).cpu(),
        opts=dict(
            xlabel='Epoch',
            ylabel='Loss',
            title='Total Loss',
            legend = ['train','val']
        )
    )

    mviz_pred = {}
    for i in range(args.maxseqlen):
        mviz_pred[i] = viz.heatmap(X=np.zeros((args.imsize,args.imsize)),
                                   opts=dict(title='Pred mask t'))

    mviz_true = {}
    for i in range(args.maxseqlen):
        mviz_true[i] = viz.heatmap(X=np.zeros((args.imsize,args.imsize)),
                                   opts=dict(title='True mask t'))


    image_lot = viz.image(np.ones((3,args.imsize,args.imsize)),
                        opts=dict(title='image'))


    return lot, elot, mviz_pred, mviz_true, image_lot

def outs_perms_to_cpu(args,outs,true_perm,h,w):
    # ugly function that turns contents of torch variables to numpy
    # (used for display during training)

    out_masks = outs
    y_mask_perm = true_perm[0]

    y_mask_perm = y_mask_perm.view(y_mask_perm.size(0),y_mask_perm.size(1),h,w)
    out_masks = out_masks.view(out_masks.size(0),out_masks.size(1),h,w)
    out_masks = out_masks.view(out_masks.size(0),out_masks.size(1),h,w)


    out_masks = out_masks.cpu().numpy()
    y_mask_perm = y_mask_perm.cpu().numpy()


    return out_masks, y_mask_perm
