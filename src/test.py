import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

def test(args, encoder, decoder, x, prev_hidden_temporal_list):

    """
    Runs forward, computes loss and (if train mode) updates parameters
    for the provided batch of inputs and targets
    """

    T = args.maxseqlen
    hidden_spatial = None
    hidden_temporal_list = []

    out_masks = []

    encoder.eval()
    decoder.eval()

    feats = encoder(x)
    
    # loop over sequence length and get predictions
    for t in range(0, T):
        #prev_hidden_temporal_list is a list with the hidden state for all instances from previous time instant
        #If this is the first frame of the sequence, hidden_temporal is initialized to None. Otherwise, it is set with the value from previous time instant.
        if prev_hidden_temporal_list is not None:
            hidden_temporal = prev_hidden_temporal_list[t]
            if args.only_temporal:
                hidden_spatial = None
        else:
            hidden_temporal = None
            
        #The decoder receives two hidden state variables: hidden_spatial (a tuple, with hidden_state and cell_state) which refers to the
        #hidden state from the previous object instance from the same time instant, and hidden_temporal which refers to the hidden state from the same
        #object instance from the previous time instant.
        out_mask, hidden = decoder(feats, hidden_spatial, hidden_temporal)
        hidden_tmp = []
        for ss in range(len(hidden)):
            hidden_tmp.append(hidden[ss][0].data)
        hidden_spatial = hidden
        hidden_temporal_list.append(hidden_tmp)

        upsample_match = nn.UpsamplingBilinear2d(size=(x.size()[-2], x.size()[-1]))
        out_mask = upsample_match(out_mask)
        out_mask = out_mask.view(out_mask.size(0), -1)

        # get predictions in list to concat later
        out_masks.append(out_mask)

        del hidden_temporal, hidden_tmp, out_mask

    # concat all outputs into single tensor to compute the loss
    t = len(out_masks)
    out_masks = torch.cat(out_masks,1).view(out_masks[0].size(0),len(out_masks), -1)
    out_masks = torch.sigmoid(out_masks)
    outs = out_masks.data
    
    del feats, x
    return outs, hidden_temporal_list
    
def test_prev_mask(args, encoder, decoder, x, prev_hidden_temporal_list, prev_mask):

    """
    Runs forward, computes loss and (if train mode) updates parameters
    for the provided batch of inputs and targets
    """

    T = args.maxseqlen
    hidden_spatial = None
    hidden_temporal_list = []

    out_masks = []

    encoder.eval()
    decoder.eval()
    encoder.train(False)
    decoder.train(False)

    feats = encoder(x)
    
    # loop over sequence length and get predictions
    for t in range(0, T):
        #prev_hidden_temporal_list is a list with the hidden state for all instances from previous time instant
        #If this is the first frame of the sequence, hidden_temporal is initialized to None. Otherwise, it is set with the value from previous time instant.
        if prev_hidden_temporal_list is not None:
            hidden_temporal = prev_hidden_temporal_list[t]
            if args.only_temporal:
                hidden_spatial = None
        else:
            hidden_temporal = None

        mask_lstm = []
        maxpool = nn.MaxPool2d((2, 2),ceil_mode=True)
        prev_mask_instance = prev_mask[:,t,:]
        prev_mask_instance = prev_mask_instance.view(prev_mask_instance.size(0),1,x.data.size(2),-1)
        prev_mask_instance = maxpool(prev_mask_instance)
        for ii in range(len(feats)):
            prev_mask_instance = maxpool(prev_mask_instance)
            mask_lstm.append(prev_mask_instance)
            
        mask_lstm = list(reversed(mask_lstm))
        
        #The decoder receives two hidden state variables: hidden_spatial (a tuple, with hidden_state and cell_state) which refers to the
        #hidden state from the previous object instance from the same time instant, and hidden_temporal which refers to the hidden state from the same
        #object instance from the previous time instant.
        out_mask, hidden = decoder(feats, mask_lstm, hidden_spatial, hidden_temporal)
        hidden_tmp = []
        for ss in range(len(hidden)):
            hidden_tmp.append(hidden[ss][0].data)
        hidden_spatial = hidden
        hidden_temporal_list.append(hidden_tmp)

        upsample_match = nn.UpsamplingBilinear2d(size=(x.size()[-2], x.size()[-1]))
        out_mask = upsample_match(out_mask)
        out_mask = out_mask.view(out_mask.size(0), -1)

        # get predictions in list to concat later
        out_masks.append(out_mask)
        
        del mask_lstm, hidden_temporal, hidden_tmp, prev_mask_instance, out_mask

    # concat all outputs into single tensor to compute the loss
    t = len(out_masks)
    out_masks = torch.cat(out_masks,1).view(out_masks[0].size(0),len(out_masks), -1)

    out_masks = torch.sigmoid(out_masks)
    outs = out_masks.data

    return outs, hidden_temporal_list
