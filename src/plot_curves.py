import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
from args import get_parser
import sys

def read_lines(txtfile):

    with open(txtfile,'r') as f:
        lines = f.readlines()
    return lines

def extract_losses(line):
    chunks = line.split('\t')

    total_loss = chunks[1].split(':')[1]
    iou_loss = chunks[2].split(':')[1]

    return total_loss, iou_loss

def plot_curves_parser(txtfile, multi = True):

    lines = read_lines(txtfile)

    if multi:
        val_losses = {'total':[],'iou':[]}
        train_losses = {'total':[],'iou':[]}
    else:
        val_loss = []
        train_loss = []
    print ("Scanning text file...")
    for line in lines:
        if '(val)' in line or '(train)' in line:

            if multi:

                total_loss, iou_loss, stop_loss = extract_losses(line)
                total_loss = float(total_loss.rstrip())
                iou_loss = float(iou_loss.rstrip())
            else:
                chunks = line.split('\t')
                loss = float(chunks[1].split('loss:')[1].rstrip())

            if '(val)' in line:
                if multi:
                    val_losses['total'].append(total_loss)
                    val_losses['iou'].append(iou_loss)
                else:
                    val_loss.append(loss)
            elif '(train)' in line:
                if multi:
                    train_losses['total'].append(total_loss)
                    train_losses['iou'].append(iou_loss)
                else:
                    train_loss.append(loss)

    print ("Done.")

    if multi:
        nb_epoch = len(val_losses['total'])
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(30,10))
    else:
        nb_epoch = len(val_loss)

    t = np.arange(0, nb_epoch, 1)

    ax1.plot(t, train_losses['total'][0:nb_epoch], 'r-*')
    ax1.plot(t, val_losses['total'], 'b-*')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.set_title('Total loss')
    ax1.legend(['train_loss','val_loss'], loc='upper right')

    ax2.plot(t, train_losses['iou'][0:nb_epoch], 'r-*')
    ax2.plot(t, val_losses['iou'], 'b-*')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.set_title('iou loss')
    ax2.legend(['train_loss','val_loss'], loc='upper right')
    
    save_file = txtfile[:-4]+'.png'
    plt.savefig(save_file)
    print ("Figure saved in %s"%(save_file))


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    model_dir = os.path.join('../models/', args.model_name)
    log_file = os.path.join(model_dir, args.log_file)

    plot_curves_parser(log_file)