# RVOS: End-to-End Recurrent Net for Video Object Segmentation

See our project website [here](https://imatge-upc.github.io/rvos/).

In order to develop this code, we used RSIS (Recurrent Semantic Instance Segmentation), which can be found [here](https://github.com/imatge-upc/rsis), and modified it to suit it to video object segmentation task.

One shot visual results

![RVOS One shot](https://github.com/imatge-upc/rvos/blob/master/rvos_supp_oneshot.gif) 

Zero shot visual results

![RVOS Zero shot](https://github.com/imatge-upc/rvos/blob/master/rvos_supp_zeroshot.gif)

## License

This code cannot be used for commercial purposes. Please contact the authors if interested in licensing this software.

## Installation
- Clone the repo:

```shell
git clone https://github.com/imatge-upc/rvos.git
```

- Install requirements ```pip install -r requirements.txt``` 
- Install [PyTorch 1.0](http://pytorch.org/) (choose the whl file according to your setup, e.g. your CUDA version):

```shell
pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision
```

## Data

### YouTube-VOS

Download the YouTube-VOS dataset from their [website](https://youtube-vos.org/). You will need to register to codalab to download the dataset. Create a folder named ```databases```in the parent folder of the root directory of this project and put there the database in a folder named ```YouTubeVOS```. The root directory (```rvos```folder) and the ```databases``` folder should be in the same directory.

The training of the RVOS model for YouTube-VOS has been implemented using a split of the train set into two subsets: train-train and train-val. The model is trained on the train-train subset and validated on the train-val subset to decide whether the model should be saved or not. To train the model according to this split, the code requires that there are two json files in the ```databases/YouTubeVOS/train/```folder named ```train-train-meta.json```and ```train-val-meta.json``` with the same format as the ```meta.json```included when downloading the dataset. You can also download the partition used in our experiments in the following links:

- [YouTube-VOS train-train-meta.json](https://imatge.upc.edu/web/sites/default/files/projects/segmentation/public_html/rvos-pretrained-models/train-train-meta.json)
- [YouTube-VOS train-val-meta.json](https://imatge.upc.edu/web/sites/default/files/projects/segmentation/public_html/rvos-pretrained-models/train-val-meta.json)

### DAVIS 2017

Download the DAVIS 2017 dataset from their [website](https://davischallenge.org/davis2017/code.html) at 480p resolution. Create a folder named ```databases```in the parent folder of the root directory of this project and put there the database in a folder named ```DAVIS2017```. The root directory (```rvos```folder) and the ```databases``` folder should be in the same directory.

### LMDB data indexing

To highly speed the data loading we recommend to generate an LMDB indexing of it by doing:
```
python dataset_lmdb_generator.py -dataset=youtube
```
or
```
python dataset_lmdb_generator.py -dataset=davis2017
```
depending on the dataset you are using.

## Training

- Train the model for one-shot video object segmentation with ```python train_previous_mask.py -model_name model_name```. Checkpoints and logs will be saved under ```../models/model_name```.
- Train the model for zero-shot video object segmentation with ```python train.py -model_name model_name```. Checkpoints and logs will be saved under ```../models/model_name```. 
- Other arguments can be passed as well. For convenience, scripts to train with typical parameters are provided under ```scripts/```.
- Plot loss curves at any time with ```python plot_curves.py -model_name model_name```.

## Evaluation

We provide bash scripts to  evaluate models for the YouTube-VOS and DAVIS 2017 datasets. You can find them under the ```scripts``` folder. On the one hand, ```eval_one_shot_youtube.sh```and ```eval_zero_shot_youtube.sh``` generate the results for YouTube-VOS dataset on one-shot video object segmentation and zero-shot video object segmentation respectively. On the other hand, ```eval_one_shot_davis.sh```and ```eval_zero_shot_davis.sh``` generate the results for DAVIS 2017 dataset on one-shot video object segmentation and zero-shot video object segmentation respectively. 

Furthermore, in the ```src``` folder, ```prepare_results_submission.py```and ```prepare_results_submission_davis``` can be applied to change the format of the results in the appropiate format to use the official evaluation servers of [YouTube-VOS](https://competitions.codalab.org/competitions/19544) and [DAVIS](https://competitions.codalab.org/competitions/16526) respectively.

## Demo

You can run ```demo.py``` to do generate the segmentation masks of a video. Just do:
```
python demo.py -model_name one-shot-model-davis --overlay_masks
```
and it will generate the resulting masks.

To run the demo for your own videos:
1. extract the frames to a folder (make sure their names are in order, e.g. 00000.jpg, 00001.jpg, ...) 
2. Have the initial mask corresponding to the first frame (e.g. 00000.png).
3. run 
  ```python demo.py -model_name one-shot-model-davis -frames_path path-to-your-frames -mask_path path-to-initial-mask --overlay_masks```

to do it for zero-shot (i.e. without initial mask) run
  ```python demo.py -model_name zero-shot-model-davis -frames_path path-to-your-frames --zero_shot --overlay_masks```
  
Also you can use the argument `-results_path` to save the results to the folder you prefer.


## Pretrained models

Download weights for models trained with:

- [YouTube-VOS (one-shot)](https://imatge.upc.edu/web/sites/default/files/projects/segmentation/public_html/rvos-pretrained-models/one-shot-model-youtubevos.zip)
- [YouTube-VOS (zero-shot)](https://imatge.upc.edu/web/sites/default/files/projects/segmentation/public_html/rvos-pretrained-models/zero-shot-model-youtubevos.zip)
- [DAVIS 2017 (one-shot)](https://imatge.upc.edu/web/sites/default/files/projects/segmentation/public_html/rvos-pretrained-models/one-shot-model-davis.zip)
- [DAVIS 2017 (zero-shot)](https://imatge.upc.edu/web/sites/default/files/projects/segmentation/public_html/rvos-pretrained-models/zero-shot-model-davis.zip)

Extract and place the obtained folder under ```models``` directory. 
You can then run evaluation scripts with the downloaded model by setting ```args.model_name``` to the name of the folder.

## Contact

For questions and suggestions use the issues section or send an e-mail to cventuraroy@uoc.edu
