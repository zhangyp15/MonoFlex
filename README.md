# MonoFlex
Released code for Objects are Different: Flexible Monocular 3D Object Detection, CVPR21

The README is still in progress :/

## Installation
This repo is tested with Ubuntu 20.04, python=3.7, pytorch=1.4.0, cuda=10.1

```bash
conda create -n monoflex python=3.7
conda activate monoflex
```
Install PyTorch and other dependencies:

```bash
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt
```
Build DCNv2 and the project
```bash
cd models/backbone/DCNv2

. make.sh

cd ../../..

python setup develop
```

## Data Preparation
Please download [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and organize the data as follows:

```
#ROOT
  |data/
    |KITTI/
      |ImageSets/ [already provided in this repo]
      |object/			
        |training/
          |calib/
          |image_2/
          |label/
        |testing/
          |calib/
          |image_2/
```

### Training & Evaluation

Move to the workplace and train the network:

```sh
CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --batch_size 8 --config runs/monoflex.yaml --output output/exp
```

The model will be evaluated every two epochs during training and you can also evaluate a checkpoint with
```s
CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --config runs/monoflex.yaml --ckpt YOUR_CKPT  --eval
```


## Acknowlegment

The code is heavily borrowed from SMOKE and thanks their contribution.
