# Glow

Code for reproducing results in ["Glow: Generative Flow with Invertible 1x1 Convolutions"](https://d4mucfpksywv.cloudfront.net/research-covers/glow/paper/glow.pdf)

## Requirements

 - Tensorflow (tested with v1.8.0)
 - Horovod (tested with v0.13.8) and (Open)MPI

Run
```
pip install -r requirements.txt
```

To setup (Open)MPI, check instructions on Horovod github [page](https://github.com/uber/horovod).

## Download datasets
The datasets are in the Google Cloud locations `https://storage.googleapis.com/glow-demo/data/{dataset_name}-tfr.tar`. The dataset_names are below, we mention the exact preprocessing / downsampling method for a correct comparison of likelihood.

Quantitative results
- `imagenet-oord` - Unconditional ImageNet 32x32 and 64x64, as described in PixelRNN/RealNVP papers (we downloaded [this](http://image-net.org/small/download.php) processed version).
- `lsun_realnvp` - LSUN 96x96. Random 64x64 crops taken at processing time, as described in RealNVP.

Qualitative results
- `celeba` - CelebA-HQ 1024x1024 dataset, as described in Progressive growing of GAN's.
- `imagenet` - ImageNet 32x32 and 64x64 with class labels. Centre cropped, area downsampled.
- `lsun` - LSUN 256x256. Centre cropped, area downsampled.

To download and extract celeb for example, run
```
wget https://storage.googleapis.com/glow-demo/data/celeba-tfr.tar
tar -xvf celeb-tfr.tar
```
Change `hps.data_dir` in train.py file to point to the above folder (or use the `--data_dir` flag when you run train.py)

## Simple Train with 1 GPU

Run wtih small depth to test
```
CUDA_VISIBLE_DEVICES=0 python train.py --depth 1
```

## Train with multiple GPUs using MPI and Horovod

Run default training script with 8 GPUs:
```
mpiexec -n 8 python train.py
```

##### Ablation experiments

```
mpiexec -n 8 python train.py --problem cifar10 --image_size 32 --n_level 3 --depth 32 --flow_permutation [0/1/2] --flow_coupling [0/1] --seed [0/1/2] --learntop --lr 0.001
```

##### CIFAR-10 Quantitative result

```
mpiexec -n 8 python train.py --problem cifar10 --image_size 32 --n_level 3 --depth 32 --flow_permutation 2 --flow_coupling 1 --seed 0 --learntop --lr 0.001 --n_bits_x 8
```

##### ImageNet 32x32 Quantitative result

```
mpiexec -n 8 python train.py --problem imagenet-oord --image_size 32 --n_level 3 --depth 48 --flow_permutation 2 --flow_coupling 1 --seed 0 --learntop --lr 0.001 --n_bits_x 8
```

##### ImageNet 64x64 Quantitative result
```
mpiexec -n 8 python train.py --problem imagenet-oord --image_size 64 --n_level 4 --depth 48 --flow_permutation 2 --flow_coupling 1 --seed 0 --learntop --lr 0.001 --n_bits_x 8
```

##### LSUN 64x64 Quantitative result
```
mpiexec -n 8 python train.py --problem lsun_realnvp --category [bedroom/church_outdoor/tower] --image_size 64 --n_level 3 --depth 48 --flow_permutation 2 --flow_coupling 1 --seed 0 --learntop --lr 0.001 --n_bits_x 8
```

##### CelebA-HQ 256x256 Qualitative result

```
mpiexec -n 40 python train.py --problem celeba --image_size 256 --n_level 6 --depth 32 --flow_permutation 2 --flow_coupling 0 --seed 0 --learntop --lr 0.001 --n_bits_x 5
```

##### LSUN 96x96 and 128x128 Qualitative result
```
mpiexec -n 40 python train.py --problem lsun --category [bedroom/church_outdoor/tower] --image_size [96/128] --n_level 5 --depth 64 --flow_permutation 2 --flow_coupling 0 --seed 0 --learntop --lr 0.001 --n_bits_x 5
```

##### Conditional CIFAR-10 Qualitative result
```
mpiexec -n 8 python train.py --problem cifar10 --image_size 32 --n_level 3 --depth 32 --flow_permutation 2 --flow_coupling 0 --seed 0 --learntop --lr 0.001 --n_bits_x 5 --ycond --weight_y=0.01
```

##### Conditional ImageNet 32x32 Qualitative result
```
mpiexec -n 8 python train.py --problem imagenet --image_size 32 --n_level 3 --depth 48 --flow_permutation 2 --flow_coupling 0 --seed 0 --learntop --lr 0.001 --n_bits_x 5 --ycond --weight_y=0.01
```
