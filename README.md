# Glow

Code for reproducing results in "Glow: Generative Flow with Invertible 1x1 Convolutions"

## Requirements

 - Tensorflow (tested with v1.8.0)
 - Horovod (tested with v0.13.4) and (Open)MPI

## Download datasets
The datasets are in the Google Cloud locations `https://storage.googleapis.com/glow-demo/{dataset_name}-tfr.tar`, where dataset names are `celeba, imagenet-oord, imagenet, lsun_realnvp, lsun`.

To download and extract celeb for example, run
```
curl https://storage.googleapis.com/glow-demo/celeb-tfr.tar
tar -xvf celeb-tfr.tar
```
Change `hps.data_dir` in train.py file to point to the above folder (or use the `--data_dir` flag when you run train.py)

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
mpiexec -n 40 python train.py --problem lsun --category bedroom/church_outdoor/tower --image_size [96/128] --n_level 5 --depth 64 --flow_permutation 2 --flow_coupling 0 --seed 0 --learntop --lr 0.001 --n_bits_x 5
```

##### Conditional CIFAR-10 Qualitative result
```
mpiexec -n 8 python train.py --problem cifar10 --image_size 32 --n_level 3 --depth 32 --flow_permutation 2 --flow_coupling 0 --seed 0 --learntop --lr 0.001 --n_bits_x 5 --ycond --weight_y=0.01
```

##### Conditional ImageNet 32x32 Qualitative result
```
mpiexec -n 8 python train.py --problem imagenet --image_size 32 --n_level 3 --depth 48 --flow_permutation 2 --flow_coupling 0 --seed 0 --learntop --lr 0.001 --n_bits_x 5 --ycond --weight_y=0.01
```