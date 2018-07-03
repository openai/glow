# Glow

Code for reproducing results in "Glow: Generative Flow with Invertible 1x1 Convolutions"

## Requirements

 - Tensorflow (tested with v1.8.0)
 - Horovod (tested with v0.13.4) and (Open)MPI

## Train with multiple GPUs using MPI and Horovod

Run default training script with 4 GPUs:
```
mpiexec -n 4 python train.py 
```

##### Ablation experiments

```
mpiexec -n 4 python train.py --problem cifar10 --image_size 32 --n_level 3 --depth 32 --flow_permutation [0/1/2] --flow_coupling [0/1] --seed [0/1/2] --learntop --lr 0.001
```

##### CIFAR-10 Quantiative result

```
mpiexec -n 4 python train.py --problem cifar10 --image_size 32 --n_level 3 --depth 32 --flow_permutation 2 --flow_coupling 1 --seed 0 --learntop --lr 0.001 --n_bits_x 8
```

##### ImageNet 32x32 quantitative result

```
mpiexec -n 4 python train.py --problem imagenet --data_dir=DIR --image_size 32 --n_level 3 --depth 48 --flow_permutation 2 --flow_coupling 1 --seed 0 --learntop --lr 0.001 --n_bits_x 8
```

##### ImageNet 64x64 quantitative result
```
mpiexec -n 4 python train.py --problem imagenet --data_dir=DIR --image_size 64 --n_level 4 --depth 48 --flow_permutation 2 --flow_coupling 1 --seed 0 --learntop --lr 0.001 --n_bits_x 8
```

##### ImageNet 64x64 quantitative result
```
mpiexec -n 4 python train.py --problem imagenet --data_dir=DIR --image_size 64 --n_level 4 --depth 48 --flow_permutation 2 --flow_coupling 1 --seed 0 --learntop --lr 0.001 --n_bits_x 8
```

