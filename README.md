# Glow

Code for reproducing results in "Glow: Generative Flow with Invertible 1x1 Convolutions"

## Requirements

 - Tensorflow (tested with v1.8.0)
 - Horovod (tested with v0.13.4)

## Train with multiple GPUs using MPI and Horovod

Run default training script with 8 GPUs:
```
mpiexec -n 8 python train.py 
```


