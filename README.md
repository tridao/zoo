We use the template from `https://github.com/ashleve/lightning-hydra-template`.
Please read the instructions there to understand the repo structure.

## GPT2 training
To train GPT2 on Openwebtext with 8 GPUs:
```sh
python run.py experiment=owt/gpt2s-flash trainer.devices=8
python run.py experiment=owt/gpt2m-flash trainer.devices=8
python run.py experiment=owt/gpt2l-flash trainer.devices=8
```
To train with bf16 instead of fp16, add `trainer.precision=bf16`.

## Requirements

Python 3.8+, Pytorch 1.9+, torchvision, torchtext, pytorch-fast-transformers, munch, einops, timm, hydra-core, hydra-colorlog, python-dotenv, rich, pytorch-lightning, triton.
We recommend CUDA 11.8 (e.g., using the Nvidia's Pytorch Docker image from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

We provide a Dockerfile that lists all the required packages.

This repo includes the following CUDA extensions:
1. Fused dropout + residual + LayerNorm, adapted from Apex's [FastLayerNorm](https://github.com/NVIDIA/apex/tree/master/apex/contrib/layer_norm).
```sh
cd csrc/layer_norm && pip install .
```
2. Fused matmul + bias (forward and backward), and fused matmul + bias + gelu
(forward and backward), adapted from Apex's [FusedDense](https://github.com/NVIDIA/apex/tree/master/apex/fused_dense).
```sh
cd csrc/fused_dense_lib && pip install .
```
3. Optimized cross-entropy loss, adapted from Apex's [Xentropy](https://github.com/NVIDIA/apex/tree/master/apex/contrib/xentropy).
```sh
cd csrc/xentropy && pip install .
```
