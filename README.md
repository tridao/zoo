We use the template from `https://github.com/ashleve/lightning-hydra-template`.
Please read the instructions there to understand the repo structure.

To run the GPT2 experiments:
```sh
python run.py experiment=owt/gpt2s-flash
```

## Requirements

Python 3.8+, Pytorch 1.9+, torchvision, torchtext, pytorch-fast-transformers, munch, einops, timm, hydra-core, hydra-colorlog, python-dotenv, rich, pytorch-lightning, triton.
We recommend CUDA 11.8 (e.g., using the Nvidia's Pytorch Docker image from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

We provide a Dockerfile that lists all the required packages.

To install the CUDA extensions:
```sh
cd csrc/xentropy && pip install .
cd csrc/layer_norm && pip install .
cd csrc/fused_dense_lib && pip install .
cd csrc/cauchy && pip install .
```