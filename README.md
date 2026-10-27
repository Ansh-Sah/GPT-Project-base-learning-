
# GPT — A Language Model Built with NumPy

An educational decoder-only Transformer implemented from scratch
using Python and NumPy.

## Features
- Character-level tokenization
- Token and positional embeddings
- Causal multi-head self-attention
- Feed-forward network
- Residual connections and layer normalization
- Custom autograd engine and backpropagation
- Adam optimization
- Cross-entropy loss
- Text generation with temperature and top-k sampling
- Model checkpoint saving and loading

## Requirements
Python 3.10+ and NumPy.

## Train
python train.py

## Generate
python generate.py --prompt "the " --tokens 100

## Test
python test_model.py

## Disclaimer
This is a small educational GPT implementation.
It is not a pretrained model and does not have ChatGPT's capabilities.
