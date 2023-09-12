# gMLP - TensorFlow
Gated MLP (TensorFlow implementation) from the paper [Pay Attention to MLPs](https://arxiv.org/pdf/2105.08050v2.pdf).

They propose an MLP-based alternative to Transformers without self-attention, which simply consists of channel projections and spatial projections with static parameterization.

## Roadmap
- [ ] AutoregressiveWrapper (top_p, top_k)
- [ ] Rotary Embeddings Experiment
- [ ] Gradient Checkpointing

> [!WARNING]
> This repository is under developemnt, but please feel free to explore and provide any feedback or suggestions you may have. :construction:

## Usage

```python
import tensorflow as tf
from gmlp_tensorflow import gMLPTransformer

model = gMLPTransformer(
    emb_dim = 128,        # embedding dimension
    n_tokens = 50256      # number of tokens used in the vocabulary
)

x = tf.random.uniform([1, 512], 0, 50256, 'int64')
logits = model(x, training = False)
```

## Citations

```bibtex
@article{Han2021PayAttention,
    title   = {Pay Attention to MLPs},
    author  = {Hanxiao Liu and Zihang Dai and David R. So and Quoc V. Le},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2201.08050}
}
```
