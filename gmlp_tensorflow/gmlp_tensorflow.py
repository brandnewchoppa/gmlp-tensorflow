import tensorflow as tf

from keras import Model, Sequential
from keras.layers import Layer
from keras.initializers import ones

from keras.layers import (
    Dense,
    LayerNormalization,
    Embedding
)

class SpatialGatingUnit(Layer):
    """
    Spatial Gating Unit (SGU)
    https://arxiv.org/abs/2105.08050

    Splits input 'z' into two independent parts (z1, z2) along feature dimension
    to form features and gates. To enable cross-token interactions SGU performs 
    a linear projection over the spatial dimension of z2.

    The 'spatial_weights' and 'spatial_biases' of the spatial projection are
    initialized to near 0 and 1, so that the gating operation is identity at the
    start of training.

    References:
    https://github.com/sooheon/gmlp-jax
    https://github.com/lucidrains/mlp-gpt-jax
    """

    def __init__(self,
                 seq_len : int,
                 n_heads : int = 4,
                 kernel_init_epsilon : float = 1e-03,
                 layer_norm_epsilon : float = 1e-06,                 
                 **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.n_heads = n_heads
        self.kernel_init_epsilon = kernel_init_epsilon
        self.layer_norm_epsilon = layer_norm_epsilon

    def build(self, z_shape):
        eps = self.kernel_init_epsilon / z_shape[1]

        self.norm = LayerNormalization(
            name = 'spatial_norm',
            epsilon = self.layer_norm_epsilon)
        
        self.weight = self.add_weight(
            name = 'spatial_weight',
            shape = (self.n_heads, self.seq_len, self.seq_len),
            initializer = tf.random_uniform_initializer(-eps, eps))
        
        self.bias = self.add_weight(
            name = 'spatial_bias',
            shape = (1, z_shape[-1] // 2),
            initializer = ones())
        
        self.built = True

    def call(self, z, causal = False, additive_gate = None):
        n = z.shape[1]

        z1, z2 = tf.split(z, 2, axis = -1)
        z2 = self.norm(z2)

        weight, bias = self.weight, self.bias

        if causal:
            weight = weight[:, :n, :n]
            weight = tf.experimental.numpy.tril(weight)

        z2 = tf.einsum('...nd, hmn -> ...md', z2, weight)
        z2 = z2 + bias

        if additive_gate is not None:
            z2 = z2 + additive_gate
        
        return z1 * z2

class TinyAttention(Layer):
    """
    Tiny Attention (TinyAttn)
    https://arxiv.org/abs/2105.08050

    To isolate the effect of self-attention there is a tiny version attached to
    the SGU gating function. The TinyAttn doesn't have to be have so it has only
    a single head with size 64.

    Reference:
    https://github.com/sooheon/gmlp-jax
    """

    def __init__(self,
                 out_dim : int,
                 n_heads : int = 1,
                 key_dim : int = 64,
                 **kwargs):
        super().__init__(**kwargs)
        self.scale = n_heads ** -0.5
        self.to_qkv = tf.recompute_grad(Dense(key_dim * 3))
        self.to_out = tf.recompute_grad(Dense(out_dim))
    
    def call(self, x, causal = False):
        n = x.shape[-2]

        qkv = self.to_qkv(x)
        q, k, v = tf.split(qkv, 3, axis = -1)
        sim = tf.einsum('id, jd -> ij', q, k) * self.scale

        if causal:
            mask = tf.cast(tf.linalg.band_part(tf.ones((n, n)), -1, 0), tf.bool)
            sim = tf.where(mask, sim, -1e10)

        attn = tf.nn.softmax(sim)
        out = tf.einsum('ij, jd -> id', attn, v)
        return self.to_out(out)

class gMLP(Layer):
    """
    Gated MLP (gMLP)
    https://arxiv.org/abs/2105.08050

    MLP-based alternative to Transformers without self-attention, which simply
    consists of channel projections and spatial projections with static
    parameterization. The authors found that spatial projections work well when
    they are linear and paired with multiplicative gating.

    Optionally can enhance spatial gate with "tiny" attention if given.
    The output of this attention is added to gate weights in SGU before
    elementwise multiplication.

    Reference:
    https://github.com/sooheon/gmlp-jax
    """

    def __init__(self,
                 seq_len : int,
                 causal : bool = False,
                 n_heads : int = 4,
                 kernel_init_epsilon : float = 1e-03,
                 layer_norm_epsilon : float = 1e-06,
                 use_attn : bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.causal = causal
        self.n_heads = n_heads
        self.kernel_init_epsilon = kernel_init_epsilon
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_attn = use_attn

    def build(self, x_shape):
        n = x_shape[-1]

        self.proj_in = Sequential([
            LayerNormalization(epsilon = self.layer_norm_epsilon),
            Dense(n * 4, activation = 'gelu')
        ])

        self.sgu = SpatialGatingUnit(
            seq_len = self.seq_len,
            n_heads = self.n_heads,
            kernel_init_epsilon = self.kernel_init_epsilon,
            layer_norm_epsilon = self.layer_norm_epsilon)
        
        if self.use_attn:
            self.attn = TinyAttention(
                out_dim = n // 2)

        self.proj_out = tf.recompute_grad(Dense(n))

        self.built = True

    def call(self, x):
        attn_gate = self.attn(x, causal = self.causal) if self.use_attn else None
        x = self.proj_in(x)
        x = self.sgu(x, causal = self.causal, additive_gate = attn_gate)
        x = self.proj_out(x)
        return x

class StochasticDepth(Layer):
    """
    StochasticDepth
    https://paperswithcode.com/method/stochastic-depth

    Stochastic Depth aims to shrink the depth of a network during training,
    while keeping it unchanged during testing. Based on a 'p' Bernoulli random
    variable the block is active or inactive.

    References:
    https://github.com/lucidrains/mlp-gpt-jax
    """

    def __init__(self,
                 fn,
                 survival_prob : float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.fn = fn
        self.survival_prob = survival_prob

    def call(self, x, training = False, seed = None):
        if training:
            p = tf.random.uniform((1,), minval = 0, maxval = 1, seed = None)
            p = tf.math.pow(p, 1.0) * tf.math.pow(1 - p, 0)
            p = tf.cast(tf.less(p, self.survival_prob), x.dtype)
            out = self.fn(x) * p + x
            return out / self.survival_prob
        return self.fn(x) + x

class gMLPTransformer(Model):
    """
    Gated MLP (gMLP)
    https://arxiv.org/abs/2105.08050
    """

    def __init__(self,
                 n_tokens : int,
                 emb_dim : int,
                 seq_len : int = 1024,
                 causal : bool = True,
                 use_attn : bool = False,
                 n_heads : int = 4,
                 depth : int = 4,
                 kernel_init_epsilon : float = 1e-03,
                 layer_norm_epsilon : float = 1e-06,
                 survival_prob : float = 0.9,
                 drop_rate : float = 0.2,
                 **kwargs):
        super().__init__(**kwargs)
        self.embs = Embedding(n_tokens, emb_dim)

        self.blocks = Sequential([ StochasticDepth(gMLP(
            seq_len = seq_len,
            causal = causal,
            use_attn = use_attn,
            n_heads = n_heads,
            kernel_init_epsilon = kernel_init_epsilon,
            layer_norm_epsilon = layer_norm_epsilon,
            name = f'gmlp{i}'
        ), survival_prob) for i in range(depth) ])

        self.to_logits = Sequential([
            LayerNormalization(),
            Dense(n_tokens)
        ])

    def call(self, x):
        x = self.embs(x)
        x = self.blocks(x)
        return self.to_logits(x)