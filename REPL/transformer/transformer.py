import haiku as hk
import jax
import jax.numpy as jnp


class SelfAttention(hk.MultiHeadAttention):
    def __call__(
        self,
        query: jax.Array,
        key: jax.Array | None = None,
        value: jax.Array | None = None,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        key = key if key is not None else query
        value = value if value is not None else query

        seq_len = query.shape[1]
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        mask = mask * causal_mask if mask is not None else causal_mask

        return super().__call__(query, key, value, mask)


class DenseBlock(hk.Module):
    """
    2-layer MLP block with residual connection, layer normalization, and GELU activation.
    """

    def __init__(
        self,
        init_scale: float,
        widening_factor: int = 4,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self._init_scale = init_scale
        self._widening_factor = widening_factor

    def __call__(
        self, x: jax.Array
    ) -> jax.Array:  # `x` is a [batch, seq_len, hiddens] array
        """
        Why `__call__`?
        In Python, `__call__` allows an instance of a class to be called like a function (e.g., `block(x)`).
        In Haiku, this is mandatory convention: `__call__` defines the "forward pass" of the module.
        When we wrap this class in `hk.transform`, Haiku specifically looks for how the instance
        is called to build the computation graph and initialize weights.

        Why does the output need the same shape as the input?
        Because the output of this dense block will be added back to the original input `x`
        (this is called a "residual connection"). To perform `x + block(x)`, their shapes must match perfectly.

        Flow:
        It receives the input `x`, applies a linear layer (expanding dimensions), applies GELU activation,
        and finally applies a second linear layer to project the dimensions back to the original size.
        """

        hiddens = x.shape[-1]
        initializer = hk.initializers.VarianceScaling(self._init_scale)
        x = hk.Linear(self._widening_factor * hiddens, w_init=initializer)(x)
        x = jax.nn.gelu(x)
        return hk.Linear(hiddens, w_init=initializer)(x)


def layer_norm(x: jax.Array, name: str | None = None) -> jax.Array:
    """
    Applies Layer Normalization to the input `x`.

    Why LayerNorm instead of BatchNorm in Transformers?
    In NLP, sequences have variable lengths and batch statistics can be unstable.
    LayerNorm normalizes across the features of EACH token independently (`axis=-1`),
    making it invariant to batch size and sequence length.

    Why `create_scale` (Gamma) and `create_offset` (Beta)?
    Forcing data to have exactly mean=0 and variance=1 can destroy useful information.
    These learnable parameters allow the neural network to undo the normalization
    partially or reshape the distribution if it helps the model learn better.
    """
    return hk.LayerNorm(
        axis=-1,
        create_scale=True,
        create_offset=True,
        name=name,
    )(x)


class TransformerBlock(hk.Module):
    """
    A single Transformer encoder block using Pre-Layer Normalization.
    """

    def __init__(
        self,
        num_heads: int,
        key_size: int,
        w_init_scale: float,
        widening_factor: int = 4,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self._num_heads = num_heads
        self._key_size = key_size
        self._w_init_scale = w_init_scale
        self._widening_factor = widening_factor

    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        # 1. Self-Attention Sublayer (with Pre-LN)
        attn_out = SelfAttention(
            num_heads=self._num_heads,
            key_size=self._key_size,
            w_init=hk.initializers.VarianceScaling(self._w_init_scale),
            name="attention",
        )(query=layer_norm(x, name="attn_ln"), mask=mask)
        x = x + attn_out

        # 2. DenseBlock Sublayer (with Pre-LN)
        dense_out = DenseBlock(
            init_scale=self._w_init_scale,
            widening_factor=self._widening_factor,
            name="mlp",
        )(layer_norm(x, name="mlp_ln"))
        x = x + dense_out

        return x
