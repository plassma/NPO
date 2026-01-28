
import jax
import jax.numpy as jnp
import numpy as np

def unravel_dict(x):
    if (len(x.shape) >= 1):
        x = x.reshape(-1, *x.shape[2:])
    else:
        x = np.mean(x)
    return x


def get_sinusoidal_positional_encoding(timestep, embedding_dim, max_position=10000.0):
    """Create a sinusoidal positional encoding (Attention is All You Need)."""
    dtype = jnp.float32
    position = jnp.asarray(timestep, dtype=dtype)
    max_position_safe = jnp.maximum(jnp.asarray(max_position, dtype=dtype), dtype(1.0))
    div_term = jnp.exp(
        jnp.arange(0, embedding_dim, 2, dtype=dtype) * (-jnp.log(max_position_safe) / embedding_dim)
    )
    angles = position[..., None] * div_term
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)

def segment_softmax(logits, segment_ids, num_segments):
    max_per_seg = jax.ops.segment_max(logits, segment_ids, num_segments)
    shifted = logits - max_per_seg[segment_ids]
    exp = jnp.exp(shifted)
    denom = jax.ops.segment_sum(exp, segment_ids, num_segments)
    return exp / (denom[segment_ids] + 1e-9)


_VMAP_GET_SINUSOIDAL_POSITIONAL_ENCODING = jax.vmap(
    get_sinusoidal_positional_encoding, in_axes=(0, None, None)
)


def vmap_get_sinusoidal_positional_encoding(timesteps, embedding_dim, max_position=10000.0):
    return _VMAP_GET_SINUSOIDAL_POSITIONAL_ENCODING(timesteps, embedding_dim, max_position)
