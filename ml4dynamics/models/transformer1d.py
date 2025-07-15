# 1D Transformer for KS equation, based on https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html
import jax
import jax.numpy as jnp
from flax import linen as nn
import math
import numpy as np

class PositionalEncoding(nn.Module):
    d_model: int
    max_len: int = 2048

    def setup(self):
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len, dtype=np.float32)[:, None]
        div_term = np.exp(np.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[None]
        self.pe = jax.device_put(pe)

    def __call__(self, x):
        x = x + self.pe[:, :x.shape[1]]
        return x


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -9e15, attn_logits)
    attention = nn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):
    embed_dim: int
    num_heads: int

    def setup(self):
        self.qkv_proj = nn.Dense(3 * self.embed_dim)
        self.o_proj = nn.Dense(self.embed_dim)

    def __call__(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, -1)
        qkv = qkv.transpose(0, 2, 1, 3)
        q, k, v = jnp.array_split(qkv, 3, axis=-1)
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.transpose(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)
        return o, attention

class EncoderBlock(nn.Module):
    input_dim: int
    num_heads: int
    dim_feedforward: int
    dropout_prob: float

    def setup(self):
        self.self_attn = MultiheadAttention(embed_dim=self.input_dim, num_heads=self.num_heads)
        self.linear1 = nn.Dense(self.dim_feedforward)
        self.linear2 = nn.Dense(self.input_dim)
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

    def __call__(self, x, mask=None, train=True):
        attn_out, _ = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out, deterministic=not train)
        x = self.norm1(x)
        linear_out = nn.relu(self.linear1(x))
        linear_out = self.dropout(linear_out, deterministic=not train)
        linear_out = self.linear2(linear_out)
        x = x + self.dropout(linear_out, deterministic=not train)
        x = self.norm2(x)
        return x

class TransformerEncoder(nn.Module):
    num_layers: int
    input_dim: int
    num_heads: int
    dim_feedforward: int
    dropout_prob: float

    def setup(self):
        self.layers = [EncoderBlock(self.input_dim, self.num_heads, self.dim_feedforward, self.dropout_prob) for _ in range(self.num_layers)]

    def __call__(self, x, mask=None, train=True):
        for l in self.layers:
            x = l(x, mask=mask, train=train)
        return x

class Transformer1D(nn.Module):
    input_features: int
    output_features: int
    model_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout_prob: float = 0.1
    input_dropout_prob: float = 0.0
    max_len: int = 2048

    def setup(self):
        self.input_dropout = nn.Dropout(self.input_dropout_prob)
        self.input_layer = nn.Dense(self.model_dim)
        self.positional_encoding = PositionalEncoding(self.model_dim, self.max_len)
        self.transformer = TransformerEncoder(
            num_layers=self.num_layers,
            input_dim=self.model_dim,
            num_heads=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout_prob=self.dropout_prob
        )
        self.output_layer = nn.Dense(self.output_features)

    def __call__(self, x, mask=None, train=True):
        # x: [batch, seq, input_features]
        x = self.input_dropout(x, deterministic=not train)
        x = self.input_layer(x)
        x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask, train=train)
        x = self.output_layer(x)
        return x
