import jax
import jax.numpy as jnp
from jax import random
from jax import nn as jnn
import flax.linen as nn
from typing import Any
from functools import partial
from dataclasses import dataclass


def check_nan(x, name):
    def raise_error(_):
        raise ValueError(f"NaN found in {name}")
    
    def f(_):
        pass
    
    jax.lax.cond(jnp.isnan(x).any(), raise_error, f, operand=None)
    
@dataclass(frozen=True)
class RWKVConfig:
    vocab_size: int
    n_layer: int
    n_embd: int
    dim_att: int
    dim_ffn: int
    head_size_a: int
    n_head: int
    head_size_divisor: int
    dropout: float
    layer_norm_epsilon: float
    chunk_size: int
    subchunk_size: int
    min_clamp: int

class GroupNorm(nn.Module):
    num_groups: int
    epsilon: float = 1e-5
    use_bias: bool = True
    use_scale: bool = True

    @nn.compact
    def __call__(self, x):
        orig_shape = x.shape
        x = x.reshape(-1, self.num_groups, x.shape[-1] // self.num_groups)
        mean = jnp.mean(x, axis=(0, 2), keepdims=True)
        var = jnp.var(x, axis=(0, 2), keepdims=True)
        check_nan(var, 'GroupNorm var')
        x = ((x - mean) / jnp.sqrt(var + self.epsilon)).reshape(orig_shape)
        check_nan(x, 'GroupNorm x normalized')
        if self.use_scale:
            x *= self.param('scale', nn.initializers.ones, (x.shape[-1],))
        if self.use_bias:
            x += self.param('bias', nn.initializers.zeros, (x.shape[-1],))
        return x

class RWKVBlock(nn.Module):
    config: RWKVConfig
    layer_id: int

    def setup(self):
        args = self.config
        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        self.min_clamp = args.min_clamp
        assert args.dim_att % self.n_head == 0

        ratio_0_to_1 = self.layer_id / (args.n_layer - 1)
        ratio_1_to_almost0 = 1.0 - (self.layer_id / args.n_layer)

        def init_time_maa(name, ratio):
            def init(key, shape):
                return 1.0 - jnp.power((jnp.arange(args.n_embd) / args.n_embd), ratio).reshape(1, 1, -1)
            return self.param(name, init, (1, 1, args.n_embd))

        self.time_maa_x = init_time_maa('time_maa_x', ratio_1_to_almost0)
        self.time_maa_w = init_time_maa('time_maa_w', ratio_1_to_almost0)
        self.time_maa_k = init_time_maa('time_maa_k', ratio_1_to_almost0)
        self.time_maa_v = init_time_maa('time_maa_v', ratio_1_to_almost0 + 0.3 * ratio_0_to_1)
        self.time_maa_r = init_time_maa('time_maa_r', 0.5 * ratio_1_to_almost0)
        self.time_maa_g = init_time_maa('time_maa_g', 0.5 * ratio_1_to_almost0)
        def init_time_decay(key, shape):
            return (-6 + 5 * (jnp.arange(args.dim_att) / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1))
        self.time_decay = self.param('time_decay', init_time_decay, (1, args.dim_att))

        def init_time_faaaa(key, shape):
            n = jnp.arange(args.dim_att)
            return (ratio_0_to_1 * (1 - (n / max(args.dim_att - 1, 1))) + ((n + 1) % 3 - 1) * 0.1).reshape((self.n_head, -1))
        self.time_faaaa = self.param('time_faaaa', init_time_faaaa, (self.n_head, self.head_size))

        self.time_maa_w1 = self.param('time_maa_w1', nn.initializers.uniform(1e-4), (args.n_embd, 5*32))
        self.time_maa_w2 = self.param('time_maa_w2', nn.initializers.uniform(1e-4), (5, 32, args.n_embd))
        self.time_decay_w1 = self.param('time_decay_w1', nn.initializers.uniform(1e-4), (args.n_embd, 64))
        self.time_decay_w2 = self.param('time_decay_w2', nn.initializers.uniform(1e-4), (64, args.n_embd))

        self.receptance = nn.Dense(args.dim_att, use_bias=False)
        self.key = nn.Dense(args.dim_att, use_bias=False)
        self.value = nn.Dense(args.dim_att, use_bias=False)
        self.output = nn.Dense(args.n_embd, use_bias=False)
        self.gate = nn.Dense(args.dim_att, use_bias=False)
        self.ln_x = GroupNorm(num_groups=self.n_head)

        self.time_maa_k_channel = init_time_maa('time_maa_k_channel', ratio_1_to_almost0)
        self.time_maa_r_channel = init_time_maa('time_maa_r_channel', ratio_1_to_almost0)

        self.key_channel = nn.Dense(args.dim_ffn, use_bias=False)
        self.receptance_channel = nn.Dense(args.n_embd, use_bias=False)
        self.value_channel = nn.Dense(args.n_embd, use_bias=False)

        self.ln1 = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon)

    def time_shift(self, x):
        shifted = jnp.pad(x[:, :-1], ((0, 0), (1, 0), (0, 0)))
        check_nan(shifted, 'time_shift')
        return shifted

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0, 0))
    def chunkwise(self, k, v, r, w, u, state):
        C = self.config.chunk_size
        L, D = r.shape
        
        pad_size = (C - L % C) % C
        pad_fn = lambda x: jnp.pad(x, ((0, pad_size), (0, 0)))
        r, k, v, w, u = map(pad_fn, [r, k, v, w, u])
        
        L_padded = L + pad_size
        r, k, v, w, u = map(lambda x: x.reshape((L_padded // C, C, -1)), [r, k, v, w, u])
        
        w_min = jnp.float32(10**(-70 / C))  
        w = jax.lax.clamp(self.min_clamp, w, 1.0)
        w = jnp.log(w)
        check_nan(w, 'chunkwise w')
        
        A = jnp.exp(jnp.cumsum(w, axis=1))
        check_nan(A, 'chunkwise A')
        A_inter = jnp.exp(jax.lax.cumsum(w, axis=1, reverse=True) - w)
        A_intra = jnp.cumsum(w, axis=1)
        kv = jnp.einsum('nij,nik->nijk', A_inter * k, v)

        def scan_fn(carry, x):
            s = carry
            A_t, kv_t = x
            new_s = A_t * s + kv_t
            return new_s, s

        init_carry = state
        new_state, S = jax.lax.scan(scan_fn, init_carry, (A.reshape(L_padded, D), kv.reshape(L_padded, D, D)))

        S = S.reshape(L_padded // C, C, D, D)
        
        inter_chunk = jnp.einsum('nij,nijk->nik', r * jnp.exp(A_intra - w), S)

        def intra_chunk_fn(r_chunk, k_chunk, v_chunk, A_chunk, u_chunk):
            mid = C // 2
            r_decay = r_chunk * jnp.exp(jnp.pad(A_chunk[:-1, :], ((1, 0), (0, 0))) - A_chunk[mid:mid+1, :])
            k_decay = k_chunk * jnp.exp(A_chunk[mid:mid+1, :] - A_chunk)
            u_diag = jnp.diag(jnp.sum(r_chunk * u_chunk * k_chunk, -1))
            return (jnp.tril(r_decay @ k_decay.T, -1) + u_diag) @ v_chunk

        intra_chunk = jax.vmap(intra_chunk_fn)(r, k, v, A_intra, u)
        
        result = (inter_chunk + intra_chunk).reshape((L_padded, -1))
        return result[:L], new_state


    def time_mixing(self, x, state):
        B, T, C = x.shape
        H, S = self.n_head, self.head_size

        xx = self.time_shift(x)
        sx = xx - x

        xx = x + sx * self.time_maa_x
        xx = jnn.tanh(xx @ self.time_maa_w1).reshape((B, T, 5, -1)).swapaxes(1, 2)
        mw, mk, mv, mr, mg = (xx @ self.time_maa_w2).reshape((5, B, T, -1))

        xw = x + sx * (self.time_maa_w + mw)
        xk = x + sx * (self.time_maa_k + mk)
        xv = x + sx * (self.time_maa_v + mv)
        xr = x + sx * (self.time_maa_r + mr)
        xg = x + sx * (self.time_maa_g + mg)

        r = self.receptance(xr).reshape(B, T, H, S)
        check_nan(r, 'time_mixing r')
        k = self.key(xk).reshape(B, T, H, S)
        check_nan(k, 'time_mixing k')
        v = self.value(xv).reshape(B, T, H, S)
        check_nan(v, 'time_mixing v')
        g = jnn.silu(self.gate(xg))
        check_nan(v, 'time_mixing g')

        time_decay = self.time_decay.reshape(1, 1, H, S)
        time_decay_offset = jnn.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        time_decay_offset = time_decay_offset.reshape(B, T, H, S)

        # w = jnp.exp(-jnp.exp(time_decay + time_decay_offset)+ 1e-6)
        w = jax.nn.softplus(-(time_decay + time_decay_offset))

        u = jnp.broadcast_to(self.time_faaaa, (B, T, H, S))

        state = state.reshape(B * H, S, S)

        k = k.transpose(0, 2, 1, 3).reshape(B * H, T, S)
        v = v.transpose(0, 2, 1, 3).reshape(B * H, T, S)
        r = r.transpose(0, 2, 1, 3).reshape(B * H, T, S)
        w = w.transpose(0, 2, 1, 3).reshape(B * H, T, S)
        u = u.transpose(0, 2, 1, 3).reshape(B * H, T, S)

        y, new_state = self.chunkwise(k, v, r, w, u, state)

        y = y.reshape(B, H, T, S).transpose(0, 2, 1, 3).reshape(B, T, H * S)
        new_state = new_state.reshape(B, H, S, S)

        y = self.ln_x(y)
        y = self.output(y * g)

        return y, new_state

    def channel_mixing(self, x):
        xx = self.time_shift(x)
        sx = xx - x
        xk = x + sx * self.time_maa_k_channel
        xr = x + sx * self.time_maa_r_channel

        k = jax.nn.relu(self.key_channel(xk)) ** 2
        check_nan(k, 'channel_mixing k')
        kv = self.value_channel(k)
        check_nan(kv, 'channel_mixing kv')
        return jax.nn.sigmoid(self.receptance_channel(xr)) * kv

    def __call__(self, x, state, deterministic=True):
        check_nan(x, 'call x')
        x_attn, new_state = self.time_mixing(self.ln1(x), state)
        x = x + x_attn
        x = x + self.channel_mixing(self.ln2(x))

        if not deterministic:
            x = nn.Dropout(rate=self.config.dropout, deterministic=deterministic)(x)

        return x, new_state

class RWKV(nn.Module):
    config: RWKVConfig

    @nn.compact
    def __call__(self, idx, state, deterministic=False):
        assert jnp.all((idx >= 0) & (idx < self.config.vocab_size)), "Input indices out of range"
        x = nn.Embed(num_embeddings=self.config.vocab_size, features=self.config.n_embd,embedding_init=nn.initializers.normal(stddev=0.01))(idx)
        check_nan(x, 'Embed x')

        new_states = []
        for i in range(self.config.n_layer):
            block = RWKVBlock(self.config, i)
            check_nan(x, 'in loop x')
            x, new_state = block(x, state[:, i], deterministic=deterministic)
            new_states.append(new_state)

        x = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon)(x)
        x = nn.Dense(self.config.vocab_size, use_bias=False)(x)

        new_states = jnp.stack(new_states, axis=1)
        return x, new_states

    @classmethod
    def get_init_state(cls, config, batch_size):
        return jnp.zeros((batch_size, config.n_layer, config.n_head, config.head_size_a, config.head_size_a))

def create_model(config):
    model = RWKV(config)
    key = random.PRNGKey(0)
    dummy_input = jnp.zeros((1, 16), dtype=jnp.int32)

    @jax.jit
    def init_fn(key):
        dummy_state = RWKV.get_init_state(config, 1)
        return model.init(key, dummy_input, dummy_state)

    params = init_fn(key)
    return model, params

@partial(jax.jit, static_argnums=(0,))
def model_forward(model, params, idx, state, deterministic=False):
    return model.apply(params, idx, state, deterministic=deterministic, rngs={'dropout': random.PRNGKey(0)})


if __name__ == "__main__":
    config = RWKVConfig(
        vocab_size=10000,
        n_layer=12,
        n_embd=768,
        dim_att=768,
        dim_ffn=3072,
        head_size_a=64,
        n_head=12,
        head_size_divisor=8,
        dropout=0.1,
        layer_norm_epsilon=1e-5,
        chunk_size=64, 
        subchunk_size=32,
        min_clamp = 0.01
    )

    model, params = create_model(config)

    batch_size = 4
    seq_length = 64
    dummy_input = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
    dummy_state = RWKV.get_init_state(config, batch_size)

    output, new_state = model_forward(model, params, dummy_input, dummy_state)

    print(f"Output shape: {output.shape}")
    print(f"New state shape: {new_state.shape}")
