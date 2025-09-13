
from functools import partial
import json
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.linen.attention import dot_product_attention
from flax.training import train_state, checkpoints
from typing import Any
import time
import matplotlib.pyplot as plt
from datetime import datetime
import os
import csv
from collections import Counter
from jax import debug as jdbg

DEBUG = True or bool(int(os.getenv("DEBUG", "0")))

from jax import debug as jdbg

def print_(*args, **kwargs):
    if DEBUG:
        jdbg.print(*args, **kwargs)

def assert_(cond, msg):
    if DEBUG:
        assert cond, msg

# ------------------
# 0. CONFIG (HF-style JSON-serializable)
# ------------------
# Minimal starter config; will expand as we add attention/blocks
TRANSFORMER_CONFIG = {
    "vocab_size": None,           # set after BPE is built
    "max_seq_len": 512,
    "d_model": 256,
    "num_heads": 4,
    "num_layers": 2,
    "d_ff": 1024,
    "dropout_rate": 0.1,
    # Sparse attention (Claude-style)
    "local_window": 128,          # tokens each query can see locally (causal)
    "global_stride": 64,          # every Nth token is global
    "attention_dropout_rate": 0.1,
    # Flash attention path (uses flax's fused dot_product_attention when available)
    "use_flash_attention": True,
}

def test_encode_decode_roundtrip():
    s = "hello world"
    arr = encode(s)
    out = decode(arr)
    assert s in out, f"Roundtrip failed: {s} -> {out}"

def test_model_forward_shapes():
    model = TransformerLM(vocab_size=vocab_size, d_model=TRANSFORMER_CONFIG["d_model"], dropout_rate=TRANSFORMER_CONFIG["dropout_rate"])
    x = jnp.ones((2, seq_len), dtype=jnp.int32)
    variables = model.init({'params': jax.random.PRNGKey(0)}, x)
    logits = model.apply(variables, x, train=False)
    assert logits.shape == (2, seq_len, vocab_size)

# ------------------
# 1. DATASET + BPE TOKENIZER
# ------------------
with open("./train/train.txt", "r", encoding="utf-8") as f:
    text = f.read()

if DEBUG:
    # Run a quick sanity check for sparse attention mask on a small T.
    try:
        _ = test_sparse_mask_small()
        print_("Sparse mask sanity test passed.")
    except Exception as e:
        print_(f"Sparse mask sanity test failed: {e}")

def test_attention_equivalence_small():
    """Verify DPA (flash) and manual attention produce near-identical outputs on a tiny input.
    We keep train=False to disable any dropout effects.
    """
    # Tiny model
    vocab = 32
    T = 8
    model = TransformerLM(vocab_size=vocab, d_model=64, dropout_rate=0.0)
    x = jnp.arange(T)[None, :].astype(jnp.int32) % vocab
    params = model.init({'params': jax.random.PRNGKey(0)}, x)['params']
    # Toggle config
    use_flash_backup = TRANSFORMER_CONFIG.get("use_flash_attention", True)
    try:
        TRANSFORMER_CONFIG["use_flash_attention"] = True
        y_flash = model.apply({'params': params}, x, train=False)
        TRANSFORMER_CONFIG["use_flash_attention"] = False
        y_manual = model.apply({'params': params}, x, train=False)
        close = jnp.allclose(y_flash, y_manual, atol=1e-5, rtol=1e-5)
        assert_(bool(close), f"Flash vs Manual attention mismatch: max diff={float(jnp.max(jnp.abs(y_flash - y_manual)))}")
        return True
    finally:
        TRANSFORMER_CONFIG["use_flash_attention"] = use_flash_backup

if DEBUG:
    try:
        _ = test_attention_equivalence_small()
        print_("Attention path equivalence test passed.")
    except Exception as e:
        print_(f"Attention path equivalence test failed: {e}")

# --- Byte Pair Encoding (BPE) implementation ---
def build_bpe_vocab(corpus, num_merges=50):
    vocab = [list(word) + ["</w>"] for word in corpus.split(" ")]
    space_token = "<space>"
    vocab_with_spaces = []
    for i, word in enumerate(vocab):
        vocab_with_spaces.append(word)
        if i < len(vocab) - 1:
            vocab_with_spaces.append([space_token])

    vocab_counts = Counter([" ".join(word) for word in vocab_with_spaces])

    def get_stats(vocab_counts):
        pairs = Counter()
        for word, freq in vocab_counts.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def merge_vocab(pair, vocab_counts):
        bigram = " ".join(pair)
        replacement = "".join(pair)
        new_vocab = {}
        for word, freq in vocab_counts.items():
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = freq
        return new_vocab

    merges = []
    for _ in range(num_merges):
        pairs = get_stats(vocab_counts)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        if space_token in best:
            continue
        vocab_counts = merge_vocab(best, vocab_counts)
        merges.append(best)
    return merges, vocab_counts, space_token

# Build BPE merges
merges, final_vocab, space_token = build_bpe_vocab(text, num_merges=50)

# Create stoi/itos mapping dynamically after merges
subwords = set()
for word in final_vocab:
    subwords.update(word.split())
vocab = sorted(subwords)

# Ensure consistency across runs: rebuild embedding params if vocab size changes
stoi = {tok: i for i, tok in enumerate(vocab)}
itos = {i: tok for tok, i in stoi.items()}
vocab_size = len(vocab)

assert_(vocab_size > 0, "Vocabulary size must be positive.")

# --- Encoding / Decoding ---
def encode(s):
    words = s.split(" ")
    symbols = []
    for i, word in enumerate(words):
        w = list(word) + ["</w>"]
        symbols.extend(w)
        if i < len(words) - 1:
            symbols.append(space_token)

    for merge in merges:
        i = 0
        while i < len(symbols) - 1:
            if (symbols[i], symbols[i+1]) == merge:
                symbols[i:i+2] = ["".join(merge)]
            else:
                i += 1
    arr = np.array([stoi[w] for w in symbols if w in stoi], dtype=np.int32)
    assert_(arr .ndim == 1, "Encoded sequence must be 1D")
    return np.array([stoi[w] for w in symbols if w in stoi], dtype=np.int32)

def decode(arr):
    tokens = [itos[int(i)] for i in arr]
    text = "".join(tokens)
    text = text.replace("</w>", "")
    text = text.replace(space_token, " ")
    return text

# Cached autoregressive generation (faster)
def generate_cached(state, start="To ", steps: int = 200, temperature: float = 1.0, rng_key: jax.Array | None = None):
    idxs = encode(start)
    assert_(idxs.ndim == 1, f"Encoded start must be 1D, got {idxs.shape}")
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    # Warmup with the prompt
    x = jnp.array([idxs], dtype=jnp.int32)  # (1, T0)
    init_cache = [{} for _ in range(TRANSFORMER_CONFIG["num_layers"])]
    # model returns (logits, cache) when cache is provided
    logits, cache = state.apply_fn({'params': state.params}, x, train=False, cache=init_cache)
    out_ids = [int(i) for i in np.array(idxs)]
    for _ in range(steps):
        # Sample next id from last logits
        last_logits = logits[0, -1] / temperature
        rng_key, subkey = jax.random.split(rng_key)
        next_id = int(jax.random.categorical(subkey, last_logits))
        out_ids.append(next_id)
        # Feed the new token and update cache
        x_step = jnp.array([[next_id]], dtype=jnp.int32)
        logits, cache = state.apply_fn({'params': state.params}, x_step, train=False, cache=cache)
    # Decode ids to text
    out_tokens = [itos[i] for i in out_ids]
    text = "".join(out_tokens)
    text = text.replace("</w>", "").replace(space_token, " ")
    return text

# Explicitly fix dtype for dataset
data = jnp.array(encode(text), dtype=jnp.int32)
seq_len = 32
batch_size = 32
num_steps = 200
num_epochs = 10

# Gradient accumulation config
ACCUM_STEPS = 4  # must divide batch_size
assert_(batch_size % ACCUM_STEPS == 0, "ACCUM_STEPS must divide batch_size")
MICRO_BSZ = batch_size // ACCUM_STEPS

# For consistent GRU input dims: embed_dim must match GRU input size
def precompute_batches(data, seq_len, batch_size, val_split=0.1):
    split = int(len(data) * (1 - val_split))
    train_data, val_data = data[:split], data[split:]

    def make_batches(dataset):
        n_steps = (len(dataset) - seq_len) // batch_size
        xs, ys = [], []
        for i in range(n_steps * batch_size):
            start = i
            end = start + seq_len
            xs.append(dataset[start:end])
            ys.append(dataset[start+1:end+1])
        xs = np.array(xs).reshape(n_steps, batch_size, seq_len)
        ys = np.array(ys).reshape(n_steps, batch_size, seq_len)
        assert_(xs.shape[1:] == (batch_size, seq_len), f"Unexpected xs shape {xs.shape}")
        return xs, ys

    train_x, train_y = make_batches(train_data)
    val_x, val_y = make_batches(val_data)
    return train_x, train_y, val_x, val_y

train_x, train_y, val_x, val_y = precompute_batches(np.array(data), seq_len, batch_size)
assert_(train_x.ndim == 3, "train_x must be 3D (steps, batch, seq_len)")

# ------------------
# 2. MODEL (Transformer + RoPE)
# ------------------

# ---- RoPE helpers ----
def _rope_get_cos_sin(seq_len: int, head_dim: int, base: float = 10000.0):
    """Compute RoPE cos/sin tables.
    Returns cos, sin with shape (seq_len, head_dim).
    """
    assert head_dim % 2 == 0, "RoPE head_dim must be even"
    half = head_dim // 2
    positions = jnp.arange(seq_len)[:, None]  # (T, 1)
    inv_freq = 1.0 / (base ** (jnp.arange(0, half) / half))  # (half,)
    angles = positions * inv_freq[None, :]  # (T, half)
    cos = jnp.repeat(jnp.cos(angles), 2, axis=1)  # (T, head_dim)
    sin = jnp.repeat(jnp.sin(angles), 2, axis=1)  # (T, head_dim)
    return cos, sin

def _rope_apply(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """Apply rotary embedding to last dimension of x.
    x: (..., T, D), cos/sin: (T, D)
    """
    assert x.ndim >= 3, f"RoPE expects (..., T, D), got {x.shape}"
    T = x.shape[-2]
    D = x.shape[-1]
    assert_(cos.shape == (T, D) and sin.shape == (T, D), f"cos/sin bad shapes {cos.shape} {sin.shape}")
    x1 = x[..., :, 0::2]
    x2 = x[..., :, 1::2]
    # Interleave for rotation: (x_even, x_odd)
    x_even = x1
    x_odd = x2
    cos_e = cos[:, 0::2]
    sin_e = sin[:, 0::2]
    # Broadcast to match x: (..., T, half)
    while cos_e.ndim < x_even.ndim:
        cos_e = cos_e[None, ...]
        sin_e = sin_e[None, ...]
    x_even_rot = x_even * cos_e - x_odd * sin_e
    x_odd_rot = x_even * sin_e + x_odd * cos_e
    # Reinterleave
    x_rot = jnp.empty_like(x)
    x_rot = x_rot.at[..., :, 0::2].set(x_even_rot)
    x_rot = x_rot.at[..., :, 1::2].set(x_odd_rot)
    return x_rot

# ---- Sparse mask debug helpers (educational) ----
def build_sparse_allowed_mask(T: int, local_window: int, global_stride: int) -> jnp.ndarray:
    """Return (T, T) boolean mask of allowed positions under our sparse pattern.
    True means attention from query i to key j is allowed.
    """
    lw = min(local_window, T)
    gs = max(1, global_stride)
    idx = jnp.arange(T)
    rel = idx[None, :] - idx[:, None]  # j - i
    local_ok = (rel <= 0) & (rel >= -(lw - 1))
    is_global = (idx % gs) == 0
    key_global = jnp.tile(is_global[None, :], (T, 1))
    query_global = jnp.tile(is_global[:, None], (1, T))
    past_ok = rel <= 0
    allowed = local_ok | key_global | (query_global & past_ok)
    # enforce causal strictly
    base_causal = jnp.triu(jnp.ones((T, T), dtype=bool), k=1)
    allowed = allowed & (~base_causal)
    return allowed

def test_sparse_mask_small():
    T, lw, gs = 16, TRANSFORMER_CONFIG["local_window"], TRANSFORMER_CONFIG["global_stride"]
    mask = build_sparse_allowed_mask(T, lw, gs)
    # diagonal should be True
    assert_(bool(jnp.all(jnp.diag(mask))), "Diagonal (self-attend) must be allowed")
    # strictly future positions should be False
    assert_(bool(jnp.all(jnp.triu(~mask, k=1))), "Future positions must be masked")
    # global keys (0, gs, 2gs, ...) should be visible to all queries
    globals_idx = jnp.arange(0, T, gs)
    assert_(bool(jnp.all(mask[:, globals_idx])), "Global keys must be visible to all queries")
    return mask

class MultiHeadSelfAttention(nn.Module):
    d_model: int
    num_heads: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True, causal: bool = True, cache: dict | None = None) -> tuple[jnp.ndarray, dict | None]:
        assert_(x.ndim == 3, f"MHA expects (B, T, C), got {x.shape}")
        B, T, C = x.shape
        assert_(C == self.d_model, f"Channel mismatch {C} != {self.d_model}")
        assert_(self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads")
        d_head = self.d_model // self.num_heads

        qkv = nn.Dense(3 * self.d_model, use_bias=False, dtype=jnp.float32)(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, d_head)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q = q.squeeze(2)  # (B, T, H, Dh)
        k = k.squeeze(2)
        v = v.squeeze(2)

        # Move heads forward for attention compute: (B, H, T, Dh)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # If KV cache is provided, append new K/V and compute with past keys
        use_cache = cache is not None
        if use_cache:
            # cache structure per layer: {"k": (B,H,Tpast,Dh), "v": (B,H,Tpast,Dh)}
            k_past = cache.get("k", None)
            v_past = cache.get("v", None)
            past_T = 0 if k_past is None else k_past.shape[2]
            # Apply RoPE with offset: build cos/sin up to past+T and slice last T for q, and last (past+T) for k if recomputing
            cos_all, sin_all = _rope_get_cos_sin(past_T + T, d_head)
            # slice for current positions
            cos_cur = cos_all[past_T: past_T + T]
            sin_cur = sin_all[past_T: past_T + T]
            q = _rope_apply(q, cos_cur, sin_cur)
            # for keys, we only need current slice; past is already rotated if stored rotated.
            k = _rope_apply(k, cos_cur, sin_cur)
            # concat to cache
            k_new = k if k_past is None else jnp.concatenate([k_past, k], axis=2)
            v_new = v if v_past is None else jnp.concatenate([v_past, v], axis=2)
            cache = {"k": k_new, "v": v_new}
            k_full = cache["k"]
            v_full = cache["v"]
            T_k = k_full.shape[2]
            # queries length
            T_q = q.shape[2]
        else:
            # RoPE on q and k for full sequence
            cos, sin = _rope_get_cos_sin(T, d_head)
            q = _rope_apply(q, cos, sin)
            k = _rope_apply(k, cos, sin)
            k_full, v_full = k, v
            T_k = T
            T_q = T

        # Scaled dot-product attention
        attn_scores = jnp.einsum('bhtd,bhkd->bhtk', q, k_full) / jnp.sqrt(jnp.array(d_head, dtype=jnp.float32))
        assert_(attn_scores.shape == (B, self.num_heads, T_q, T_k), f"attn_scores bad shape {attn_scores.shape}")

        # Build sparse mask (Claude-style): causal + local window + global tokens
        lw = min(TRANSFORMER_CONFIG["local_window"], T_k)
        gs = max(1, TRANSFORMER_CONFIG["global_stride"])  # every gs-th token is global
        # causal mask base (upper triangle True means masked) over key length
        base_causal = jnp.triu(jnp.ones((T_q, T_k), dtype=bool), k=1 if not use_cache else 0)
        # local visibility relative to keys length
        q_idx = jnp.arange(T_q) + (0 if not use_cache else (T_k - T_q))
        k_idx = jnp.arange(T_k)
        rel = k_idx[None, :] - q_idx[:, None]  # (T_q, T_k) j - i
        local_ok = (rel <= 0) & (rel >= -(lw - 1))
        # global keys and queries
        is_global = (k_idx % gs) == 0  # (T_k,)
        # global tokens can be attended by anyone; and global queries can see all past keys
        key_global = jnp.tile(is_global[None, :], (T_q, 1))
        query_is_global = ((q_idx % gs) == 0)[:, None]  # (T_q,1)
        # allowed if local_ok OR key is global OR (query is global and key is in the past)
        past_ok = rel <= 0
        allowed = local_ok | key_global | (query_is_global & past_ok)
        # apply causal base too (should be redundant with past_ok, but keep safety)
        allowed = allowed & (~base_causal)
        # expand to (B, H, T, T)
        allowed = allowed[None, None, :, :]

        use_flash = bool(TRANSFORMER_CONFIG.get("use_flash_attention", True))
        if use_flash:
            # Flax DPA expects attention_bias or attention_mask; we'll pass a mask where
            # True = keep, False = mask. DPA takes mask with shape (B, H, T, T) or (T, T).
            # We'll pass (B,H,T,T) boolean mask.
            attn_mask = allowed
            # dot_product_attention expects q,k,v as (..., T, Dh). We already have (B,H,T,Dh).
            out = dot_product_attention(
                query=q,
                key=k_full,
                value=v_full,
                bias=None,
                mask=attn_mask,
                dropout_rate=TRANSFORMER_CONFIG["attention_dropout_rate"] if train else 0.0,
                deterministic=not train,
            )  # (B, H, T, Dh)
        else:
            # Manual attention with masking
            attn_scores = jnp.where(allowed, attn_scores, jnp.full_like(attn_scores, -1e9))
            attn_weights = jax.nn.softmax(attn_scores, axis=-1)
            if train:
                attn_weights = nn.Dropout(TRANSFORMER_CONFIG["attention_dropout_rate"])(attn_weights, deterministic=not train)
            out = jnp.einsum('bhtk,bhkd->bhtd', attn_weights, v_full)  # (B, H, T_q, Dh)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(B, T_q, C)
        out = nn.Dense(self.d_model, dtype=jnp.float32)(out)
        return out, cache

class FeedForward(nn.Module):
    d_model: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        h = nn.Dense(self.d_ff, dtype=jnp.float32)(x)
        h = nn.gelu(h)
        if train:
            h = nn.Dropout(self.dropout_rate)(h, deterministic=not train)
        h = nn.Dense(self.d_model, dtype=jnp.float32)(h)
        return h

class TransformerLM(nn.Module):
    """
    Minimal Transformer language model with RoPE-based self-attention.
    Stack of pre-norm Transformer blocks, then LM head.
    """
    vocab_size: int
    d_model: int = 256
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True, cache: list[dict] | None = None) -> jnp.ndarray | tuple[jnp.ndarray, list[dict]]:
        assert_(x.ndim == 2, f"Input tokens must be (batch, seq), got {x.shape}")
        B, T = x.shape
        emb = nn.Embed(self.vocab_size, self.d_model, dtype=jnp.float32)
        h = emb(x)  # (B, T, d_model)
        assert_(h.ndim == 3 and h.shape[0] == B and h.shape[1] == T, f"Embed bad shape {h.shape}")

        # Transformer blocks (pre-norm)
        new_cache = [] if cache is not None else None
        for i in range(TRANSFORMER_CONFIG["num_layers"]):
            h_norm = nn.LayerNorm(dtype=jnp.float32, name=f"ln_attn_{i}")(h)
            attn_out, layer_cache = MultiHeadSelfAttention(
                d_model=self.d_model,
                num_heads=TRANSFORMER_CONFIG["num_heads"],
                dropout_rate=self.dropout_rate,
                name=f"mha_{i}"
            )(h_norm, train=train, causal=True, cache=(None if cache is None else cache[i]))
            if new_cache is not None:
                new_cache.append(layer_cache)
            if train:
                attn_out = nn.Dropout(self.dropout_rate, name=f"drop_attn_{i}")(attn_out, deterministic=not train)
            h = h + attn_out

            ff_in = nn.LayerNorm(dtype=jnp.float32, name=f"ln_ff_{i}")(h)
            ff_out = FeedForward(self.d_model, TRANSFORMER_CONFIG["d_ff"], self.dropout_rate, name=f"ff_{i}")(ff_in, train=train)
            if train:
                ff_out = nn.Dropout(self.dropout_rate, name=f"drop_ff_{i}")(ff_out, deterministic=not train)
            h = h + ff_out

        h = nn.LayerNorm(dtype=jnp.float32, name="ln_out")(h)
        logits = nn.Dense(self.vocab_size, dtype=jnp.float32, name="lm_head")(h)  # (B, T, V)
        assert_(logits.shape == (B, T, self.vocab_size), f"Logits bad shape {logits.shape}")
        if new_cache is not None:
            return logits, new_cache
        return logits
    
# ------------------
# 3. TRAINING UTIL
# ------------------
# (TODO weight decay will be added to optimizer)

def cross_entropy_loss(logits, targets):
    one_hot = jax.nn.one_hot(targets, logits.shape[-1], dtype=jnp.float32)
    loss = optax.softmax_cross_entropy(logits, one_hot)
    return loss.mean()

class TrainState(train_state.TrainState):
    pass

def _loss_and_grads(params, x, y, apply_fn, dropout_rng):
    def inner(p):
        logits = apply_fn({'params': p}, x, train=True, rngs={'dropout': dropout_rng})
        loss = cross_entropy_loss(logits, y)
        return loss, None
    grad_fn = jax.value_and_grad(inner, has_aux=True)
    (loss, _), grads = grad_fn(params)
    return loss, grads

# ------------------
# 4. INIT + TRAIN + CSV LOGGING
# ------------------

# Notes for future improvements:
# - Tune regularization:
# e.g. try dropout_rate = 0.1 or 0.3 (instead of 0.2)
# e.g. weight_decay = 1e-5 or 5e-4 (instead of 1e-4)
# e.g. grad_clip_norm = 0.5 or 2.0 (instead of 1.0)
# - Adjust learning schedule:
# e.g. warmup_steps = 500 (instead of 100)
# e.g. decay_steps = 5000 (instead of 1000)
# e.g. peak_value = 5e-4 or 2e-3 (instead of 1e-3)
# e.g. end_value = 1e-6 (instead of 1e-5)
# - Add more layers:
# Code changes are required. For example, wrap multiple GRUCells in nn.Sequential,
# or define a stack of GRUs inside RNNLM. Simply changing a parameter won’t add depth.

rng = jax.random.PRNGKey(0)
dropout_rng, init_rng = jax.random.split(rng)
# Update config with runtime vocab size
TRANSFORMER_CONFIG["vocab_size"] = int(len(vocab))
model = TransformerLM(vocab_size=vocab_size, d_model=TRANSFORMER_CONFIG["d_model"], dropout_rate=TRANSFORMER_CONFIG["dropout_rate"])
x0 = jnp.array([data[:seq_len]], dtype=jnp.int32)
y0 = jnp.array([data[1:seq_len+1]], dtype=jnp.int32)
variables = model.init({'params': init_rng, 'dropout': dropout_rng}, x0)
params = variables['params']

warmup_steps = 500
base_lr = 1e-3
schedule_fn = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=base_lr,
    warmup_steps=warmup_steps,
    decay_steps=5000,
    end_value=1e-6,
)

grad_clip_norm = 0.5
clip_adam_wd = optax.chain(
    optax.clip_by_global_norm(grad_clip_norm),
    optax.adamw(schedule_fn, weight_decay=1e-5)
)

state = TrainState.create(apply_fn=model.apply, params=params, tx=clip_adam_wd)

USE_CHECKPOINTS = False
if USE_CHECKPOINTS:
    ckpt_dir = os.path.abspath("checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    state = checkpoints.restore_checkpoint(ckpt_dir, state)
    if state.params['Embed_0']['embedding'].shape[0] != vocab_size:
        variables = model.init({'params': init_rng, 'dropout': dropout_rng}, x0)
        state = TrainState.create(apply_fn=model.apply, params=variables['params'], tx=clip_adam_wd)
else:
    print("Starting fresh: skipping checkpoint restore.")

# CSV logger setup
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")

with open(log_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "step", "train_loss", "val_loss", "perplexity"])

print(f"Training log will be saved to {log_file}")

@jax.jit
def train_step_accum(state, x_mb, y_mb, dropout_rng, accum_steps=4):
    """Accumulate grads across the first axis of x_mb/y_mb (accum_steps).
    x_mb: (accum_steps, micro_bsz, seq_len)
    y_mb: (accum_steps, micro_bsz, seq_len)
    """
    def acc_fn(carry, batch):
        s, grads_acc, loss_acc, rng = carry
        xb, yb = batch  # (micro_bsz, seq_len)
        rng, key = jax.random.split(rng)
        loss, grads = _loss_and_grads(s.params, xb, yb, s.apply_fn, key)
        # Mixed-precision accumulation
        grads = jax.tree_util.tree_map(lambda g: jax.lax.convert_element_type(g, jnp.bfloat16), grads)
        grads_acc = jax.tree_util.tree_map(lambda a, b: a + b, grads_acc, grads)
        loss_acc = loss_acc + loss
        return (s, grads_acc, loss_acc, rng), None

    # initialize grads accumulator with zeros
    init_grads = jax.tree_util.tree_map(jnp.zeros_like, state.params)
    (state, grads_acc, loss_acc, _), _ = jax.lax.scan(
        acc_fn,
        (state, init_grads, 0.0, dropout_rng),
        (x_mb, y_mb)
    )
    grads_acc = jax.tree_util.tree_map(lambda g: jnp.asarray(g, jnp.float32) / accum_steps, grads_acc)
    state = state.apply_gradients(grads=grads_acc)
    return state, loss_acc / accum_steps

# Simple training loop with per-step logging and buffered CSV writing
try:
    @partial(jax.jit, static_argnames=("apply_fn",))
    def eval_batch(params, x, y, apply_fn):
        logits = apply_fn({'params': params}, x, train=False)
        loss = cross_entropy_loss(logits, y)
        return loss

# Fallback for older JAX
except TypeError:
    @partial(jax.jit, static_argnums=(3,))  # apply_fn is the 4th arg (0-based index 3)
    def eval_batch(params, x, y, apply_fn):
        logits = apply_fn({'params': params}, x, train=False)
        loss = cross_entropy_loss(logits, y)
        return loss

def train_loop(state, train_x, train_y, val_x, val_y, rng, epochs=5):
    steps_per_epoch = train_x.shape[0]
    for epoch in range(1, epochs+1):
        logs = []
        loss = 0.0 # Initialize loss to handle empty training set
        
        # Create a new key for each epoch to ensure different dropout masks
        rng, epoch_rng = jax.random.split(rng)
        step_keys = jax.random.split(epoch_rng, steps_per_epoch)

        for step in range(steps_per_epoch):
            x = jnp.array(train_x[step], dtype=jnp.int32)
            y = jnp.array(train_y[step], dtype=jnp.int32)
            x_mb = jnp.reshape(x, (ACCUM_STEPS, MICRO_BSZ, seq_len))
            y_mb = jnp.reshape(y, (ACCUM_STEPS, MICRO_BSZ, seq_len))
            state, loss = train_step_accum(state, x_mb, y_mb, step_keys[step], accum_steps=ACCUM_STEPS)

            if step % 10 == 0:  # log every 10 steps
                logs.append([epoch, step, float(loss), None, None])

        # validation
        val_losses = []
        for i in range(val_x.shape[0]):
            vx = jnp.array(val_x[i], dtype=jnp.int32)
            vy = jnp.array(val_y[i], dtype=jnp.int32)
            vloss = eval_batch(state.params, vx, vy, state.apply_fn)
            val_losses.append(float(vloss))
        if val_losses:
            avg_val_loss = np.mean(val_losses)
            ppl = np.exp(avg_val_loss)
            # "loss" is possibly unboundPylancereportPossiblyUnboundVariable
            print_(f"Epoch {epoch}: train_loss={float(loss):.4f}, val_loss={avg_val_loss:.4f}, ppl={ppl:.2f}")
            logs.append([epoch, steps_per_epoch, float(loss), avg_val_loss, ppl])

        # write buffered logs to CSV
        if logs:
            with open(log_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(logs)
    return state

start_time = time.time()

state = train_loop(state, train_x, train_y, val_x, val_y, rng, epochs=num_epochs)

training_time = time.time() - start_time
print(f"Total training time: {training_time:.2f} seconds")

# ------------------
# 5. TEXT GENERATION
# ------------------
def generate(state, start="To ", length=200, carry=None, temperature=1.0, rng_key: jax.Array | None = None):
    idxs = encode(start)
    assert_(idxs.ndim == 1, f"Encoded start must be 1D, got {idxs.shape}")
    if len(idxs) == 0:
        return start
    x = jnp.array([idxs], dtype=jnp.int32)
    out_tokens = [itos[int(i)] for i in idxs]
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    for _ in range(length):
        logits = state.apply_fn({'params': state.params}, x, train=False)
        logits = logits[0, -1] / temperature
        rng_key, subkey = jax.random.split(rng_key)
        next_id = int(jax.random.categorical(subkey, logits))
        out_tokens.append(itos[next_id])
        x = jnp.array([[next_id]], dtype=jnp.int32)
    text = "".join(out_tokens)
    text = text.replace("</w>", "")
    text = text.replace(space_token, " ")
    return text

# Run generation and save output with parameters
gen_start = """Q: Why optimize Tokenizer efficiency?
A:"""
gen_temp = 0.8
gen_len = 200
output_text = generate(state, start=gen_start, temperature=gen_temp, rng_key=jax.random.PRNGKey(42))

print("\nGenerated text:\n", output_text)

# Save to file with parameters
out_record = {
    "init": {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_epochs": num_epochs,
        "learning_rate": base_lr,
        "grad_clip_norm": grad_clip_norm,
        "d_model": TRANSFORMER_CONFIG["d_model"],
        "dropout_rate": TRANSFORMER_CONFIG["dropout_rate"],
    },
    "training": {
        "training_time_sec": training_time,
        "log_file": log_file
    },
    "generation": {
        "start": gen_start,
        "temperature": gen_temp,
        "length": gen_len,
        "output": output_text
    }
}
with open("generation_output.json", "w", encoding="utf-8") as f:
    json.dump(out_record, f, ensure_ascii=False, indent=2)

