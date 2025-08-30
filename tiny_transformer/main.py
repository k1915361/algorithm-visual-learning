
from functools import partial
import json
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state, checkpoints
from typing import Any
import time
import matplotlib.pyplot as plt
from datetime import datetime
import os
import csv
from collections import Counter
from jax import debug as jdbg

DEBUG = False  # or bool(int(os.getenv("DEBUG", "0")))

# TODO add debugprint for troubleshoot and assert checks

from jax import debug as jdbg

def debugprint(*args, **kwargs):
    if DEBUG:
        jdbg.print(*args, **kwargs)

# ------------------
# 1. DATASET + BPE TOKENIZER
# ------------------
text = """Q: What does the cross_entropy_loss function do?
A: It measures how well the predicted probability distribution matches the true labels.

Q: How is it calculated in our RNN?
A: First, the targets are converted into one-hot vectors. Then the softmax of logits is compared with the one-hot vector using cross-entropy.

Q: What is the math formula for cross-entropy?
A: loss = - Σ y_true * log(y_pred), averaged over all tokens.

Q: Why do we use it?
A: To train the model so that predicted probabilities for correct tokens are maximized.

Q: What is perplexity?
A: Perplexity = exp(cross_entropy_loss). It tells us on average how many equally likely choices the model is considering.

Q: What is the role of embeddings?
A: They map token indices into dense vectors so that similar tokens have similar representations.

Q: What is the GRU cell used for?
A: The GRU cell updates the hidden state at each time step, capturing sequence information without exploding/vanishing gradients.

Q: Why do we clip gradients?
A: To prevent exploding gradients, ensuring stable training.

Q: What does the generate function do?
A: It autoregressively predicts the next token from the model’s output, sampling until the desired length is reached."""

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
    return np.array([stoi[w] for w in symbols if w in stoi], dtype=np.int32)

def decode(arr):
    tokens = [itos[int(i)] for i in arr]
    text = "".join(tokens)
    text = text.replace("</w>", "")
    text = text.replace(space_token, " ")
    return text

# Explicitly fix dtype for dataset
data = jnp.array(encode(text), dtype=jnp.int32)
seq_len = 32
batch_size = 16
num_steps = 200
num_epochs = 10

# Gradient accumulation config
ACCUM_STEPS = 4  # must divide batch_size
assert batch_size % ACCUM_STEPS == 0, "ACCUM_STEPS must divide batch_size"
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
        return xs, ys

    train_x, train_y = make_batches(train_data)
    val_x, val_y = make_batches(val_data)
    return train_x, train_y, val_x, val_y

train_x, train_y, val_x, val_y = precompute_batches(np.array(data), seq_len, batch_size)

# ------------------
# 2. MODEL
# ------------------
class RNNLM(nn.Module):
    vocab_size: int
    hidden_size: int = 128
    embed_dim: int = 128
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(self, x, carry=None, train: bool = True):
        embed = nn.Embed(self.vocab_size, self.embed_dim, dtype=jnp.float32)
        x = embed(x)  # (batch, seq_len, embed_dim)
        if train:
            x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)
        if carry is None:
            carry = jnp.zeros((x.shape[0], self.hidden_size), dtype=jnp.float32)

        # Use flax.linen.scan to scan a GRUCell over time with params broadcast.
        ScanGRU = nn.scan(
            nn.GRUCell,
            variable_broadcast='params',
            split_rngs={'params': False},
            in_axes=1,
            out_axes=1,
            length=None,
        )
        gru = ScanGRU(features=self.hidden_size, dtype=jnp.float32)
        carry, h = gru(carry, x)  # x: (batch, time, embed_dim) -> h: (batch, time, hidden_size)

        if train:
            h = nn.Dropout(self.dropout_rate)(h, deterministic=not train)
        logits = nn.Dense(self.vocab_size, dtype=jnp.float32)(h)
        return logits, carry, x
    
# ------------------
# 3. TRAINING UTIL
# ------------------
# (TODO weight decay will be added to optimizer)

def cross_entropy_loss(logits, targets):
    one_hot = jax.nn.one_hot(targets, logits.shape[-1], dtype=jnp.float32)
    loss = optax.softmax_cross_entropy(logits, one_hot)
    return loss.mean()

class TrainState(train_state.TrainState):
    carry: Any = None

def _loss_and_grads(params, x, y, carry, apply_fn, dropout_rng):
    def inner(p):
        out = apply_fn({'params': p}, x, carry, train=True, rngs={'dropout': dropout_rng})
        logits, new_carry, embeds = out
        loss = cross_entropy_loss(logits, y)
        return loss, (new_carry, embeds)
    grad_fn = jax.value_and_grad(inner, has_aux=True)
    (loss, (new_carry, embeds)), grads = grad_fn(params)
    return loss, new_carry, embeds, grads

# ------------------
# 4. INIT + TRAIN + CSV LOGGING
# ------------------
rng = jax.random.PRNGKey(0)
dropout_rng, init_rng = jax.random.split(rng)
model = RNNLM(vocab_size=vocab_size)
x0 = jnp.array([data[:seq_len]], dtype=jnp.int32)
y0 = jnp.array([data[1:seq_len+1]], dtype=jnp.int32)
variables = model.init({'params': init_rng, 'dropout': dropout_rng}, x0, None)
params = variables['params']

warmup_steps = 100
base_lr = 1e-3
schedule_fn = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=base_lr,
    warmup_steps=warmup_steps,
    decay_steps=1000,
    end_value=1e-5,
)

grad_clip_norm = 1.0
clip_adam_wd = optax.chain(
    optax.clip_by_global_norm(grad_clip_norm),
    optax.adamw(schedule_fn, weight_decay=1e-4)
)

init_carry = jnp.zeros((batch_size, model.hidden_size), dtype=jnp.float32)
state = TrainState.create(apply_fn=model.apply, params=params, tx=clip_adam_wd, carry=init_carry)

USE_CHECKPOINTS = False
if USE_CHECKPOINTS:
    ckpt_dir = os.path.abspath("checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    state = checkpoints.restore_checkpoint(ckpt_dir, state)
    if state.params['Embed_0']['embedding'].shape[0] != vocab_size:
        variables = model.init({'params': init_rng, 'dropout': dropout_rng}, x0, None)
        state = TrainState.create(apply_fn=model.apply, params=variables['params'], tx=clip_adam_wd, carry=init_carry)
else:
    print("Starting fresh: skipping checkpoint restore.")

# CSV logger setup
log_file = "training_log.csv"
with open(log_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "step", "loss", "val_loss", "ppl"])

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
        # Re-init carry per micro-batch (pass None) to avoid shape mismatches
        loss, _new_carry, embeds, grads = _loss_and_grads(s.params, xb, yb, None, s.apply_fn, key)
        # Optional: debug shapes
        debugprint("acc_fn xb shape = {}", xb.shape)
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
    def eval_batch(params, x, y, carry, apply_fn):
        logits, new_carry, embeds = apply_fn({'params': params}, x, carry, train=False)
        loss = cross_entropy_loss(logits, y)
        return loss, new_carry

# Fallback for older JAX
except TypeError:
    @partial(jax.jit, static_argnums=(4,))  # apply_fn is the 5th arg (0-based index 4)
    def eval_batch(params, x, y, carry, apply_fn):
        logits, new_carry, embeds = apply_fn({'params': params}, x, carry, train=False)
        loss = cross_entropy_loss(logits, y)
        return loss, new_carry
    
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
        carry = state.carry
        for i in range(val_x.shape[0]):
            vx = jnp.array(val_x[i], dtype=jnp.int32)
            vy = jnp.array(val_y[i], dtype=jnp.int32)
            vloss, carry = eval_batch(state.params, vx, vy, carry, state.apply_fn)
            val_losses.append(float(vloss))
        if val_losses:
            avg_val_loss = np.mean(val_losses)
            ppl = np.exp(avg_val_loss)
            # "loss" is possibly unboundPylancereportPossiblyUnboundVariable
            print(f"Epoch {epoch}: train_loss={float(loss):.4f}, val_loss={avg_val_loss:.4f}, ppl={ppl:.2f}")
            logs.append([epoch, steps_per_epoch, float(loss), avg_val_loss, ppl])

        # write buffered logs to CSV
        if logs:
            with open(log_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(logs)
    return state

state = train_loop(state, train_x, train_y, val_x, val_y, rng, epochs=num_epochs)


# ------------------
# 5. TEXT GENERATION
# ------------------
def generate(state, start="To ", length=200, carry=None, temperature=1.0):
    idxs = encode(start)
    if len(idxs) == 0:
        return start
    x = jnp.array([idxs], dtype=jnp.int32)
    carry = carry if carry is not None else state.carry
    out_tokens = [itos[int(i)] for i in idxs]
    for _ in range(length):
        logits, carry, embeds = state.apply_fn({'params': state.params}, x, carry, train=False)
        logits = logits[0, -1] / temperature
        probs = jax.nn.softmax(logits)
        next_id = int(np.random.choice(len(probs), p=np.array(probs)))
        out_tokens.append(itos[next_id])
        x = jnp.array([[next_id]], dtype=jnp.int32)
    text = "".join(out_tokens)
    text = text.replace("</w>", "")
    text = text.replace(space_token, " ")
    return text

# Run generation and save output with parameters
gen_start = "To "
gen_temp = 0.8
gen_len = 200
output_text = generate(state, start=gen_start, carry=state.carry, temperature=gen_temp)

print("\nGenerated text:\n", output_text)

# Save to file with parameters
out_record = {
    "start": gen_start,
    "temperature": gen_temp,
    "length": gen_len,
    "output": output_text
}
with open("generation_output.json", "w", encoding="utf-8") as f:
    json.dump(out_record, f, ensure_ascii=False, indent=2)

