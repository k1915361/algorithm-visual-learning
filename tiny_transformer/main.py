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

# ------------------
# 1. DATASET (bigram-level)
# ------------------
text = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune..."""

# Build bigrams (2-character tokens)
bigrams = [text[i:i+2] for i in range(len(text)-1)]
vocab = sorted(set(bigrams))
stoi = {bg: i for i, bg in enumerate(vocab)}
itos = {i: bg for bg, i in stoi.items()}
vocab_size = len(vocab)

print("[DEBUG] Vocab size:", vocab_size)

def encode(s):
    pairs = [s[i:i+2] for i in range(len(s)-1)]
    return np.array([stoi[p] for p in pairs if p in stoi], dtype=np.int32)

def decode(arr):
    return "".join([itos[int(i)][0] for i in arr] + [itos[int(arr[-1])][1]]) if len(arr) > 0 else ""

# Explicitly fix dtype for dataset
data = jnp.array(encode(text), dtype=jnp.int32)
seq_len = 32  # shorter because tokens are bigger
batch_size = 16
num_steps = 200
num_epochs = 10

print("[DEBUG] Data shape:", data.shape)

# ------------------
# 2. MODEL
# ------------------
class RNNLM(nn.Module):
    vocab_size: int
    hidden_size: int = 128
    embed_dim: int = 64

    @nn.compact
    def __call__(self, x, carry=None):
        print("[DEBUG] Input shape to model:", x.shape)
        embed = nn.Embed(self.vocab_size, self.embed_dim, dtype=jnp.float32)
        expected_shape = (self.vocab_size, self.embed_dim)
        print(f"[DEBUG] Creating embedding with expected shape {expected_shape}")
        x = embed(x)
        print("[DEBUG] After embedding shape:", x.shape)
        gru = nn.GRUCell(features=self.hidden_size, dtype=jnp.float32)
        outputs = []
        if carry is None:
            carry = jnp.zeros((x.shape[0], self.hidden_size), dtype=jnp.float32)
        for t in range(x.shape[1]):
            carry, out = gru(carry, x[:, t])
            outputs.append(out)
        h = jnp.stack(outputs, axis=1)
        logits = nn.Dense(self.vocab_size, dtype=jnp.float32)(h)
        # TODO resolve: flax.errors.ScopeParamShapeError: Initializer expected to generate shape (26, 64) but got shape (85, 64) instead for parameter "embedding" in "/Embed_0".
        print("[DEBUG] Logits shape:", logits.shape)
        return logits, carry, x

# ------------------
# 3. TRAINING UTIL
# ------------------
def cross_entropy_loss(logits, targets):
    one_hot = jax.nn.one_hot(targets, logits.shape[-1], dtype=jnp.float32)
    loss = optax.softmax_cross_entropy(logits, one_hot)
    return loss.mean()

class TrainState(train_state.TrainState):
    carry: Any = None

def debug_param_shapes(params, label):
    print(f"[DEBUG] Param shapes at {label}:")
    def recurse(d, prefix=""):
        if isinstance(d, dict):
            for k, v in d.items():
                recurse(v, prefix + k + "/")
        else:
            try:
                print(f"  {prefix[:-1]}: {d.shape}")
            except AttributeError:
                print(f"  {prefix[:-1]}: (no shape, type {type(d)})")
    recurse(params)

def _loss_and_grads(params, x, y, carry, apply_fn):
    def inner(p):
        print("[DEBUG] _loss_and_grads input shapes -> x:", x.shape, "y:", y.shape)
        debug_param_shapes(p, "apply-start")
        out = apply_fn({'params': p}, x, carry)
        logits, new_carry, embeds = out
        print("[DEBUG] _loss_and_grads logits shape:", logits.shape)
        debug_param_shapes(p, "apply-end")
        loss = cross_entropy_loss(logits, y)
        return loss, (new_carry, embeds)
    grad_fn = jax.value_and_grad(inner, has_aux=True)
    (loss, (new_carry, embeds)), grads = grad_fn(params)
    return loss, new_carry, embeds, grads

def train_step(state, x, y):
    print("[DEBUG] train_step batch_x shape:", x.shape, "batch_y shape:", y.shape)
    loss, carry, embeds, grads = _loss_and_grads(state.params, x, y, state.carry, state.apply_fn)
    state = state.apply_gradients(grads=grads)
    state = state.replace(carry=carry)
    return state, loss, embeds

# ------------------
# 4. INIT + TRAIN
# ------------------
rng = jax.random.PRNGKey(0)
model = RNNLM(vocab_size=vocab_size)
x0 = jnp.array([data[:seq_len]], dtype=jnp.int32)
y0 = jnp.array([data[1:seq_len+1]], dtype=jnp.int32)
print("[DEBUG] x0 shape:", x0.shape, "y0 shape:", y0.shape)
variables = model.init(rng, x0, None)
print("[DEBUG] Embed param shape from init:", variables['params']['Embed_0']['embedding'].shape)
params = variables['params']
debug_param_shapes(params, "init")

learning_rate = 1e-3
grad_clip_norm = 1.0
clip_and_adam = optax.chain(
    optax.clip_by_global_norm(grad_clip_norm),
    optax.adam(learning_rate)
)

init_carry = jnp.zeros((batch_size, model.hidden_size), dtype=jnp.float32)
state = TrainState.create(apply_fn=model.apply, params=params, tx=clip_and_adam, carry=init_carry)

ckpt_dir = os.path.abspath("checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)
state = checkpoints.restore_checkpoint(ckpt_dir, state)

def train_epoch(state, rng, num_steps=num_steps):
    def body_fn(carry, step):
        state, rng = carry
        rng, subkey = jax.random.split(rng)
        ix = jax.random.randint(subkey, (batch_size,), 0, len(data) - seq_len - 1)
        def make_example(i):
            x = jax.lax.dynamic_slice(data, (i,), (seq_len,))
            y = jax.lax.dynamic_slice(data, (i+1,), (seq_len,))
            return x, y
        batch_x, batch_y = jax.vmap(make_example)(ix)
        print("[DEBUG] batch_x shape:", batch_x.shape, "batch_y shape:", batch_y.shape)
        debug_param_shapes(state.params, "train-epoch")
        state = state.replace(carry=init_carry)
        state, loss, embeds = train_step(state, batch_x, batch_y)
        embed_norm = jnp.linalg.norm(embeds) / batch_size
        hidden_norm = jnp.linalg.norm(state.carry) / batch_size
        return (state, rng), (loss, hidden_norm, embed_norm)
    steps = jnp.arange(num_steps)
    (state, rng), metrics = jax.lax.scan(body_fn, (state, rng), steps)
    metrics_stacked = list(zip(*metrics))
    losses = jnp.stack(metrics_stacked[0])
    hidden_norms = jnp.stack(metrics_stacked[1])
    embed_norms = jnp.stack(metrics_stacked[2])
    return state, rng, losses.mean(), hidden_norms.mean(), embed_norms.mean()

def eval_batch(params, x, y, carry, apply_fn):
    print("[DEBUG] eval_batch shapes -> x:", x.shape, "y:", y.shape)
    debug_param_shapes(params, "eval-start")
    out = apply_fn({'params': params}, x, carry)
    logits, carry, embeds = out
    print("[DEBUG] eval_batch logits shape:", logits.shape)
    debug_param_shapes(params, "eval-end")
    loss = cross_entropy_loss(logits, y)
    return loss, carry

# ------------------
# (rest unchanged)

def evaluate(state, num_batches=20):
    losses = []
    carry = state.carry
    for _ in range(num_batches):
        ix = np.random.randint(0, len(data) - seq_len - 1, (batch_size,))
        xb = jnp.stack([jnp.array(data[i:i+seq_len], dtype=jnp.int32) for i in ix])
        yb = jnp.stack([jnp.array(data[i+1:i+seq_len+1], dtype=jnp.int32) for i in ix])
        loss, carry = eval_batch(state.params, xb, yb, carry)
        losses.append(loss)
    mean_loss = float(np.mean(jax.device_get(np.array(losses))))
    ppl = float(jnp.exp(mean_loss))
    return mean_loss, ppl

# ------------------
# 5. TRAINING LOOP WITH CHECKPOINTING + CSV LOGGING
# ------------------
log_dir = os.path.abspath("logs")
os.makedirs(log_dir, exist_ok=True)
csv_path = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")

best_eval_loss = float("inf")
best_state = None

with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "loss", "hidden_norm", "embed_norm", "eval_loss", "eval_ppl"])

    for epoch in range(num_epochs):
        start_time = time.time()
        state, rng, avg_loss, avg_hidden_norm, avg_embed_norm = train_epoch(state, rng)
        end_time = time.time()
        train_duration = end_time - start_time

        eval_loss, eval_ppl = evaluate(state)
        print(f"Epoch {epoch+1}/{num_epochs}, Train time {train_duration:.2f}s, Loss {avg_loss:.4f}, Eval loss {eval_loss:.4f}, Eval ppl {eval_ppl:.2f}")

        writer.writerow([epoch+1, float(avg_loss), float(avg_hidden_norm), float(avg_embed_norm), float(eval_loss), float(eval_ppl)])

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_state = state
            checkpoints.save_checkpoint(ckpt_dir, state, step=epoch, overwrite=True)
            print(f"  Saved new best checkpoint with eval loss {eval_loss:.4f}")

state = checkpoints.restore_checkpoint(ckpt_dir, state)

# ------------------
# 6. PLOTTING
# ------------------
log_data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
epochs = log_data[:, 0]
losses = log_data[:, 1]
hidden_norms = log_data[:, 2]
embed_norms = log_data[:, 3]

os.makedirs("plots", exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

fig, ax1 = plt.subplots()
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Training Loss", color='tab:blue')
line1, = ax1.plot(epochs, losses, color='tab:blue', label="Training Loss")
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel("Norms", color='tab:orange')
line2, = ax2.plot(epochs, hidden_norms, label="Hidden Norm", color='tab:orange')
line3, = ax2.plot(epochs, embed_norms, label="Embed Norm", color='tab:green')
ax2.tick_params(axis='y', labelcolor='tab:orange')

# Combine legends from both axes
lines = [line1, line2, line3]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper right")

plt.title("Average Loss vs GRU Hidden & Embedding Norms")
plt.tight_layout()
plt.savefig(os.path.join("plots", f"loss_vs_norms_{timestamp}.png"))
plt.show()

# ------------------
# 7. TEXT GENERATION
# ------------------
def generate(state, start="To ", length=200, carry=None, temperature=1.0):
    idxs = encode(start)
    if len(idxs) == 0:
        return start
    x = jnp.array([idxs], dtype=jnp.int32)
    carry = carry if carry is not None else state.carry
    out_text = list(start)
    for _ in range(length):
        logits, carry, embeds = state.apply_fn({'params': state.params}, x, carry)
        logits = logits[0, -1] / temperature
        probs = jax.nn.softmax(logits)
        next_id = int(np.random.choice(len(probs), p=np.array(probs)))
        out_text.append(itos[next_id][1])  # append second char of bigram
        x = jnp.array([[next_id]], dtype=jnp.int32)
    return "".join(out_text)

print("\nGenerated with best checkpoint:\n", generate(state, start="To ", carry=state.carry, temperature=0.8))
