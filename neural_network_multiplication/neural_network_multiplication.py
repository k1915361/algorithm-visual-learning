# Q. Can NN or approximations speed up or improve matrix multiplication?

# I later realised those cannot beat BLAS implementations even at 70-90% accuracy. Please ignore this file.

# Dense multiplications in Transformers/diffusion are run by cuBLAS/cuDNN kernels that already give near-peak GPU throughput with **exact results (100% accuracy)**. Approximate methods (Monte Carlo, NN approximators) can’t beat them in speed at small/moderate dimensions, even if you accept 70–90% accuracy, because the GPU hardware is optimized for exact GEMM.

# Attention: Approximations only make sense when the operation is *too large to fit memory or compute time*. For LLMs this shows up in **attention** (long sequence lengths) where approximate/low-rank/kernalized methods give big speedups at the cost of accuracy. 
# Attention/Sampling: For image diffusion models, approximations target **attention layers or sampling steps**, not the dense multiplications in linear layers or convolutions.

# - **Dense matmul** → exact BLAS kernels are faster and more accurate.
# - **Attention/long-sequence ops** → approximations can save cost with tolerable accuracy loss.
# - **Neural net approximations** could replace blocks (distillation, pruning, low-rank factorization), but not the single dense multiply kernel.

# Benchmark: Direct Multiply vs Neural Network (NumPy vs JAX vs PyTorch + Probabilistic Approximation, with JAX Monte Carlo)
import os
import time
import timeit
import numpy as np
import logging

os.environ["JAX_PLATFORM_NAME"] = "cpu"

logging.basicConfig(filename="run.log", level=logging.INFO)
log = logging.getLogger(__name__)

try:
    import jax
    import jax.numpy as jnp
    from jax import jit
except ImportError:
    log.info("jax not found. using numpy instead.")
    jax = None

try:
    import torch
except ImportError:
    log.info("torch not found.")
    torch = None

rng = np.random.default_rng(0)

def now():
    return time.perf_counter()

# ----------------------
# Python Direct Multiply (naive loops)
# ----------------------
def python_matrix_mult(A, B):
    n = len(A)
    m = len(A[0])
    p = len(B[0])
    C = [[0.0] * p for _ in range(n)]
    for i in range(n):
        for j in range(p):
            for k in range(m):
                C[i][j] += A[i][k] * B[k][j]
    return C

# ----------------------
# Simple NN with NumPy
# ----------------------
class NumpyMLP:
    def __init__(self, in_dim, hidden, out_dim, rng, scale=0.1):
        self.W1 = rng.normal(scale=scale, size=(in_dim, hidden))
        self.b1 = np.zeros((hidden,))
        self.W2 = rng.normal(scale=scale, size=(hidden, out_dim))
        self.b2 = np.zeros((out_dim,))

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.H1 = np.tanh(self.Z1)
        self.Y  = self.H1 @ self.W2 + self.b2
        return self.Y

# ----------------------
# Simple NN with JAX
# ----------------------
if jax is not None:
    class JaxMLP:
        def __init__(self, in_dim, hidden, out_dim, rng_key, scale=0.1):
            k1, k2 = jax.random.split(rng_key)
            self.W1 = jax.random.normal(k1, (in_dim, hidden)) * scale
            self.b1 = jnp.zeros((hidden,))
            self.W2 = jax.random.normal(k2, (hidden, out_dim)) * scale
            self.b2 = jnp.zeros((out_dim,))

        def forward(self, X):
            Z1 = X @ self.W1 + self.b1
            H1 = jnp.tanh(Z1)
            Y = H1 @ self.W2 + self.b2
            return Y

        def jit_forward(self, X):
            return jit(self.forward)(X)

# ----------------------
# Probabilistic Approximation: Monte Carlo Matrix Multiplication

# ----------------------
def monte_carlo_matrix_mult(A, B, samples=16, rounds=20):
    n, m = A.shape
    m2, p = B.shape
    assert m == m2
    acc = np.zeros((n, p))
    for _ in range(rounds):
        idx = rng.integers(0, m, size=(samples,))
        A_samples = A[:, idx]
        B_samples = B[idx, :]
        contribs = A_samples @ B_samples
        acc += (m / samples) * contribs
    return acc / rounds

def monte_carlo_batch(A, B, samples=16, rounds=20):
    n, d, _ = A.shape
    results = np.empty((n, d, d))
    for i in range(n):
        results[i] = monte_carlo_matrix_mult(A[i], B[i], samples=samples, rounds=rounds)
    return results

# ----------------------
# JAX Monte Carlo Approximation
# ----------------------
if jax is not None:
    def jax_monte_carlo_matrix_mult(A, B, key, samples=16, rounds=20):
        """
        Use more samples and more rounds to reduce variance (lower error).
        """
        n, m = A.shape
        _, p = B.shape
        acc = jnp.zeros((n, p))

        def body_fun(i, acc):
            k = jax.random.fold_in(key, i)
            idx = jax.random.randint(k, (samples,), 0, m)
            A_samples = A[:, idx]
            B_samples = B[idx, :]
            contribs = A_samples @ B_samples
            return acc + (m / samples) * contribs

        acc = jax.lax.fori_loop(0, rounds, body_fun, acc)
        return acc / rounds

    jax_monte_carlo_matrix_mult = jit(jax_monte_carlo_matrix_mult, static_argnums=(3,4))

    def jax_monte_carlo_batch(A, B, samples=16, rounds=20):
        results = []
        key = jax.random.PRNGKey(0)
        for i in range(A.shape[0]):
            results.append(jax_monte_carlo_matrix_mult(A[i], B[i], key, samples, rounds))
        return jnp.stack(results)

# ----------------------
# Profiling utilities (with timeit)
# ----------------------
def profile_numpy_matrix_mult(n=5000):
    A = rng.normal(size=(n, 32, 32))
    B = rng.normal(size=(n, 32, 32))

    exact = A @ B

    t_calc = timeit.timeit(lambda: A @ B, number=3) / 3

    def python_test():
        for i in range(min(n,10)):
            _ = python_matrix_mult(A[i].tolist(), B[i].tolist())
    t_python = timeit.timeit(python_test, number=3) / 3

    mlp = NumpyMLP(2048, 64, 64, rng)
    X = np.concatenate([A.reshape(n, -1), B.reshape(n, -1)], axis=1)
    out_nn = mlp.forward(X)
    t_nn = timeit.timeit(lambda: mlp.forward(X), number=3) / 3

    approx = monte_carlo_batch(A, B, samples=16, rounds=5)
    t_prob = timeit.timeit(lambda: monte_carlo_batch(A, B, samples=16, rounds=5), number=3) / 3

    # accuracy vs exact
    error_prob = np.linalg.norm(approx - exact) / np.linalg.norm(exact)
    error_nn = np.linalg.norm(out_nn - exact.reshape(n, -1)[:, :out_nn.shape[1]]) / np.linalg.norm(exact)

    return t_calc, t_python, t_nn, t_prob, error_prob, error_nn


def profile_jax_matrix_mult(n=5000):
    if jax is None:
        return None, None, None, None
    A = jnp.array(rng.normal(size=(n, 32, 32)))
    B = jnp.array(rng.normal(size=(n, 32, 32)))

    t_calc = timeit.timeit(lambda: jax.block_until_ready(A @ B), number=3) / 3

    rng_key = jax.random.PRNGKey(0)
    mlp = JaxMLP(2048, 64, 64, rng_key)
    X = jnp.concatenate([A.reshape(n, -1), B.reshape(n, -1)], axis=1)

    fwd = mlp.jit_forward(X)
    out_nn = fwd
    t_nn = timeit.timeit(lambda: jax.block_until_ready(fwd), number=3) / 3

    # compare with exact (reshaped)
    exact = np.array(A) @ np.array(B)
    error_nn = np.linalg.norm(np.array(out_nn) - exact.reshape(n, -1)[:, :out_nn.shape[1]]) / np.linalg.norm(exact)

    # JAX Monte Carlo
    samples=32
    rounds=20
    """
    samples
    16
    32
    """
    jax_mc = jax_monte_carlo_batch(A, B, samples=samples, rounds=rounds)
    t_mc = timeit.timeit(lambda: jax.block_until_ready(jax_monte_carlo_batch(A, B, samples=samples, rounds=rounds)), number=3) / 3
    error_mc = np.linalg.norm(np.array(jax_mc) - exact) / np.linalg.norm(exact)

    return t_calc, t_nn, error_nn, (t_mc, error_mc)


def profile_torch_matrix_mult(n=5000):
    if torch is None:
        return None
    A = torch.randn((n, 32, 32))
    B = torch.randn((n, 32, 32))

    def torch_test():
        C = A @ B
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    t_calc = timeit.timeit(torch_test, number=3) / 3
    return t_calc

# ----------------------
# Format helper
# ----------------------
def format_time_per_op(seconds):
    ms = seconds * 1000
    if ms < 1:
        us = ms * 1000
        if us < 1:
            ns = us * 1000
            return f"{ns:.2f} ns/op"
        return f"{us:.2f} µs/op"
    return f"{ms:.4f} ms/op"

# ----------------------
# Run benchmark
# ----------------------
n = 500
numpy_calc, python_calc, numpy_nn, numpy_prob, prob_err, numpy_err = profile_numpy_matrix_mult(n)
jax_calc, jax_nn, jax_err, jax_mc = profile_jax_matrix_mult(n)
torch_calc = profile_torch_matrix_mult(n)

# log.info("Python direct multiply:", format_time_per_op(python_calc/min(n,10)))
log.info("Matrix Multiplication Profiling (time per operation, via timeit):")
log.info("NumPy direct multiply: %s", format_time_per_op(numpy_calc/n))
log.info("NumPy NN forward: %s | rel error: %.4e (%.2f%%)", format_time_per_op(numpy_nn/n), numpy_err, numpy_err*100)
log.info("Monte Carlo approx: %s | rel error: %.4e (%.2f%%)", format_time_per_op(numpy_prob/n), prob_err, prob_err*100)
if jax_calc is not None:
    log.info("JAX direct multiply: %s", format_time_per_op(jax_calc/n))
    log.info("JAX NN forward (jit): %s | rel error: %.4e (%.2f%%)", format_time_per_op(jax_nn/n), jax_err, jax_err*100)
    if jax_mc is not None:
        log.info("JAX Monte Carlo: %s | rel error: %.4e (%.2f%%)", format_time_per_op(jax_mc[0]/n), jax_mc[1], jax_mc[1]*100)
if torch_calc is not None:
    log.info("Torch direct multiply: %s", format_time_per_op(torch_calc/n))

f"""
n = 500
Python direct multiply: 4.2580 ms/op
NumPy NN forward: 6.73 µs/op | rel error: 2.5141e-01 (25.14%)
Monte Carlo approx: 130.26 µs/op | rel error: 6.2712e-01 (62.71%)
JAX direct multiply: 22.01 µs/op
JAX NN forward (jit): 10.80 ns/op | rel error: 2.5440e-01 (25.44%)
JAX Monte Carlo: 323.85 µs/op | rel error: 5.9020e-01 (59.02%)
Torch direct multiply: 1.66 µs/op
"""

"""
1 ms   = 1e-3 s
1 µs   = 1e-6 s
1 ns   = 1e-9 s

0.1    = 1e-1
1      = 1e+0
10     = 1e+1
100    = 1e+2
1000   = 1e+3
"""
