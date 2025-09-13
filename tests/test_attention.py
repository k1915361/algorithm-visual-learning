import jax
import jax.numpy as jnp
from tiny_transformer.main import TransformerLM, TRANSFORMER_CONFIG, build_sparse_allowed_mask


def test_sparse_mask_pattern_small():
    T, lw, gs = 16, TRANSFORMER_CONFIG["local_window"], TRANSFORMER_CONFIG["global_stride"]
    mask = build_sparse_allowed_mask(T, lw, gs)
    assert mask.shape == (T, T)
    # diagonal allowed
    assert bool(jnp.all(jnp.diag(mask)))
    # future masked
    assert bool(jnp.all(jnp.triu(~mask, k=1)))
    # globals visible
    globals_idx = jnp.arange(0, T, gs)
    assert bool(jnp.all(mask[:, globals_idx]))


def test_attention_equivalence_flash_vs_manual():
    # Tiny deterministic model
    vocab = 32
    T = 8
    TRANSFORMER_CONFIG["use_flash_attention"] = True
    model = TransformerLM(vocab_size=vocab, d_model=64, dropout_rate=0.0)
    x = jnp.arange(T)[None, :].astype(jnp.int32) % vocab
    params = model.init({'params': jax.random.PRNGKey(0)}, x)['params']
    y_flash = model.apply({'params': params}, x, train=False)

    TRANSFORMER_CONFIG["use_flash_attention"] = False
    y_manual = model.apply({'params': params}, x, train=False)

    assert jnp.allclose(y_flash, y_manual, atol=1e-5, rtol=1e-5)
    # restore
    TRANSFORMER_CONFIG["use_flash_attention"] = True
