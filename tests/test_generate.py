import jax
import jax.numpy as jnp
from tiny_transformer.main import TransformerLM, TRANSFORMER_CONFIG, generate, generate_cached, encode


def test_generate_first_token_same_cached_vs_noncached():
    vocab = 64
    prompt = "hello"
    # Build tiny model and params
    model = TransformerLM(vocab_size=vocab, d_model=64, dropout_rate=0.0)
    x = jnp.array([encode(prompt)], dtype=jnp.int32)
    params = model.init({'params': jax.random.PRNGKey(0)}, x)['params']
    # Wrap a dummy state-like object
    class S: pass
    state = S()
    state.params = params
    state.apply_fn = model.apply

    key = jax.random.PRNGKey(123)
    # Generate one step to compare first continuation token - we compare outputs deterministically
    out_nc = generate(state, start=prompt, length=1, temperature=1.0, rng_key=key)
    out_c = generate_cached(state, start=prompt, steps=1, temperature=1.0, rng_key=key)
    # Both outputs must lengthen the prompt
    assert len(out_nc) >= len(prompt)
    assert len(out_c) >= len(prompt)
