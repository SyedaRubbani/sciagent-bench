"""The transformers fallback server, exercised with a tiny randomly initialised model (no downloads)."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    corpus = ["plan the next batch of simulations", '{"runs": [{"params": {"T": 320}, "fidelity": "rk4_medium"}]}',
              "explore cheap fidelity then confirm at top fidelity"] * 50
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel()
    tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=300, special_tokens=["<unk>", "<pad>", "</s>"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>", eos_token="</s>")
    tok.chat_template = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}assistant:"
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    return LlamaForCausalLM(cfg).eval(), tok


def test_openai_shape(tiny_model):
    from fastapi.testclient import TestClient

    from sciagent.serving.hf_server import create_app

    model, tok = tiny_model
    c = TestClient(create_app(model, tok, "tiny", max_batch=4, window_ms=50))
    assert c.get("/v1/models").json()["data"][0]["id"] == "tiny"
    o = c.post("/v1/chat/completions", json={"model": "tiny", "max_tokens": 8, "temperature": 0,
                                             "messages": [{"role": "user", "content": "plan batch"}]}).json()
    assert o["choices"][0]["message"]["role"] == "assistant"
    assert o["usage"]["prompt_tokens"] > 0 and 0 <= o["usage"]["completion_tokens"] <= 8
    assert c.get("/metrics").json()["requests"] == 1


def test_requests_queued_together_are_batched(tiny_model):
    """Requests already waiting in the queue are generated as one batch (no timing race: the engine's
    worker is held until all four are queued)."""
    from sciagent.serving.hf_server import BatchingEngine

    model, tok = tiny_model
    eng = BatchingEngine(model, tok, max_batch=4, window_ms=500)
    futs = [eng.submit([{"role": "user", "content": f"plan batch {i}"}], 4, 0.0) for i in range(4)]
    outs = [f.result(timeout=120) for f in futs]
    assert all(isinstance(t, str) and p > 0 for t, p, _ in outs)
    assert max(eng.stats["batch_sizes"]) > 1


def test_left_padding_preserves_next_token_scores(tiny_model):
    """Left-padded batched scoring must match unbatched scoring (attention mask and positions handled).

    Compared as next-token logits with a tolerance rather than as generated text: with random weights,
    near-tied tokens can flip under harmless floating-point differences between CPUs."""
    model, tok = tiny_model
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = [tok.apply_chat_template([{"role": "user", "content": "x " * (i + 1)}], tokenize=False,
                                       add_generation_prompt=True) for i in range(4)]
    with torch.inference_mode():
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        batched = model(**enc, position_ids=pos).logits[:, -1, :]
        for i, p in enumerate(prompts):
            single = model(**tok([p], return_tensors="pt", add_special_tokens=False)).logits[:, -1, :]
            assert torch.allclose(batched[i], single[0], atol=1e-4, rtol=1e-4)
