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


def test_openai_shape_and_dynamic_batching(tiny_model):
    from concurrent.futures import ThreadPoolExecutor

    from fastapi.testclient import TestClient

    from sciagent.serving.hf_server import create_app

    model, tok = tiny_model
    app = create_app(model, tok, "tiny", max_batch=4, window_ms=200)
    c = TestClient(app)
    assert c.get("/v1/models").json()["data"][0]["id"] == "tiny"

    def call(i):
        return c.post("/v1/chat/completions", json={"model": "tiny", "max_tokens": 8, "temperature": 0,
                                                     "messages": [{"role": "user", "content": f"plan batch {i}"}]}).json()
    with ThreadPoolExecutor(4) as pool:
        outs = list(pool.map(call, range(4)))
    for o in outs:
        assert o["choices"][0]["message"]["role"] == "assistant"
        assert o["usage"]["prompt_tokens"] > 0 and 0 <= o["usage"]["completion_tokens"] <= 8
    stats = c.get("/metrics").json()
    assert stats["requests"] == 4
    assert max(int(k) for k in stats["batch_sizes"]) > 1, "concurrent requests should be batched"


def test_batched_greedy_equals_single(tiny_model):
    """Left-padding must not change greedy outputs (attention mask and positions are handled)."""
    from sciagent.serving.hf_server import BatchingEngine

    model, tok = tiny_model
    eng1 = BatchingEngine(model, tok, max_batch=1, window_ms=1)
    eng4 = BatchingEngine(model, tok, max_batch=4, window_ms=200)
    msgs = [[{"role": "user", "content": "x " * (i + 1)}] for i in range(4)]
    single = [eng1.submit(m, 6, 0.0).result(timeout=60)[0] for m in msgs]
    futs = [eng4.submit(m, 6, 0.0) for m in msgs]
    batched = [f.result(timeout=60)[0] for f in futs]
    assert single == batched
