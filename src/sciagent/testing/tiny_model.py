"""Build a tiny, randomly initialised Llama-style chat model locally (no downloads).

Used by the test suite and by the ``cpu-smoke`` experiment profile to exercise training and serving code
paths end to end. Its outputs are meaningless text.
"""

from __future__ import annotations

from pathlib import Path


def make_tiny_model(out_dir: str | Path, texts: list[str] | None = None, vocab: int = 2000, seed: int = 0) -> str:
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    out = Path(out_dir)
    if (out / "config.json").exists():
        return str(out)
    texts = texts or ['{"reasoning_summary": "explore", "stop": false, "runs": [{"params": {"T": 320.0}, '
                      '"fidelity": "rk4_medium", "why": "refine"}]}', "STATE_JSON plan the next batch"] * 100
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel()
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=vocab, special_tokens=["<unk>", "<pad>", "</s>"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>", eos_token="</s>")
    tok.chat_template = ("{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}\n{% endfor %}"
                         "{% if add_generation_prompt %}<assistant>{% endif %}")
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=16384,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    LlamaForCausalLM(cfg).save_pretrained(out)
    tok.save_pretrained(out)
    return str(out)
