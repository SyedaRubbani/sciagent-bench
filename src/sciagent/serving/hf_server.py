"""OpenAI-compatible Chat Completions server on Hugging Face transformers, with dynamic batching.

A portable fallback for GPUs or environments where vLLM cannot be installed (e.g. older Kaggle/Colab GPUs).
Requests that arrive within ``--batch-window-ms`` of each other are left-padded and generated together, which
is the simplest form of request batching; vLLM adds paged KV-cache and continuous batching on top, and the
notebook measures the difference.

    python -m sciagent.serving.hf_server --model Qwen/Qwen2.5-0.5B-Instruct --port 8002 [--load-in-4bit] [--lora DIR]
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
import uuid
from concurrent.futures import Future

from fastapi import FastAPI, Request


class BatchingEngine:
    def __init__(self, model, tokenizer, max_batch: int = 8, window_ms: float = 30.0, max_model_len: int = 16384):
        self.model, self.tok = model, tokenizer
        self.max_batch, self.window = max_batch, window_ms / 1000.0
        self.max_model_len = max_model_len
        self.q: queue.Queue = queue.Queue()
        self.stats = {"requests": 0, "batches": 0, "prompt_tokens": 0, "completion_tokens": 0, "batch_sizes": {},
                      "generate_s": 0.0}
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, messages: list[dict], max_tokens: int, temperature: float) -> Future:
        f: Future = Future()
        self.q.put((messages, max_tokens, temperature, f))
        return f

    def _loop(self):
        while True:
            batch = [self.q.get()]
            deadline = time.perf_counter() + self.window
            while len(batch) < self.max_batch:
                left = deadline - time.perf_counter()
                if left <= 0:
                    break
                try:
                    batch.append(self.q.get(timeout=left))
                except queue.Empty:
                    break
            try:
                self._run(batch)
            except Exception as e:  # surface errors to every caller in the batch
                for *_, f in batch:
                    if not f.done():
                        f.set_exception(e)

    def _run(self, batch):
        import torch

        prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m, *_ in batch]
        enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_model_len,
                       add_special_tokens=False).to(self.model.device)
        max_new = max(b[1] for b in batch)
        temp = max(b[2] for b in batch)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=max_new, do_sample=temp > 0,
                                      temperature=temp if temp > 0 else None, top_p=None, top_k=None,
                                      pad_token_id=self.tok.pad_token_id)
        self.stats["generate_s"] += time.perf_counter() - t0
        n_in = enc["input_ids"].shape[1]
        self.stats["batches"] += 1
        self.stats["batch_sizes"][len(batch)] = self.stats["batch_sizes"].get(len(batch), 0) + 1
        for i, (_m, mt, _t, f) in enumerate(batch):
            gen = out[i, n_in:][:mt]
            eos = self.tok.eos_token_id
            ids = gen.tolist()
            if eos is not None and eos in ids:
                ids = ids[: ids.index(eos)]
            pad = self.tok.pad_token_id
            ids = [t for t in ids if t != pad]
            text = self.tok.decode(ids, skip_special_tokens=True)
            p_tok = int(enc["attention_mask"][i].sum())
            self.stats["requests"] += 1
            self.stats["prompt_tokens"] += p_tok
            self.stats["completion_tokens"] += len(ids)
            f.set_result((text, p_tok, len(ids)))


def create_app(model, tokenizer, served_name: str, **engine_kw) -> FastAPI:
    import asyncio

    eng = BatchingEngine(model, tokenizer, **engine_kw)
    app = FastAPI(title="sciagent hf_server")

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": served_name, "object": "model"}]}

    @app.get("/metrics")
    def metrics():
        return eng.stats

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        fut = eng.submit(body["messages"], int(body.get("max_tokens") or 512), float(body.get("temperature") or 0.0))
        text, p, c = await asyncio.wrap_future(fut)
        return {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()),
                "model": served_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}}

    app.state.engine = eng
    return app


def load(model_id: str, dtype: str = "auto", load_in_4bit: bool = False, lora: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cuda = torch.cuda.is_available()
    if dtype == "auto":
        td = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if cuda else torch.float32
    else:
        td = getattr(torch, dtype)
    kw = {"device_map": "auto"} if cuda else {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=td)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=td, **kw)
    except TypeError:  # transformers < 4.56 uses torch_dtype
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=td, **kw)
    if lora:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora)
    model.eval()
    return model, AutoTokenizer.from_pretrained(model_id)


def main():
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--served-model-name")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--lora")
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--batch-window-ms", type=float, default=30.0)
    ap.add_argument("--max-model-len", type=int, default=16384)
    a = ap.parse_args()
    model, tok = load(a.model, a.dtype, a.load_in_4bit, a.lora)
    app = create_app(model, tok, a.served_model_name or a.model, max_batch=a.max_batch, window_ms=a.batch_window_ms,
                     max_model_len=a.max_model_len)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
