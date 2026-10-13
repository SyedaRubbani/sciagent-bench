"""A minimal OpenAI-compatible Chat Completions server backed by the deterministic mock planner.

Lets CI exercise the *real* HTTP path of the LLM planner (request shape, auth header, usage accounting,
retries) without a GPU or an API key:

    python -m sciagent.testing.fake_openai --port 8009
    # planner: {kind: llm, llm: {provider: openai, base_url: "http://127.0.0.1:8009/v1", model: fake}}
"""

from __future__ import annotations

import argparse
import threading
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..llm.providers import MockProvider


def create_app(fail_every: int = 0) -> FastAPI:
    app = FastAPI()
    mock = MockProvider()
    state = {"n": 0}

    @app.get("/v1/models")
    def models():
        return {"data": [{"id": "fake", "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        state["n"] += 1
        if fail_every and state["n"] % fail_every == 0:
            return JSONResponse({"error": "overloaded"}, status_code=503)  # exercises client retries
        msgs = [m for m in body["messages"] if m["role"] != "system"]
        r = mock.complete("", msgs, body.get("temperature", 0.0), body.get("max_tokens", 1000))
        return {"id": f"cmpl-{state['n']}", "object": "chat.completion", "created": int(time.time()),
                "model": body["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": r.text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": r.input_tokens, "completion_tokens": r.output_tokens,
                          "total_tokens": r.input_tokens + r.output_tokens}}

    app.state.calls = state
    return app


def serve_in_thread(port: int = 0, fail_every: int = 0):
    """Start the server on a free port; returns (base_url, server). Call server.should_exit = True to stop."""
    import socket

    import uvicorn

    if port == 0:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(fail_every), host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}/v1", server


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8009)
    ap.add_argument("--fail-every", type=int, default=0)
    a = ap.parse_args()
    uvicorn.run(create_app(a.fail_every), host="127.0.0.1", port=a.port)
