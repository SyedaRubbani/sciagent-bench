# Security

- The HTTP API binds to `127.0.0.1` by default. Before exposing it, set `SCIAGENT_API_TOKEN`; every `/api` route except
  `/api/health` then requires `Authorization: Bearer <token>`. Put it behind TLS (a reverse proxy) on shared networks.
- Study specs can name a simulator as `module:Class`, which imports Python code. Only accept specs from people you
  would let run code on the machine.
- API keys are read from environment variables named in the spec (`api_key_env`) and are never written to traces,
  cassettes or manifests.
- Report vulnerabilities privately to the maintainer (see `CITATION.cff`).
