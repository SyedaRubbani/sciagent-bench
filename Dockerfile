# CPU image: API + viewer + all planners + surrogate tooling.
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu ".[service,surrogate,mcp,plot]"
COPY studies ./studies
RUN useradd -m sciagent && mkdir -p /app/runs && chown sciagent /app/runs
USER sciagent
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["sciagent", "serve", "--host", "0.0.0.0", "--port", "8000", "--runs-dir", "/app/runs"]
