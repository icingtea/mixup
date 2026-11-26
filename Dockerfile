FROM python:3.13

# uv for workspace-aware installs (mixup-node depends on the mixup-core member).
RUN pip install --no-cache-dir uv

WORKDIR /usr/local/app

COPY pyproject.toml uv.lock ./
COPY packages ./packages

# Install mixup-node and its workspace dependency mixup-core, frozen to uv.lock,
# skipping the dev group (ruff/pytest) and the orchestrator (docker).
# The uv cache mount persists downloaded wheels across builds, so code-only
# changes re-run `uv sync` without re-downloading torch.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package mixup-node

ENV PATH="/usr/local/app/.venv/bin:$PATH"

EXPOSE 8000

RUN useradd app
USER app

CMD ["mixup-node"]
