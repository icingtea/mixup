FROM python:3.13

# uv for workspace-aware installs (mixup-node depends on the mixup-core member).
RUN pip install --no-cache-dir uv

WORKDIR /usr/local/app

# ---------------------------------------------------------------------------
# Dependency layer.
# Copy ONLY the metadata needed to install third-party dependencies — no source
# code. This is the expensive layer (torch + the nvidia/CUDA toolchain land
# here), and it is invalidated only when uv.lock or a manifest changes, never by
# a code edit. The uv cache mount persists the downloaded wheels, so even a
# real dependency change re-runs this layer without re-downloading.
# ---------------------------------------------------------------------------
COPY pyproject.toml uv.lock ./
COPY packages/mixup-core/pyproject.toml packages/mixup-core/pyproject.toml
COPY packages/mixup-node/pyproject.toml packages/mixup-node/pyproject.toml
COPY packages/mixup-orchestrator/pyproject.toml packages/mixup-orchestrator/pyproject.toml
COPY packages/mixup-repl/pyproject.toml packages/mixup-repl/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package mixup-node \
        --no-install-workspace --no-install-project

# ---------------------------------------------------------------------------
# Application layer.
# Copy the (tiny) workspace source and install just those packages. The
# third-party environment above is already cached, so this is fast on every
# code change — no torch reinstall.
# ---------------------------------------------------------------------------
COPY packages ./packages

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package mixup-node

ENV PATH="/usr/local/app/.venv/bin:$PATH"

# All containers share one GPU via CDI. Expandable segments avoid allocator
# fragmentation across repeated expert reloads, which otherwise OOMs the
# shared GPU (CUBLAS_STATUS_ALLOC_FAILED / torch.OutOfMemoryError).
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPOSE 8000

RUN useradd app
USER app

CMD ["mixup-node"]
