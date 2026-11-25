# mixup — documentation

Distributed inference for Mixture-of-Experts (MoE) language models. `mixup` takes an MoE
model, assigns **Docker containers as nodes**, gives each node a slice of the model's
**experts** (not layers), and runs a distributed forward pass where the central node
fans expert work out over HTTP and sums the replies.

Target model: **`ibm-granite/granite-3.1-1b-a400m-instruct`** — 24 layers, 32 experts
per layer, top-8 routing, ~400M active parameters.

## Reading guide

| Document | What it covers |
|---|---|
| [architecture.md](architecture.md) | The system: nodes, roles, data flow, lifecycle |
| [math.md](math.md) | The MoE mathematics, end to end |
| [routing.md](routing.md) | How experts are assigned to nodes (the LUT) and validated |
| [config.md](config.md) | The cluster config schema, validation, environment variables |
| [protocol.md](protocol.md) | The HTTP wire protocol between central and experts |
| [packages.md](packages.md) | The three packages and their dependency boundaries |
| [decisions.md](decisions.md) | Every design decision and its rationale (ADRs) |
| [nix-and-uv.md](nix-and-uv.md) | uv2nix, the dev shell, and the lockfile |
| [running.md](running.md) | Setup, running, the smoke test, troubleshooting |
| [development.md](development.md) | Testing, linting, conventions |

## The idea in one paragraph

A dense transformer runs every layer on every token. An MoE transformer replaces the
feed-forward network (FFN) of each layer with **E experts** (small FFNs), plus a
**router** that, per token, picks the **k** most relevant experts and averages their
outputs. `mixup` exploits the fact that *only the chosen experts run*: instead of one
machine holding all 32 experts, the experts are partitioned across containers. The
central container holds the *dense* parts (embeddings, attention, router, norms) and,
after the router picks experts for a token, sends the hidden state to the containers
that hold those experts. Each expert container runs *only its own* experts and returns
a partial sum; the central adds the partials. See [math.md](math.md).
