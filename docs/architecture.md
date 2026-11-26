# Architecture

## Overview

`mixup` runs a Granite MoE model across a cluster of Docker containers on one Docker
bridge network. There are exactly two *roles*:

```
                    ┌─────────────────────────────────────────────┐
                    │                 central node                 │
                    │  (embeddings + attention + router + norms)   │
                    │                                             │
                    │   ExpertSlot(0)  ──┐  (dispatches layer 0)   │
                    │   ExpertSlot(1)  ──┤                          │
                    │       ...          │                          │
                    │   ExpertSlot(23) ──┘                          │
                    └───────────────┬─────┬─────┬─────┬────────────┘
                            HTTP    │     │     │     │   HTTP
                    ┌───────────────▼──┐ ┌▼───────────────▼──┐   ...
                    │  expert node     │ │  expert node      │
                    │  container_2     │ │  container_3      │
                    │  experts 0..7    │ │  experts 8..15    │
                    └──────────────────┘ └───────────────────┘
```

- **Central** (`container_1`): holds *everything except the experts* — token
  embeddings, self-attention, the routers, and the final norm/LM head. At each layer,
  the real experts module is replaced by an `ExpertSlot`, a stub that forwards to a
  dispatcher instead of computing locally.
- **Expert** (`container_2..5`): holds a contiguous block of experts (8 of 32) for
  **every** layer, and serves them over HTTP at `POST /<layer>`.

There is no layer-parallelism here: the *layers* execute sequentially on the central;
only the *expert* computation is distributed. This is deliberate — attention is dense
and small in a 1B model, whereas the experts dominate FLOPs and are embarrassingly
partitionable.

## The forward pass, step by step

For one token `x` at layer `l`:

1. **Attention + norm** run on the central (dense, not distributed).
2. **Router** runs on the central: it produces logits over all 32 experts, selects the
   top-8, and softmaxes them into weights `w`. See [math.md](math.md).
3. **Dispatch**: the central looks up which containers hold the chosen experts and
   sends `(x, top_k_index, top_k_weights)` to each such container **in parallel**
   (a `ThreadPoolExecutor` sized to the number of expert containers).
4. **Expert forward**: each contacted container runs *only the experts it owns* on the
   token, producing a partial sum `y_j = Σ w_i E_i(x)` over the experts it holds.
5. **Reduce**: the central sums the partials from every container it contacted:
   `y = Σ_j y_j`. Tokens that picked no expert on a given container get zero from it,
   so the sum is exact.

Steps 2–5 repeat for all 24 layers, and the whole thing repeats once per generated
token (autoregressive generation).

## Container lifecycle

### Startup (orchestrator → nodes)

1. `mixup-orchestrator` reads `config.json`.
2. It checks the model file exists (see [running.md](running.md)).
3. It builds the image (`Dockerfile`) — installing `mixup-core` + `mixup-node` with
   `uv sync` into the image.
4. It creates (or reuses) the `cluster` bridge network.
5. It starts one container per config entry, injecting a single env var:
   - `MIXUP_NODE_ENV` — a serialized `NodeEnv` (container name + cluster config +
     prompt + token budget), the one runtime contract between host and node.
6. The host `models/` directory is mounted read-only into every container at
   `/usr/local/app/models`.

### Node startup (inside a container)

1. Read `MIXUP_NODE_ENV` and deserialize it into a typed `NodeEnv`
   (`load_node_env()`).
2. Load the model's `config.json` (`GraniteMoeConfig.from_pretrained`).
3. Build the **LUT** (expert → container/slot) from `config.experts`.
4. `validate_partition` — fail fast if the expert assignment is incomplete or
   non-contiguous (see [routing.md](routing.md)).
5. Branch on `config.role(name)`:
   - **central** → build the full model on the `meta` device, swap each layer's
     experts for an `ExpertSlot`, load the non-expert weights, then wait for experts
     and generate.
   - **expert** → build an `ExpertGroup` (only this container's experts, for every
     layer) and serve `POST /<layer>`.

### The model is loaded *sliced*, not whole

`mixup-node` never calls `AutoModelForCausalLM.from_pretrained` on the weights. It
opens `model.safetensors` with `safe_open` and reads **slices**:

- The **central** reads every key *except* `input_linear` / `output_linear` (the
  experts), then `load_state_dict(..., strict=False, assign=True)`.
- Each **expert** reads only `model.layers.<l>.block_sparse_moe.{input,output}_linear.weight`
  and takes a contiguous slice `[first_expert : last_expert+1]` per layer.

This is what makes the partition cheap: no node ever materializes the whole model.
