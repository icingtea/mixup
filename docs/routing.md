# Routing

How the cluster decides *which node runs which experts*, and how that decision is
validated. All of this lives in `mixup_core.topology`.

## The LUT (lookup table)

`build_lut(config)` turns the explicit `config.experts` lists into a mapping from a
global expert id to its home:

```
lut: dict[expert_id] = (container_name, local_slot)
```

It is built directly from the config:

```python
for container, experts in config.experts.items():
    for slot, expert in enumerate(experts):
        lut[expert] = (container, slot)
```

`local_slot` is simply the expert's *position* in that container's list. For the
default config:

| Expert | Container | Local slot |
|---|---|---|
| 0 … 7 | `container_2` | 0 … 7 |
| 8 … 15 | `container_3` | 0 … 7 |
| 16 … 23 | `container_4` | 0 … 7 |
| 24 … 31 | `container_5` | 0 … 7 |

> **Why not `// %`?** An earlier prototype derived the mapping with
> `container = experts // (E/N)` and `slot = experts % (E/N)`, which *ignored* the
> explicit lists and only happened to match. Deriving the LUT from the config's own
> lists means the routing table can never silently disagree with the config — the
> config is the single source of truth.

## Validation (`validate_partition`)

Because the distributed forward pass depends on two invariants, the node validates the
partition at startup and fails loudly instead of mis-routing:

1. **Coverage** — every expert `0..E-1` is assigned to exactly one container.
2. **Contiguity** — each container's experts form one ascending, contiguous block.

Why each matters:

- **Coverage**: `ExpertGroup.forward` indexes `lut[e]` for every `e` in `0..E-1` to
  build the `mine`/`slot` masks. A missing expert would `KeyError`.
- **Contiguity**: `ExpertGroup._load_layer` reads each container's experts as a single
  `slice(first, last+1)` of the big safetensors file. Non-contiguous experts can't be
  read as one slice.

Coverage is checked against the model's `num_local_experts` (32), which only the node
knows (the core is model-agnostic).

## Who uses the LUT

- **Central** (`Dispatcher`): for a chosen expert `e`, `lut[e][0]` is the container to
  contact.
- **Expert** (`ExpertGroup`): builds `mine`/`slot` from `lut`, as described in
  [math.md](math.md).

Both roles are in `mixup-node`; the LUT itself is pure and lives in `mixup-core` so it
can be unit-tested and so the smoke test can run without torch.

## `SKIP` sentinel

`SKIP = 999` is a *node*-internal constant (it's a transformers runtime detail, so it
lives in `mixup_node.model`, not core). It marks "this chosen expert lives on another
container" in the local slot tensor. Any value `≥ g` works; 999 is arbitrary.

## `expert_containers` / `experts_for`

- `expert_containers(config)` — the expert container names, in config order (the
  dispatcher's worker-pool size comes from its length).
- `experts_for(lut, container)` — the sorted experts owned by a container (used by
  `ExpertGroup` to know its own block).
