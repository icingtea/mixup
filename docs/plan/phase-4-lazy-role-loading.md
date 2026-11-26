# Phase 4 — Lazy role loading (minimal memory footprint)

**Goal:** enforce decision #1 — a node loads from the shared read-only `models/` mount
*only what its current role needs*, and releases it when the role changes. No node ever
pre-loads attention/dense layers; only the leader does, and only while it is leader.

## Why

The dense/attention part of the model (embeddings, 24× self-attention, routers, norms,
LM head) is a few hundred MB in bf16 — not free, and utterly wasted on a follower that
only serves experts. Loading it lazily on promotion, and unloading on demotion, keeps the
steady-state follower at its minimum (experts only + a few-MB KV-ledger blob).

This is the direct answer to the "minimal memory footprint" decision: **loading is a
function of role, and role changes at runtime.**

## Loading, concretely

The shared mount makes this cheap: every node can read any weight by `safe_open` slice
with no weight transfer (same mechanism as today's `ExpertGroup` and `load_central`).

New module `packages/mixup-node/src/mixup_node/roles.py`:

```python
class RoleManager:
    """Owns the node's current role and the modules loaded for it."""

    def become_leader(self) -> None:
        """Shed owned experts (a rebalance), then lazily load the dense model."""

    def become_expert(self) -> None:
        """Unload the dense model (free memory); (re)load the experts assigned to us."""
```

- `become_leader()` first **sheds** its owned experts: it drops its `ExpertGroup` and
  triggers a rebalance so survivors take those experts (Phase 6) — this is what keeps a
  node from holding experts + dense simultaneously. Then it builds the model with
  `ExpertSlot` stubs (as today's `load_central`) and loads the non-expert weights via
  `safe_open` — **only now**, on promotion.
- `become_expert()` drops the reference to the dense model and (on CUDA) calls
  `torch.cuda.empty_cache()` to return memory; the `ExpertGroup` is loaded for whatever
  runs the current routing table assigns.
- The dense model load time (~a second for a few hundred MB from a local mount) is
  part of the promotion path; it overlaps with rehydrating the KV ledger (Phase 5), so
  the promotion flow is: shed experts → load dense → rehydrate ledger → resume.

## Minimal footprint accounting

| Role | Loaded | Approx. (bf16) |
|---|---|---|
| Follower | its expert slices (runs) + KV-ledger blob | ~600 MB + ~5 MB |
| Leader | dense/attention/router/norms/LM head (+ no experts) | ~300–400 MB |

The leader holds **no experts** (it dispatches), mirroring today's `central` behavior.
Invariant #5 (never experts + dense simultaneously) is enforced by the **shed-first**
promotion order above, not by convention.

## Threading & lifecycle

The role change is coordinated with the `ExpertGroup` (loaded/unloaded with the role) and
the generation loop (Phase 5). A background asyncio task watches `ClusterView`:

- sees `role == LEADER` and `self is leader` → ensure dense model loaded, run/continue
  generation;
- sees a higher epoch → `become_expert()` (unload) and stop generating.

## Methodology notes

- **Lazy loading** is the same pattern transformers uses for `low_cpu_mem_usage` /
  `device_map="auto"` loading (weights read on demand); we hand-roll the tiny subset we
  need rather than pulling in `accelerate`. Reference: [HF-llm-optimization].
- **Memory release** on CUDA relies on dropping references + `empty_cache()`; on the
  current prototype the model is small enough that this is about discipline, not GC
  pressure. The `RoleManager` keeps an explicit `self._dense_model: GraniteMoeForCausalLM
  | None` so the reference is unambiguous.
- **Shared read-only mount** means a follower's `become_leader()` reads exactly the same
  `model.safetensors` the old leader read — no transfer, no cache coherency issue.

## Acceptance criteria

- A follower boots with only its experts loaded (asserted via a `--dump-memory` debug
  endpoint or `torch.cuda.memory_allocated`).
- Promote a follower: dense model loads, generation proceeds; demote: memory returns to
  the follower baseline.
- No node ever holds experts + dense simultaneously (assert invariant #5).

## Risks

- **Load latency on promotion.** A few hundred MB read from a local bind mount is
  ~O(1 s). Acceptable (failover is rare); if it matters, a future optimization is to
  keep the dense weights *mapped* (mmap) but not resident. Documented, not in scope.
- **Reference leaks.** A stray reference to the dense model defeats `empty_cache()`.
  Mitigation: `RoleManager` is the single owner; tests assert the allocator drops.
