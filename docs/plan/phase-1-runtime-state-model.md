# Phase 1 — Runtime state model (torch-free)

**Goal:** add the types and pure functions that represent the *live* cluster — membership,
the versioned routing table, gossip messages, election, scheduling, and the KV-ledger
header — all in `mixup-core`, with no torch/docker/fastapi imports.

This is the foundation every later phase builds on, and the reason it belongs in core is
the same as the existing `topology.py`: the logic is a pure function of inputs and must
be unit-testable on the host (see [testing.md](testing.md)).

## New modules

```
packages/mixup-core/src/mixup_core/
├── types.py          # + NodeId, Epoch, NodePriority, NodeRole, Incarnation
├── runtime.py        # MemberState, RoutingTable, ClusterView, RoutingEntry
├── gossip.py         # SWIM message types + membership state machine (pure)
├── election.py       # elect(...)  — freshness + priority
├── scheduler.py      # PlacementPolicy + rebalance_experts(...)
└── kvledger.py       # KVSeq, LedgerHeader codec (schema only; tensors live in node)
```

## Types (`types.py`)

```python
NodeId = str                 # container name, hostname on the Docker network
Epoch = int                  # leadership generation, monotonic
NodePriority = int           # election tie-break, higher wins
Incarnation = int            # boot counter, incremented each restart

class NodeRole(StrEnum):
    LEADER = "leader"
    EXPERT = "expert"        # follower that serves experts
```

`NodeRole` replaces the current `ContainerType`. A node's role is now **dynamic**
(an expert can be promoted to leader), so it is no longer a pure function of the config;
it lives in `MemberState` (below), not in `StartupConfig`.

## Membership & routing (`runtime.py`)

```python
class LoadMetrics(BaseModel):
    active_requests: int = 0          # in-flight expert requests
    queue_depth: int = 0              # queued, not yet started
    free_slots: int = 0               # expert capacity headroom
    memory_headroom_bytes: int = 0    # reported by the node

class MemberState(BaseModel):
    node_id: NodeId
    role: NodeRole
    epoch: Epoch
    incarnation: Incarnation
    priority: NodePriority
    owned_experts: list[ExpertId]
    loads: LoadMetrics
    kv_seq: int = 0                   # freshest ledger entry this node holds

class RoutingEntry(BaseModel):
    node_id: NodeId
    slot: int                         # local slot on that node

class RoutingTable(BaseModel):
    epoch: Epoch
    assignment: dict[ExpertId, RoutingEntry]   # the live LUT, versioned
```

`RoutingTable` is the **versioned replacement** for today's `build_lut` dict. It carries
an `epoch` so a stale table can never overwrite a newer one (see Phase 3), and a version
comparison function `RoutingTable.is_newer_than(other) -> bool` based on `epoch` (with a
tie-break hash for determinism).

`ClusterView` is the aggregate a node maintains: `{members: dict[NodeId, MemberState],
routing: RoutingTable}`.

## Gossip message types & SWIM state machine (`gossip.py`)

Pure logic only — transport is Phase 2. Message types:

```python
class Ping(BaseModel):      node_id, incarnation, epoch, kv_seq
class Ack(BaseModel):       node_id, incarnation, epoch, kv_seq, routing_epoch
class PingReq(BaseModel):   from_node, target, incarnation
class Suspect(BaseModel):   node_id, incarnation, by_node
class Alive(BaseModel):     node_id, incarnation, epoch
class Dead(BaseModel):      node_id, incarnation, by_node
```

The **state machine** is a pure function:

```python
class MemberStatus(StrEnum): ALIVE, SUSPECT, DEAD

def apply_gossip(view: ClusterView, msg: GossipMsg) -> ClusterView: ...
```

`apply_gossip` encodes the SWIM transitions (alive → suspect → dead, and the resurrection
rule: a higher-incarnation `Alive` revives a suspect/dead node). Keeping this pure means
the entire failure-detection protocol is testable as a table of `(initial_view, msg,
expected_view)` cases with no network. Methodology references: [SWIM], [Serf-gossip].

## Election (`election.py`)

```python
def elect(view: ClusterView) -> NodeId | None:
    """Winner among alive members: highest kv_seq, then priority, then node_id."""
```

Pure and deterministic. The **freshness-first** ordering is the "ledger as election
credential" decision (see [phase-3-leader-election.md](phase-3-leader-election.md)); it
mirrors Raft's election restriction that the candidate with the most up-to-date log wins
([Raft] §5.4.1). Returns `None` if no member is alive (or if only the current leader is
alive and unchanged).

## Scheduler (`scheduler.py`)

The long-term home of "dynamic dispatch based on heuristics":

```python
class PlacementPolicy(Protocol):
    def place(self, view: ClusterView, experts: set[ExpertId]) -> RoutingTable: ...

def rebalance_experts(
    view: ClusterView,
    prev: RoutingTable,
    policy: PlacementPolicy,
) -> RoutingTable:
    """Recompute assignment given membership + loads; bump epoch."""
```

`rebalance_experts` is the pure core: given the current members and their loads, produce
a new assignment that (a) covers every expert, (b) respects each node's capacity, and
(c) minimizes churn (moves few experts). Failover is the degenerate case (a member is
`DEAD`, its experts are unowned, so they must move). See
[phase-6-dynamic-dispatch.md](phase-6-dynamic-dispatch.md) for the policies.

## KV-ledger header (`kvledger.py`)

The *schema* of the replicated KV cache, minus the tensor bytes (which stay in node-land
via safetensors):

```python
class LedgerHeader(BaseModel):
    kv_seq: int
    epoch: Epoch
    input_ids: list[int]            # prompt + generated tokens so far
    position_ids: list[int]
    cache_position: list[int]
    attention_mask: list[int]
    layer_shapes: list[tuple[int, ...]]   # (num_kv_heads, head_dim) per layer

def encode_header(h) -> bytes / decode_header(b) -> LedgerHeader
```

The header is what travels in gossip (cheap); the actual `past_key_values` tensors are
pulled/pushed over the data path (Phase 5). Keeping the codec here means the format is a
single, testable source of truth.

## Methodology — why "pure logic in core"

This follows the existing ADR-002 (`mixup-core` is torch-free). The payoff is amplified
here because distributed-systems logic is *exactly* the kind of code that needs
exhaustive unit testing with deterministic simulations (message reordering, partitions,
crashes) — none of which requires torch, Docker, or a GPU. See [testing.md](testing.md)
for the simulation harness. References: [Lamport] (logical clocks/ordering), [Demers]
(gossip convergence), [SWIM], [Raft].

## Acceptance criteria

- All new types import with `pydantic` only; `mixup-core` still has no torch/docker.
- `apply_gossip`, `elect`, `rebalance_experts`, and the ledger codec have unit tests.
- `mixup-core`'s public surface is exported from `__init__.py` with `__all__`.

## Risks

- **Over-modeling.** Pydantic models for every message can become verbose. Mitigation:
  keep messages minimal (only fields actually read), and let `ClusterView` be the single
  aggregate rather than proliferating sub-views.
- **Versioning semantics.** `is_newer_than` must be total and transitive or rebalances
  will flap. Mitigation: epoch is the primary key; the hash is only a deterministic
  tie-break for identical epochs (which should be rare).
