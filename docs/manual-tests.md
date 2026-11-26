# Manual tests

Hands-on checks you can run after each phase, from the repo root. Each tells you what to
look for and, if it fails, what to fix. The test venv is set up once (see
[docs/development.md](development.md)):

```bash
nix shell nixpkgs#python313 --command bash -lc '
  export UV_PYTHON=$(command -v python3)
  uv venv /tmp/mixup-test-venv
  uv pip install --python /tmp/mixup-test-venv/bin/python -q pytest pydantic fastapi httpx
'
```

---

## Phase 3 — leader election (pure logic)

The election decision is torch-free, so the manual test is the unit suite:

```bash
PYTHONPATH="packages/mixup-core/src:packages/mixup-node/src" \
  /tmp/mixup-test-venv/bin/python -m pytest tests/test_election.py -v
```

**Expected:** 10 tests pass. They cover `elect` (freshest `kv_seq` → `priority` → `node_id`),
`current_leader`, `should_become_leader`, and `promote` (role + epoch bump).

**If it fails:** the election ranking or the promote/demote transition is wrong. Check
`mixup_core/election.py` — the rank key is `(kv_seq, priority, node_id)`, and `promote`
must demote any other `LEADER` and bump the epoch to `max + 1`.

The *cluster-visible* leader failover (stop the leader, watch a follower take over and
resume generation) is the **Phase 5 manual test**; Phase 3 only provides the decision logic.

---

## Phase 4 — leader failover (restart from prompt)

Rebuild + start the cluster, then kill the leader mid-generation:

```bash
docker rm -f container_1 container_2 container_3 container_4 container_5 2>/dev/null
mixup-orchestrator
docker logs -f container_2    # a follower, second terminal
# once the central starts generating, in a third terminal:
docker stop container_1
```

**Expected:** `container_2` (the highest-priority follower) logs, then produces a
completion (Phase 4 restarts from the prompt — no resume yet):

```
WARNING mixup_node.node: leader container_1 appears down; checking election
INFO mixup_node.node: container_2: I win the election; promoting to leader
INFO mixup_node.node: container_2: shed experts, becoming leader
INFO mixup_node.node: container_2: dense model loaded; I am leader
INFO mixup_node.infer: generated: ...
```

**If it fails:**
- No `leader ... appears down` log → the watchdog isn't polling. Check `Node._watchdog`
  (interval, `current_leader`) and that the leader serves `/health`.
- `I win the election` on more than one follower → split-brain. Check `elect` /
  `should_become_leader` ranking.
- Promotion crashes around "dense model loaded" → `load_central` / `Node._become_leader`.
- Followers never reload the shed experts → the leader's `broadcast_routing` or the
  followers' `POST /routing` is broken.

Known limitation (single-failure only): the other followers don't learn the *new* leader's
identity (they spin idle with a dead leader in their view), so a second leader failure
isn't handled yet — that's the gossip-leader-announcement work, not in Phase 4.

---

## Phase 5 — leader failover with resume

Same drill as Phase 4, but now the new leader resumes instead of restarting:

```bash
docker rm -f container_1 container_2 container_3 container_4 container_5 2>/dev/null
mixup-orchestrator
docker logs -f container_2
# once the central starts generating, in another terminal:
docker stop container_1
```

**Expected:** `container_2` logs `resuming from checkpoint: kv_seq=N, M tokens left`
(not `generated:` from scratch), then produces a completion. The text should match an
uninterrupted run — greedy decoding is deterministic; resume recomputes the prefix but
loses ≤1 token's work.

**If it fails:**
- No `resuming from checkpoint` → the checkpoint wasn't written or the ledger volume isn't
  mounted. Check `var/ledger/checkpoint.json` exists on the host once the leader starts.
- Always `kv_seq=0` → the per-token writer isn't running. Check `_CheckpointProcessor` is
  passed to `model.generate` and the ledger dir is writable (`var/ledger` is rw-mounted).
- Wrong resume text → the `prompt_len`/`kv_seq` math is off. Check
  `prompt_len = len(input_ids) - kv_seq`.

---
