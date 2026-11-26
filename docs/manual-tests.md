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
