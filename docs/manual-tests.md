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

## Phase 4 — leader failover

Start the cluster, trigger a generation with the REPL, and kill the leader mid-request:

```bash
docker rm -f container_1 container_2 container_3 container_4 container_5 2>/dev/null
mixup-orchestrator
docker logs -f container_2                                  # a follower, second terminal
mixup-repl "What is mixture of experts?" --max-new-tokens 100   # third terminal
# while the REPL is still waiting, stop the leader:
docker stop container_1
```

**Expected:** `container_2` (the highest-priority follower) promotes:

```
WARNING mixup_node.node: leader container_1 appears down; checking election
INFO mixup_node.node: container_2: I win the election; promoting to leader
INFO mixup_node.node: container_2: shed experts, becoming leader
INFO mixup_node.node: container_2: dense model loaded; I am leader
```

The new leader does **not** auto-generate — it loads the dense model and waits for the
next `/generate`. Re-run `mixup-repl "..."` to confirm it now serves requests.

**If it fails:**
- No `leader ... appears down` log → the watchdog isn't polling. Check `Node._watchdog`
  (interval, `current_leader`) and that the leader serves `/health`.
- `I win the election` on more than one follower → split-brain. Check `elect` /
  `should_become_leader` ranking.
- Promotion crashes around "dense model loaded" → `load_central` / `Node._become_leader`.
- Followers never reload the shed experts → the leader's `broadcast_routing` or the
  followers' `POST /routing` is broken.

The new leader broadcasts its identity to the other followers (`POST /leader`), so they
poll *it* and can survive a *second* leader failure.

---

## Phase 5 — resume on leader failover (deferred)

Removed with on-demand generation: the leader no longer boot-generates, so there is no
automatic "resume from checkpoint" on promotion. The promoted leader just waits for the
next `/generate`, and a client re-sends its prompt after re-discovering the leader.

The checkpoint write is retained (each `/generate` still records `input_ids` to
`var/ledger`), so a REPL-driven resume (re-discover + resume the partial generation) can
be added later if "minimal loss" matters again.

---

## Phase 9 — REPL client (`mixup-repl`)

With the cluster running, prompt it from the host (no need to read docker logs):

```bash
mixup-repl "What is mixture of experts?" --max-new-tokens 20
```

**Expected:** it discovers the leader and prints a completion.

**If it fails:**
- `no leader found` → the cluster isn't running, or the port mapping is off. Check
  `docker ps` (leader on `localhost:9000`) and `node_urls` in `mixup_repl/repl.py`.
- `error talking to http://localhost:9000` → the leader's `/generate` returned non-2xx.
  Check `docker logs container_1` (the leader must have its dense model loaded; a
  follower returns 503).

---
