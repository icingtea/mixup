"""The commit log: the replicated ledger that makes token emission safe under failover.

Phase 5 (commit quorum, not Raft election). The leader appends each generated token to a
log and replicates it to followers; a token is *committed* (safe to emit) once a majority
has acked it. A monotonic ``committed`` watermark is the resume point: a successor leader
emits nothing below it and never re-emits a committed token.

Pure pydantic, torch-free, so the quorum logic is unit-testable.
"""

from pydantic import BaseModel

from .types import Epoch


def quorum_size(n: int) -> int:
    """Number of acks needed for a commit decision: a strict majority of ``n`` voters."""
    return n // 2 + 1


class CommitRecord(BaseModel):
    """One generated token and the state needed to resume after it."""

    term: Epoch
    """The leadership epoch this token was generated under."""

    index: int
    """Monotonic token index (1-based) within the request."""

    token_id: int
    """The token id produced at this step."""

    kv_seq: int
    """The global KV-cache sequence at this step (for warm resume)."""


class CommitLog(BaseModel):
    """An append-only token log with a monotonic committed watermark."""

    records: list[CommitRecord] = []
    committed: int = 0
    """Highest committed index (0 = nothing committed)."""

    def append(self, record: CommitRecord) -> None:
        """Append a record, enforcing a monotonic index (later than the last)."""
        if self.records and record.index <= self.records[-1].index:
            raise ValueError(
                f"non-monotonic append: index {record.index} <= {self.records[-1].index}"
            )
        self.records.append(record)

    def advance(self, index: int) -> None:
        """Advance the committed watermark (never regresses)."""
        self.committed = max(self.committed, index)

    def committed_prefix(self) -> list[CommitRecord]:
        """The records up to and including the committed watermark, in order."""
        return [r for r in self.records if r.index <= self.committed]

    def truncate_to(self, index: int) -> None:
        """Drop records beyond ``index`` (a successor resumes from here)."""
        self.records = [r for r in self.records if r.index <= index]
        self.committed = min(self.committed, index)

    @property
    def next_index(self) -> int:
        return (self.records[-1].index + 1) if self.records else 1


class CommitTracker:
    """Tracks per-token acks and advances the committed watermark on majority-ack.

    The leader appends each generated token and records acks (including its own) as
    followers confirm replication. The watermark advances to the highest *contiguous*
    index with a majority ack — a token is only committed once every token before it is.
    """

    def __init__(self, num_nodes: int, log: CommitLog | None = None) -> None:
        self.num_nodes = num_nodes
        self.quorum = quorum_size(num_nodes)
        self.log = log or CommitLog()
        self._acks: dict[int, set[str]] = {}

    def append(self, term: int, token_id: int, kv_seq: int) -> CommitRecord:
        record = CommitRecord(
            term=term, index=self.log.next_index, token_id=token_id, kv_seq=kv_seq
        )
        self.log.append(record)
        return record

    def record_ack(self, index: int, node_id: str) -> int:
        """Record an ack from ``node_id`` for ``index``; return the new watermark."""
        self._acks.setdefault(index, set()).add(node_id)
        return self._advance()

    def record_acks_through(self, index: int, node_id: str) -> int:
        """Record acks for indices 1..``index`` (a follower stored the whole log)."""
        for i in range(1, index + 1):
            self._acks.setdefault(i, set()).add(node_id)
        return self._advance()

    def _advance(self) -> int:
        while (
            self.log.committed + 1
        ) in self._acks and len(self._acks[self.log.committed + 1]) >= self.quorum:
            self.log.advance(self.log.committed + 1)
        return self.log.committed

    def ack_count(self, index: int) -> int:
        return len(self._acks.get(index, set()))
