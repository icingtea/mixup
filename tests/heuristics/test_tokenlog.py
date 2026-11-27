"""Tests for the commit log and quorum logic."""

import pytest

from mixup_core.tokenlog import CommitLog, CommitRecord, CommitTracker, quorum_size


def _rec(index: int, token: int = 0, term: int = 1, kv_seq: int = 0) -> CommitRecord:
    return CommitRecord(term=term, index=index, token_id=token, kv_seq=kv_seq)


def test_quorum_size_is_majority() -> None:
    assert quorum_size(1) == 1
    assert quorum_size(3) == 2
    assert quorum_size(5) == 3
    assert quorum_size(4) == 3


def test_append_and_next_index() -> None:
    log = CommitLog()
    assert log.next_index == 1
    log.append(_rec(1))
    log.append(_rec(2))
    assert log.next_index == 3


def test_append_rejects_non_monotonic() -> None:
    log = CommitLog()
    log.append(_rec(2))
    with pytest.raises(ValueError):
        log.append(_rec(1))


def test_advance_never_regresses() -> None:
    log = CommitLog()
    log.advance(5)
    log.advance(3)
    assert log.committed == 5


def test_committed_prefix_filters() -> None:
    log = CommitLog()
    for i in range(1, 5):
        log.append(_rec(i))
    log.advance(2)
    assert [r.index for r in log.committed_prefix()] == [1, 2]


def test_truncate_to_drops_tail_and_clamps_watermark() -> None:
    log = CommitLog()
    for i in range(1, 5):
        log.append(_rec(i))
    log.advance(4)
    log.truncate_to(2)
    assert [r.index for r in log.records] == [1, 2]
    assert log.committed == 2


def test_empty_log_prefix() -> None:
    assert CommitLog().committed_prefix() == []


def test_tracker_commits_on_majority_ack() -> None:
    tracker = CommitTracker(num_nodes=5)  # quorum 3
    tracker.append(term=1, token_id=10, kv_seq=0)
    tracker.append(term=1, token_id=11, kv_seq=1)
    # self + two peers ack token 1 -> committed
    assert tracker.record_ack(1, "self") == 0
    assert tracker.record_ack(1, "a") == 0
    assert tracker.record_ack(1, "b") == 1  # majority reached
    assert tracker.log.committed == 1


def test_tracker_requires_contiguous_commits() -> None:
    tracker = CommitTracker(num_nodes=3)  # quorum 2
    tracker.append(term=1, token_id=10, kv_seq=0)
    tracker.append(term=1, token_id=11, kv_seq=1)
    # ack token 2 to a majority, but token 1 isn't committed -> watermark stays 0.
    tracker.record_ack(2, "self")
    tracker.record_ack(2, "a")
    assert tracker.log.committed == 0


def test_tracker_ack_count() -> None:
    tracker = CommitTracker(num_nodes=5)
    tracker.append(term=1, token_id=10, kv_seq=0)
    tracker.record_ack(1, "self")
    tracker.record_ack(1, "a")
    assert tracker.ack_count(1) == 2
