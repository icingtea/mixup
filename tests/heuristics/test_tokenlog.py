"""Tests for the commit log and quorum logic."""

import pytest

from mixup_core.tokenlog import CommitLog, CommitRecord, quorum_size


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
