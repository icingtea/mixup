"""Tests for the SWIM failure-detector timing logic (pure, time-injected)."""

from mixup_core.failure_detection import Action, FailureDetector


def make_detector() -> FailureDetector:
    return FailureDetector(ping_timeout=1.0, suspect_timeout=3.0)


def test_direct_timeout_requests_indirect_probe() -> None:
    d = make_detector()
    d.ping("b", now=0.0)
    assert d.expire("b", now=1.0) is Action.PINGREQ


def test_indirect_timeout_marks_suspect() -> None:
    d = make_detector()
    d.ping("b", now=0.0)
    assert d.expire("b", now=1.0) is Action.PINGREQ  # direct failed
    assert d.expire("b", now=2.0) is Action.SUSPECT  # indirect failed


def test_suspect_becomes_dead_after_suspect_timeout() -> None:
    d = make_detector()
    d.ping("b", now=0.0)
    d.expire("b", now=1.0)  # PINGREQ
    d.expire("b", now=2.0)  # SUSPECT (suspect since 2.0)

    assert d.expire("b", now=4.9) is Action.WAIT  # 2.9s < 3.0s
    assert d.expire("b", now=5.0) is Action.DEAD  # exactly 3.0s


def test_ack_clears_pending_probe() -> None:
    d = make_detector()
    d.ping("b", now=0.0)
    assert d.ack("b") is False  # wasn't suspect
    assert d.expire("b", now=1.0) is Action.WAIT  # nothing in flight


def test_ack_after_suspect_reports_revival() -> None:
    d = make_detector()
    d.ping("b", now=0.0)
    d.expire("b", now=1.0)  # PINGREQ
    d.expire("b", now=2.0)  # SUSPECT
    assert d.ack("b") is True  # revived from suspicion


def test_expire_on_unknown_peer_is_wait() -> None:
    d = make_detector()
    assert d.expire("never-pinged", now=10.0) is Action.WAIT


def test_dead_peer_stays_dead() -> None:
    d = make_detector()
    d.ping("b", now=0.0)
    d.expire("b", now=1.0)  # PINGREQ
    d.expire("b", now=2.0)  # SUSPECT
    d.expire("b", now=5.0)  # DEAD
    assert d.expire("b", now=6.0) is Action.WAIT  # already dead, no further action
