"""Tests for the metrics collector (torch-free)."""

import json

from mixup_node.metrics import MetricsCollector


def test_metrics_collector_writes_jsonl(tmp_path) -> None:
    collector = MetricsCollector("n1", "hello", 50, path=tmp_path / "metrics_n1.jsonl")
    collector.record_expert(
        token=1,
        layer=0,
        expert="c2",
        serialize=0.001,
        http=0.010,
        server=0.005,
        deserialize=0.002,
    )
    collector.record_layer(
        token=1, layer=0, experts=["c2"], dispatch=0.020, reduce_=0.001
    )
    collector.record_generation(tokens=4, total=1.0, tok_per_sec=4.0)
    collector.close()

    lines = (tmp_path / "metrics_n1.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]

    assert [r["type"] for r in records] == ["expert", "layer", "generation"]
    assert records[0]["expert"] == "c2"
    assert records[0]["network_seconds"] == 0.005
    assert records[1]["experts"] == ["c2"]
    assert records[2]["tokens"] == 4
    assert records[2]["dispatch_seconds"] == 0.020
    assert records[2]["dense_seconds"] == 0.98  # 1.0 - 0.020
    assert records[2]["expert_requests"] == 1


def test_metrics_generation_id_shared(tmp_path) -> None:
    collector = MetricsCollector("n1", "hi", 10, path=tmp_path / "m.jsonl")
    collector.record_expert(
        token=1, layer=1, expert="c2", serialize=0, http=0, server=0, deserialize=0
    )
    collector.record_generation(tokens=1, total=0.1, tok_per_sec=10.0)
    collector.close()

    lines = (tmp_path / "m.jsonl").read_text().splitlines()
    ids = {json.loads(line)["generation_id"] for line in lines}
    assert len(ids) == 1  # both lines share one generation id
