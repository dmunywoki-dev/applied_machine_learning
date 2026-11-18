import json
from datetime import datetime, timedelta, timezone

from mthl.ingest import build_process_dataset


def _release(ocid, when, tags, *, awards=None, method="open", tenderers=None, tender=True):
    item = {
        "ocid": ocid,
        "id": f"{ocid}-{tags[0]}",
        "date": when.isoformat().replace("+00:00", "Z"),
        "tag": tags,
        "buyer": {"id": "B-1", "name": "Buyer One", "address": {"region": "Nairobi"}},
    }
    if tender:
        parties = [{"id": x, "name": x, "roles": ["tenderer"]} for x in (tenderers or [])]
        item["parties"] = parties
        item["tender"] = {
            "title": "Test goods tender", "description": "A public supply purchase",
            "procurementMethod": method, "mainProcurementCategory": "goods", "status": "active",
            "value": {"amount": 1000, "currency": "KES"},
            "tenderPeriod": {"startDate": (when + timedelta(days=1)).isoformat(), "endDate": (when + timedelta(days=8)).isoformat()},
        }
    if awards is not None:
        item["awards"] = awards
    return item


def test_one_observation_per_ocid_and_future_award_not_used_for_own_score():
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    releases = [
        _release("A", t0, ["tender"], tenderers=["X", "Y"]),
        _release("A", t0 + timedelta(days=20), ["award"], tender=False,
                 awards=[{"value": {"amount": 100}, "suppliers": [{"id": "X", "name": "X"}]}]),
        _release("B", t0 + timedelta(days=40), ["tender"], tenderers=["X", "Y"]),
        _release("B", t0 + timedelta(days=60), ["award"], tender=False,
                 awards=[{"value": {"amount": 200}, "suppliers": [{"id": "X", "name": "X"}]}]),
        _release("C", t0 + timedelta(days=80), ["tender"], tenderers=[]),
    ]
    df, report = build_process_dataset(releases)
    assert len(df) == 3
    a = df.set_index("ocid").loc["A"]
    b = df.set_index("ocid").loc["B"]
    c = df.set_index("ocid").loc["C"]
    assert a["buyer_prior_awarded_process_count"] == 0
    assert b["buyer_prior_awarded_process_count"] == 1
    assert c["buyer_prior_awarded_process_count"] == 2
    assert b["hyperedge_prior_recurrence"] == 1
    assert c["num_tenderers"] == 0
    assert report["conditional_bidder_hypergraph_enabled"] is False or report["conditional_bidder_hypergraph_enabled"] is True


def test_supplier_role_is_not_counted_as_tenderer():
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    release = _release("A", t0, ["tender"], tenderers=[])
    release["parties"].append({"id": "SUP", "name": "Supplier", "roles": ["supplier"]})
    df, _ = build_process_dataset([release])
    assert int(df.iloc[0]["num_tenderers"]) == 0


def test_same_timestamp_records_do_not_leak_into_each_others_prior_history():
    t0 = datetime(2025, 3, 1, tzinfo=timezone.utc)
    releases = [
        _release("A", t0, ["tender"], tenderers=["X", "Y"]),
        _release("B", t0, ["tender"], tenderers=["X", "Y"]),
        _release("C", t0 + timedelta(days=1), ["tender"], tenderers=["X", "Y"]),
    ]
    frame, _ = build_process_dataset(releases)
    by_ocid = frame.set_index("ocid")
    assert int(by_ocid.loc["A", "buyer_prior_process_count"]) == 0
    assert int(by_ocid.loc["B", "buyer_prior_process_count"]) == 0
    assert int(by_ocid.loc["C", "buyer_prior_process_count"]) == 2
    assert int(by_ocid.loc["A", "hyperedge_prior_recurrence"]) == 0
    assert int(by_ocid.loc["B", "hyperedge_prior_recurrence"]) == 0
    assert int(by_ocid.loc["C", "hyperedge_prior_recurrence"]) == 2


def test_read_jsonl_gz_package(tmp_path):
    import gzip
    import json
    from mthl.ingest import read_release_packages

    payload = {
        "ocid": "ocds-jsonl-test",
        "id": "release-1",
        "date": "2025-01-01T00:00:00Z",
        "tender": {"title": "JSONL tender"},
        "tag": ["tender"],
    }
    path = tmp_path / "full.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    releases, audit = read_release_packages(tmp_path)
    assert len(releases) == 1
    assert releases[0]["ocid"] == "ocds-jsonl-test"
    assert audit[0]["object_count"] == 1


def test_compiled_snapshot_is_marked_as_latest_state_not_tender_stage():
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    first = _release("A", t0, ["compiled"], awards=[{
        "value": {"amount": 1000},
        "suppliers": [{"id": "S1", "name": "Supplier 1"}],
    }])
    first["tag"] = ["compiled"]
    second = _release("B", t0 + timedelta(days=30), ["compiled"])
    second["tag"] = ["compiled"]
    frame, quality = build_process_dataset([
        {**first, "_representation": "compiled_snapshot"},
        {**second, "_representation": "compiled_snapshot"},
    ])
    assert set(frame["source_representation"]) == {"compiled_snapshot"}
    assert set(frame["prediction_stage"]) == {"latest_compiled_process_snapshot"}
    assert quality["input_representation_mode"] == "compiled_snapshot_only"
    assert quality["point_in_time_tender_stage_reconstruction_available"] is False
    assert int(frame.set_index("ocid").loc["B", "buyer_prior_awarded_process_count"]) == 1


def test_compiled_jsonl_is_recognised_from_registry_style_line(tmp_path):
    import gzip
    import json
    from mthl.ingest import read_release_packages, build_process_dataset

    compiled = {
        "ocid": "ocds-compiled-001", "id": "ocds-compiled-001-2025-01-01",
        "date": "2025-01-01T00:00:00Z", "tag": ["compiled"],
        "buyer": {"id": "B", "name": "Buyer"},
        "tender": {"title": "Current tender snapshot", "status": "complete",
                   "procurementMethod": "open", "mainProcurementCategory": "goods"},
        "awards": [{"value": {"amount": 1000}, "suppliers": [{"id": "S", "name": "Supplier"}]}],
    }
    path = tmp_path / "full.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(compiled) + "\n")
    objects, _ = read_release_packages(tmp_path)
    assert objects[0]["_representation"] == "compiled_snapshot"
    frame, report = build_process_dataset(objects)
    assert frame.loc[0, "prediction_stage"] == "latest_compiled_process_snapshot"
    assert report["input_representation_mode"] == "compiled_snapshot_only"
    assert report["input_release_count"] == 0
    assert report["input_compiled_snapshot_count"] == 1
