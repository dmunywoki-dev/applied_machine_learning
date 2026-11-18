"""Read raw OCDS release packages and build leakage-aware process observations."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _parse_date(value: Any) -> pd.Timestamp | pd.NaT:
    if value in (None, ""):
        return pd.NaT
    try:
        return pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return pd.NaT


def _number(value: Any) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _normalise_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _entity_key(obj: dict, fallback_name: Any = None) -> str:
    obj = _as_dict(obj)
    identifier = str(obj.get("id") or "").strip()
    if identifier:
        return f"id:{identifier}"
    name = _normalise_name(obj.get("name") or fallback_name)
    return f"name:{name}" if name else "unknown"


def _buyer_info(release: dict) -> tuple[str, str, str]:
    buyer = _as_dict(release.get("buyer"))
    name = str(buyer.get("name") or "Unknown buyer").strip()
    key = _entity_key(buyer, name)
    address = _as_dict(buyer.get("address"))
    county = str(address.get("region") or address.get("locality") or "Unknown").strip()
    return key, name, county


def _tenderers(release: dict) -> list[str]:
    """Return only parties explicitly labelled tenderer; do not count awarded suppliers."""
    values = []
    for party in _as_list(release.get("parties")):
        party = _as_dict(party)
        roles = {str(x).lower() for x in _as_list(party.get("roles"))}
        if "tenderer" in roles:
            key = _entity_key(party)
            if key != "unknown":
                values.append(key)
    return sorted(set(values))


def _supplier_awards(release: dict) -> tuple[float, dict[str, float], int]:
    total = 0.0
    suppliers: dict[str, float] = defaultdict(float)
    award_count = 0
    for award in _as_list(release.get("awards")):
        award = _as_dict(award)
        award_count += 1
        value = _number(_as_dict(award.get("value")).get("amount"))
        value = 0.0 if np.isnan(value) else max(0.0, value)
        total += value
        award_suppliers = _as_list(award.get("suppliers"))
        if award_suppliers:
            split = value / max(1, len(award_suppliers))
            for supplier in award_suppliers:
                key = _entity_key(_as_dict(supplier))
                if key != "unknown":
                    suppliers[key] += split
    return total, dict(suppliers), award_count


def read_release_packages(input_dir: str | Path) -> tuple[list[dict], list[dict]]:
    """Load OCDS JSON packages or OCP Registry JSONL/JSONL.GZ compiled releases."""
    input_dir = Path(input_dir)
    candidates = []
    for pattern in ("*.json", "*.jsonl", "*.jsonl.gz", "*.ndjson"):
        candidates.extend(input_dir.glob(pattern))
    paths = sorted(set(candidates))
    if not paths:
        raise FileNotFoundError(
            f"No .json, .jsonl or .jsonl.gz files found in {input_dir}. Place raw PPIP/OCDS exports there."
        )
    all_releases: list[dict] = []
    file_audit: list[dict] = []

    def add_release(release: dict, source: str, representation: str, parent_ocid: str | None = None):
        item = dict(release)
        if not item.get("ocid") and parent_ocid:
            item["ocid"] = parent_ocid
        item["_source_file"] = source
        item["_representation"] = representation
        all_releases.append(item)

    def extend_compiled(obj: Any, source: str):
        if isinstance(obj, dict):
            parent_ocid = str(obj.get("ocid") or "").strip() or None
            added_embedded_release = False
            if isinstance(obj.get("releases"), list):
                # A record's releases can be embedded full release objects or
                # linked-release references. Only embedded OCDS releases have
                # enough fields for a point-in-time reconstruction.
                for release in obj["releases"]:
                    if not isinstance(release, dict):
                        continue
                    has_release_content = any(k in release for k in ("tender", "awards", "contracts", "planning"))
                    if has_release_content:
                        add_release(release, source, "raw_release", parent_ocid)
                        added_embedded_release = True
            if isinstance(obj.get("compiledRelease"), dict):
                add_release(obj["compiledRelease"], source, "compiled_snapshot", parent_ocid)
            elif not added_embedded_release and parent_ocid and any(k in obj for k in ("tender", "awards", "contracts")):
                tags = {str(x).lower() for x in _as_list(obj.get("tag"))}
                representation = "compiled_snapshot" if "compiled" in tags else "raw_release"
                add_release(obj, source, representation)
            elif not added_embedded_release and not isinstance(obj.get("compiledRelease"), dict):
                raise ValueError(f"{source} contains no embedded OCDS release or compiledRelease")
        elif isinstance(obj, list):
            for value in obj:
                extend_compiled(value, source)
        else:
            raise ValueError(f"{source} contains an unsupported JSON value")

    for path in paths:
        raw_bytes = path.read_bytes()
        file_audit.append({
            "file": path.name,
            "bytes": len(raw_bytes),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        })
        is_jsonl = path.name.endswith((".jsonl", ".jsonl.gz", ".ndjson"))
        if is_jsonl:
            opener = gzip.open if path.name.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8-sig") as handle:
                before = len(all_releases)
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        extend_compiled(json.loads(line), path.name)
                    except Exception as exc:
                        raise ValueError(f"Invalid OCDS JSONL at {path.name}:{line_number}: {exc}") from exc
                added = all_releases[before:]
                file_audit[-1]["object_count"] = len(added)
                file_audit[-1]["raw_release_count"] = sum(x.get("_representation") == "raw_release" for x in added)
                file_audit[-1]["compiled_snapshot_count"] = sum(x.get("_representation") == "compiled_snapshot" for x in added)
        else:
            payload = json.loads(raw_bytes.decode("utf-8-sig"))
            if isinstance(payload, dict) and payload.get("data") and isinstance(payload["data"], dict):
                payload = payload["data"]
            before = len(all_releases)
            extend_compiled(payload, path.name)
            added = all_releases[before:]
            file_audit[-1]["object_count"] = len(added)
            file_audit[-1]["raw_release_count"] = sum(x.get("_representation") == "raw_release" for x in added)
            file_audit[-1]["compiled_snapshot_count"] = sum(x.get("_representation") == "compiled_snapshot" for x in added)
    return all_releases, file_audit


def build_process_dataset(releases: Iterable[dict]) -> tuple[pd.DataFrame, dict]:
    """Build one process observation using raw releases or latest compiled snapshots.

    Raw release histories score a process at its first tender-stage release. If the
    source contains only compiled releases, each process is scored at the date of
    its latest compiled state; this is a later-state review task, not a reconstruction
    of the original tender-stage information set.
    """
    release_items = [
        ({**item, "_representation": item.get("_representation", "raw_release")})
        for item in releases
    ]
    input_object_count = len(release_items)
    input_raw_release_count = sum(r.get("_representation") == "raw_release" for r in release_items)
    input_compiled_snapshot_count = sum(r.get("_representation") == "compiled_snapshot" for r in release_items)
    input_fingerprints = {
        (str(r.get("id") or ""), str(r.get("ocid") or ""), str(r.get("date") or ""))
        for r in release_items
    }
    raw_ocids = {str(r.get("ocid") or "").strip() for r in release_items if r.get("_representation") == "raw_release"}
    compiled_only_ocids = {str(r.get("ocid") or "").strip() for r in release_items if r.get("_representation") == "compiled_snapshot"} - raw_ocids
    if raw_ocids:
        # Prefer release history consistently when available. Ignore compiled
        # snapshots so that a mixed source cannot silently mix prediction stages.
        release_items = [r for r in release_items if r.get("_representation") != "compiled_snapshot"]
    by_ocid: dict[str, list[dict]] = defaultdict(list)
    invalid_ocid = 0
    invalid_date = 0
    for release in release_items:
        ocid = str(release.get("ocid") or "").strip()
        if not ocid:
            invalid_ocid += 1
            continue
        date = _parse_date(release.get("date"))
        if pd.isna(date):
            invalid_date += 1
            continue
        item = dict(release)
        item["_parsed_date"] = date
        by_ocid[ocid].append(item)

    observations: list[dict] = []
    award_events: list[dict] = []
    for ocid, items in by_ocid.items():
        items.sort(key=lambda x: (x["_parsed_date"], str(x.get("id") or "")))
        tender_releases = [
            item for item in items
            if isinstance(item.get("tender"), dict) and item.get("tender")
        ]
        if tender_releases:
            raw_tender_releases = [x for x in tender_releases if x.get("_representation", "raw_release") == "raw_release"]
            if raw_tender_releases:
                candidates = [x for x in raw_tender_releases if "tender" in _as_list(x.get("tag"))]
                selected = (candidates or raw_tender_releases)[0]
                source_representation = "raw_release_history"
                prediction_stage = "first_tender_release"
            else:
                compiled_candidates = [x for x in tender_releases if x.get("_representation") == "compiled_snapshot"]
                selected = max(compiled_candidates or tender_releases, key=lambda x: x["_parsed_date"])
                source_representation = "compiled_snapshot"
                prediction_stage = "latest_compiled_process_snapshot"
            tender = _as_dict(selected.get("tender"))
            buyer_key, buyer_name, county = _buyer_info(selected)
            period = _as_dict(tender.get("tenderPeriod"))
            start = _parse_date(period.get("startDate"))
            end = _parse_date(period.get("endDate"))
            duration = (end - start).total_seconds() / 86400 if not pd.isna(start) and not pd.isna(end) else np.nan
            if not pd.isna(duration) and (duration < 0 or duration > 365):
                duration = np.nan
            value = _number(_as_dict(tender.get("value")).get("amount"))
            title = str(tender.get("title") or "").strip()
            description = str(tender.get("description") or "").strip()
            text = (title + " " + description).strip()
            tenderers = _tenderers(selected)
            method = str(tender.get("procurementMethod") or "unknown").lower()
            category = str(tender.get("mainProcurementCategory") or "unknown").lower()
            status = str(tender.get("status") or "unknown").lower()
            timestamp = selected["_parsed_date"]
            observations.append({
                "ocid": ocid,
                "is_synthetic": int(bool(selected.get("_synthetic_demo", False))),
                "prediction_time": timestamp,
                "source_representation": source_representation,
                "prediction_stage": prediction_stage,
                "buyer_id": buyer_key,
                "buyer_name": buyer_name,
                "county": county,
                "procurement_method": method,
                "procurement_category": category,
                "tender_status": status,
                "title": title,
                "description": description,
                "text": text if text else "untitled procurement",
                "estimated_value": value,
                "duration_days": duration,
                "num_tenderers": len(tenderers),
                "tenderer_data_available": int(len(tenderers) > 0),
                "_tenderer_ids": tenderers,
            })

        # A process contributes at most one first-observed award event to later histories.
        award_release = next((x for x in items if _as_list(x.get("awards"))), None)
        if award_release is not None:
            buyer_key, _, _ = _buyer_info(award_release)
            total, suppliers, award_count = _supplier_awards(award_release)
            award_events.append({
                "ocid": ocid,
                "event_time": award_release["_parsed_date"],
                "buyer_id": buyer_key,
                "award_value": total,
                "suppliers": suppliers,
                "award_count": award_count,
            })

    observations.sort(key=lambda x: (x["prediction_time"], x["ocid"]))
    award_events.sort(key=lambda x: (x["event_time"], x["ocid"]))

    # Event-sourced buyer and bidder-group states. Award events are consumed before a
    # tender only when their publication time is strictly earlier than its prediction time.
    buyer_process_state: dict[str, dict] = defaultdict(lambda: {
        "process_count": 0, "direct_count": 0, "last_tender_time": None
    })
    buyer_award_state: dict[str, dict] = defaultdict(lambda: {
        "award_count": 0, "award_value": 0.0, "supplier_values": defaultdict(float), "suppliers": set()
    })
    group_counts: dict[str, int] = defaultdict(int)
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    event_idx = 0
    rows: list[dict] = []

    # Process equal prediction timestamps as a batch. No process published at the
    # same timestamp can become another process's "prior" history because source
    # ordering within that timestamp is not reliable.
    observations_by_time: dict[pd.Timestamp, list[dict]] = defaultdict(list)
    for obs in observations:
        observations_by_time[obs["prediction_time"]].append(obs)

    for t in sorted(observations_by_time):
        current_batch = observations_by_time[t]
        # Apply only award events published strictly before this timestamp.
        while event_idx < len(award_events) and award_events[event_idx]["event_time"] < t:
            event = award_events[event_idx]
            state = buyer_award_state[event["buyer_id"]]
            state["award_count"] += event["award_count"]
            state["award_value"] += event["award_value"]
            for supplier, amount in event["suppliers"].items():
                state["supplier_values"][supplier] += amount
                state["suppliers"].add(supplier)
            event_idx += 1

        # Snapshot features for every process in the timestamp batch before
        # mutating buyer or bidder-group histories.
        batch_updates: list[tuple[dict, str, list[str], str]] = []
        for obs in current_batch:
            bstate = buyer_process_state[obs["buyer_id"]]
            astate = buyer_award_state[obs["buyer_id"]]
            supplier_values = np.array(list(astate["supplier_values"].values()), dtype=float)
            if supplier_values.size and supplier_values.sum() > 0:
                shares = supplier_values / supplier_values.sum()
                hhi = float(np.square(shares).sum())
                top_share = float(shares.max())
            else:
                hhi, top_share = 0.0, 0.0

            tenderer_ids = obs.pop("_tenderer_ids")
            group_key = "|".join(tenderer_ids) if len(tenderer_ids) >= 2 else ""
            pair_values = []
            if len(tenderer_ids) >= 2:
                for i, left in enumerate(tenderer_ids):
                    for right in tenderer_ids[i + 1:]:
                        pair_values.append(pair_counts[tuple(sorted((left, right)))])

            month = int(t.month)
            duration = obs["duration_days"]
            estimated = obs["estimated_value"]
            method = obs["procurement_method"]
            text = obs["text"]
            row = dict(obs)
            row.update({
                "estimated_value_log1p": float(np.log1p(max(0.0, estimated))) if not pd.isna(estimated) else np.nan,
                "buyer_prior_process_count": int(bstate["process_count"]),
                "buyer_prior_direct_share": float(bstate["direct_count"] / bstate["process_count"]) if bstate["process_count"] else 0.0,
                "buyer_prior_awarded_process_count": int(astate["award_count"]),
                "buyer_prior_award_value_log1p": float(np.log1p(max(0.0, astate["award_value"]))),
                "buyer_prior_unique_suppliers": int(len(astate["suppliers"])),
                "buyer_prior_supplier_hhi": hhi,
                "buyer_prior_top_supplier_share": top_share,
                "days_since_buyer_previous_tender": (
                    float((t - bstate["last_tender_time"]).total_seconds() / 86400)
                    if bstate["last_tender_time"] is not None else np.nan
                ),
                "pub_month_sin": float(np.sin(2 * np.pi * month / 12.0)),
                "pub_month_cos": float(np.cos(2 * np.pi * month / 12.0)),
                "pub_weekday": int(t.dayofweek),
                "flag_short_tender": int(not pd.isna(duration) and 0 < duration < 7),
                "flag_non_open_method": int(method not in {"open", "unknown", ""}),
                "flag_single_disclosed_tenderer": int(len(tenderer_ids) == 1),
                "flag_high_prior_supplier_concentration": int(hhi >= 0.50 and astate["award_count"] > 0),
                "text_missing": int(not bool(text.strip()) or text == "untitled procurement"),
                "buyer_missing": int(obs["buyer_id"] == "unknown" or obs["buyer_name"] == "Unknown buyer"),
                "duration_missing": int(pd.isna(duration)),
                "estimated_value_missing": int(pd.isna(estimated)),
                "hyperedge_size": int(len(tenderer_ids)) if len(tenderer_ids) >= 2 else 0,
                "hyperedge_prior_recurrence": int(group_counts[group_key]) if group_key else 0,
                "hyperedge_prior_pair_recurrence_mean": float(np.mean(pair_values)) if pair_values else 0.0,
            })
            rows.append(row)
            batch_updates.append((obs, obs["buyer_id"], tenderer_ids, group_key))

        # Only after all same-time processes are featurised do they enter history.
        for obs, buyer_key, tenderer_ids, group_key in batch_updates:
            bstate = buyer_process_state[buyer_key]
            bstate["process_count"] += 1
            if obs["procurement_method"] == "direct":
                bstate["direct_count"] += 1
            bstate["last_tender_time"] = t
            if group_key:
                group_counts[group_key] += 1
                for i, left in enumerate(tenderer_ids):
                    for right in tenderer_ids[i + 1:]:
                        pair_counts[tuple(sorted((left, right)))] += 1

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No procurement observations with an OCID, publication date and tender object were found.")
    df["prediction_time"] = pd.to_datetime(df["prediction_time"], utc=True).astype(str)
    # Remove internal diagnostics and fields never sent to the model/API.
    if "_buyer_has_prior_awards" in df:
        df = df.drop(columns=["_buyer_has_prior_awards"])

    tenderer_coverage = float((df["num_tenderers"] >= 2).mean())
    hypergraph_enabled = int((df["num_tenderers"] >= 2).sum()) >= 10 and tenderer_coverage >= 0.02
    quality = {
        "input_object_count": int(input_object_count),
        "input_release_count": int(input_raw_release_count),
        "input_compiled_snapshot_count": int(input_compiled_snapshot_count),
        "unique_ocid_count": int(len(by_ocid)),
        "process_observation_count": int(len(df)),
        "raw_release_history_processes": int((df["source_representation"] == "raw_release_history").sum()),
        "compiled_snapshot_processes": int((df["source_representation"] == "compiled_snapshot").sum()),
        "compiled_only_ocids_excluded_when_raw_release_history_was_present": int(len(compiled_only_ocids)) if raw_ocids else 0,
        "input_representation_mode": "raw_release_history" if raw_ocids else "compiled_snapshot_only",
        "point_in_time_tender_stage_reconstruction_available": bool(raw_ocids and len(df) > 0),
        "prediction_time_semantics": (
            "first tender-stage release publication timestamp; strict earlier-release history is available"
            if raw_ocids else
            "date of latest compiled release per process; latest-state review only, not original tender-stage reconstruction"
        ),
        "invalid_ocid_releases": int(invalid_ocid),
        "invalid_publication_date_releases": int(invalid_date),
        "duplicate_release_excess": int(max(0, input_object_count - len(input_fingerprints))),
        "time_min": str(pd.to_datetime(df["prediction_time"], utc=True).min()),
        "time_max": str(pd.to_datetime(df["prediction_time"], utc=True).max()),
        "tenderer_disclosure_processes_with_2plus": int((df["num_tenderers"] >= 2).sum()),
        "tenderer_disclosure_rate_2plus": tenderer_coverage,
        "conditional_bidder_hypergraph_enabled": bool(hypergraph_enabled),
        "field_completeness": {
            c: float(1.0 - df[c].isna().mean()) for c in [
                "buyer_id", "procurement_method", "procurement_category", "text",
                "estimated_value", "duration_days", "num_tenderers"
            ] if c in df.columns
        },
        "interpretation": "Missing or undisclosed fields indicate data limitations, not misconduct.",
    }
    return df, quality


def run_ingestion(input_dir: str | Path, output_path: str | Path, quality_path: str | Path) -> dict:
    releases, file_audit = read_release_packages(input_dir)
    df, quality = build_process_dataset(releases)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    quality["source_files"] = file_audit
    Path(quality_path).parent.mkdir(parents=True, exist_ok=True)
    Path(quality_path).write_text(json.dumps(quality, indent=2), encoding="utf-8")
    return quality
