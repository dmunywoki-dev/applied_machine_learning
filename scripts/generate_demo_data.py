"""Generate a compact synthetic OCDS-like fixture for reproducibility/API smoke tests only."""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path


def generate(output: str = "examples/demo_data/tender-releases-demo.json", n_processes: int = 180, seed: int = 113719):
    rng = random.Random(seed)
    buyers = [
        ("BUYER-001", "Nairobi County Department of Health", "Nairobi"),
        ("BUYER-002", "Mombasa County Department of Works", "Mombasa"),
        ("BUYER-003", "Kenya Digital Services Agency", "Nairobi"),
        ("BUYER-004", "Rift Valley Water Authority", "Nakuru"),
        ("BUYER-005", "Western Regional Hospital", "Kakamega"),
    ]
    suppliers = [(f"SUP-{i:03d}", f"Supplier {i:03d} Limited") for i in range(1, 31)]
    groups = [suppliers[i:i+3] for i in range(0, 15, 3)]
    categories = ["goods", "services", "works"]
    releases = []
    start = datetime(2018, 7, 1, 9, 0, tzinfo=timezone.utc)
    for i in range(n_processes):
        buyer_id, buyer_name, county = buyers[i % len(buyers)]
        date = start + timedelta(days=i * 16, hours=rng.randint(0, 8))
        ocid = f"ocds-demo-{i+1:06d}"
        category = categories[i % len(categories)]
        method = "direct" if rng.random() < 0.13 else "open"
        duration = rng.choice([2, 4, 6, 14, 21, 30, 45, 60])
        tender_start = date + timedelta(days=1)
        tender_end = tender_start + timedelta(days=duration)
        estimated = round(rng.uniform(150_000, 18_000_000), 2)
        if i % 29 == 0:
            title, description = "Emergency supply of medical equipment", "Urgent delivery of essential equipment for public health services."
        elif category == "works":
            title, description = "Rehabilitation of public access road", "Road maintenance and drainage works within the county."
        elif category == "goods":
            title, description = "Supply and delivery of office equipment", "Supply of computers, printers and related accessories."
        else:
            title, description = "Provision of ICT support services", "Managed information technology and systems support services."

        disclose = rng.random() < 0.68
        chosen_group = groups[(i // 3) % len(groups)] if i % 4 == 0 else rng.sample(suppliers, k=rng.randint(2, 4))
        parties = []
        if disclose:
            # Deliberately use only explicit tenderer roles. Awarded suppliers are added separately.
            for sid, sname in chosen_group:
                parties.append({"id": sid, "name": sname, "roles": ["tenderer"]})
        tender_release = {
            "ocid": ocid,
            "_synthetic_demo": True,
            "id": f"{ocid}-tender",
            "date": date.isoformat().replace("+00:00", "Z"),
            "tag": ["tender"],
            "buyer": {"id": buyer_id, "name": buyer_name, "address": {"region": county}},
            "parties": parties,
            "tender": {
                "id": f"T-{i+1:06d}",
                "title": title,
                "description": description,
                "status": "active",
                "procurementMethod": method,
                "mainProcurementCategory": category,
                "value": {"amount": estimated, "currency": "KES"},
                "tenderPeriod": {"startDate": tender_start.isoformat().replace("+00:00", "Z"), "endDate": tender_end.isoformat().replace("+00:00", "Z")},
            },
        }
        releases.append(tender_release)

        # Award release is published later and must not leak into this process's initial score.
        supplier_id, supplier_name = rng.choice(chosen_group if disclose else suppliers)
        award_value = round(estimated * rng.uniform(0.75, 1.08), 2)
        award_date = tender_end + timedelta(days=rng.randint(3, 40))
        award_release = {
            "ocid": ocid,
            "_synthetic_demo": True,
            "id": f"{ocid}-award",
            "date": award_date.isoformat().replace("+00:00", "Z"),
            "tag": ["award"],
            "buyer": {"id": buyer_id, "name": buyer_name, "address": {"region": county}},
            "awards": [{
                "id": f"A-{i+1:06d}",
                "status": "active",
                "value": {"amount": award_value, "currency": "KES"},
                "suppliers": [{"id": supplier_id, "name": supplier_name}],
            }],
        }
        releases.append(award_release)

    payload = {
        "uri": "https://example.invalid/synthetic-ocds-demo",
        "version": "1.1",
        "publisher": {"name": "MTHL synthetic fixture", "scheme": "GB-COH"},
        "publishedDate": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "releases": sorted(releases, key=lambda x: x["date"]),
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(releases):,} synthetic OCDS-like releases for {n_processes:,} processes to {path}")
    print("WARNING: this is synthetic data for software testing, not evidence about Kenyan procurement.")


if __name__ == "__main__":
    generate()
