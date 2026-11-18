import pandas as pd

from mthl.train import chronological_split


def test_chronological_split_keeps_equal_timestamps_together():
    frame = pd.DataFrame({
        "ocid": [f"P{i}" for i in range(20)],
        "prediction_time": [
            "2024-01-01T00:00:00Z" if i < 2 else
            "2024-01-02T00:00:00Z" if i < 4 else
            f"2024-01-{i:02d}T00:00:00Z"
            for i in range(1, 21)
        ],
    })
    train, validation, test = chronological_split(frame)
    sets = [set(x["prediction_time"]) for x in (train, validation, test)]
    assert not (sets[0] & sets[1])
    assert not (sets[0] & sets[2])
    assert not (sets[1] & sets[2])
    assert len(train) + len(validation) + len(test) == len(frame)


def test_nnpu_training_uses_only_findings_published_by_training_cutoff(tmp_path):
    import json
    from pathlib import Path

    import pandas as pd

    from mthl.train import train

    project_root = Path(__file__).resolve().parents[1]
    processed = project_root / "data/processed/demo_procurements.csv"
    frame = pd.read_csv(processed).sort_values("prediction_time")
    early = frame.iloc[:3]["ocid"].astype(str).tolist()
    late = str(frame.iloc[-1]["ocid"])
    labels = pd.DataFrame({
        "ocid": early + [late],
        "published_at": [
            "2020-01-01T00:00:00Z",
            "2020-01-02T00:00:00Z",
            "2020-01-03T00:00:00Z",
            "2030-01-01T00:00:00Z",
        ],
    })
    labels_path = tmp_path / "positive_labels.csv"
    labels.to_csv(labels_path, index=False)
    report = train(
        str(processed), str(tmp_path / "model.joblib"), str(tmp_path / "report.json"),
        positive_labels_path=str(labels_path), positive_prior=0.05,
    )
    evaluation = report["evaluation"]
    assert evaluation["known_positive_train_count"] == 3
    assert evaluation["positive_labels_excluded_as_not_yet_published_at_training_cutoff"] == 1
    assert report["final_service_model"]["training_mode"] == "positive_unlabelled_learning"
