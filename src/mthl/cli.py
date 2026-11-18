"""Command-line entry points for the MTHL pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ingest import run_ingestion
from .train import train


def main():
    parser = argparse.ArgumentParser(prog="mthl", description="MTHL procurement screening pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest raw OCDS release packages")
    ingest.add_argument("--input-dir", default="data/raw")
    ingest.add_argument("--output", default="data/processed/procurements.csv")
    ingest.add_argument("--quality", default="reports/data_quality.json")

    fit = sub.add_parser("train", help="Train the model and write an evaluation record")
    fit.add_argument("--input", default="data/processed/procurements.csv")
    fit.add_argument("--artifact", default="artifacts/model.joblib")
    fit.add_argument("--report", default="reports/evaluation.json")
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--mlflow", action="store_true")
    fit.add_argument("--positive-labels", default=None, help="CSV with verified positive OCIDs and finding publication timestamps")
    fit.add_argument("--positive-prior", type=float, default=None, help="Estimated positive class prior for optional nnPU mode")

    args = parser.parse_args()
    if args.command == "ingest":
        report = run_ingestion(args.input_dir, args.output, args.quality)
    else:
        report = train(args.input, args.artifact, args.report, args.seed, args.mlflow, args.positive_labels, args.positive_prior)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
