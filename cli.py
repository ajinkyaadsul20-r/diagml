"""Command line interface: diagml {train,predict,sample,serve}."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence

from . import __version__
from .config import load_config
from .data import load_data
from .predict import Predictor
from .train import train


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="diagml", description=__doc__)
    p.add_argument("--version", action="version", version=f"diagml {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Train, compare and save the best model")
    t.add_argument("--config", default=None, help="Path to YAML config")

    pr = sub.add_parser("predict", help="Predict from a JSON file of {feature: value}")
    pr.add_argument("json_file", help="JSON file with a single record")
    pr.add_argument("--artifacts", default="artifacts")

    s = sub.add_parser("sample", help="Print an example record (row index) as JSON")
    s.add_argument("--index", type=int, default=0)

    sv = sub.add_parser("serve", help="Start the REST API (needs the [api] extra)")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--artifacts", default="artifacts")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)

    if args.command == "train":
        meta = train(load_config(args.config))
        print(json.dumps({"model": meta["model_name"], **meta["test_metrics"]}, indent=2))
    elif args.command == "predict":
        with open(args.json_file) as fh:
            record = json.load(fh)
        print(json.dumps(Predictor(args.artifacts).predict_one(record), indent=2))
    elif args.command == "sample":
        X, _ = load_data()
        print(json.dumps(X.iloc[args.index].to_dict(), indent=2))
    elif args.command == "serve":
        import os

        import uvicorn

        os.environ["DIAGML_ARTIFACTS_DIR"] = args.artifacts
        uvicorn.run("diagml.api:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
