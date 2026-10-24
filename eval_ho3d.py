#!/usr/bin/env python3
"""Evaluate HO3D flow predictions with the paper's 18-sequence protocol."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    protocol_path = ROOT / "configs/ho3d_eval_frames.json"
    protocol = json.loads(protocol_path.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, default=ROOT / "results/ho3d/chunk_128/re_1",
                        help="Directory containing <sequence>.pkl (default: second refinement pass)")
    parser.add_argument("--ho3d-root", type=Path, required=True,
                        help="HO3D_v3 root containing train/, train.txt, and models/")
    parser.add_argument("--init-root", type=Path, default=ROOT / "_DATA/ho3dv3_init")
    parser.add_argument("--sequences", "--seq", nargs="+", choices=list(protocol), default=list(protocol))
    parser.add_argument("--output", type=Path, help="JSON report (default: <prediction-dir>/eval_ho3d.json)")
    parser.add_argument("--seed", type=int, default=3407, help="Sampling, registration, and containment seed (default: 3407)")
    args = parser.parse_args(argv)
    if len(set(args.sequences)) != len(args.sequences):
        parser.error("--sequences must not contain duplicates")
    if args.output is None:
        args.output = args.prediction_dir / "eval_ho3d.json"
    if args.output.suffix.lower() != ".json":
        parser.error("--output must be a .json report path")
    return args, protocol


def main(argv=None):
    args, protocol = parse_args(argv)
    import numpy as np
    import torch
    from hof.evaluation.ho3d import create_mano, evaluate_sequence

    # Small MANO batches benefit from bounded CPU threading on large machines.
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    valid_frames = set((args.ho3d_root / "train.txt").read_text().split())
    mano = create_mano(ROOT / "_DATA/mano")
    sequence_means = []
    for index, sequence in enumerate(args.sequences, start=1):
        print(f"[{index}/{len(args.sequences)}] Evaluating {sequence}...", flush=True)
        result = evaluate_sequence(
            sequence, prediction_dir=args.prediction_dir, init_root=args.init_root,
            ho3d_root=args.ho3d_root, protocol_frames=protocol[sequence],
            valid_frames=valid_frames, mano=mano, seed=args.seed)
        sequence_means.append(result["metrics"])
        summary = ", ".join(f"{key}={value:.4f}" for key, value in result["metrics"].items())
        print(f"{sequence}: {summary}", flush=True)
    # Each metric is already averaged over frames; give every sequence equal weight.
    mean = {key: float(np.mean([metrics[key] for metrics in sequence_means]))
            for key in sequence_means[0]}
    report = {"protocol": "HO3D train, correspondence frames below 1024, valid train.txt annotations",
              "prediction_dir": str(args.prediction_dir.resolve()),
              "init_root": str(args.init_root.resolve()), "ho3d_root": str(args.ho3d_root.resolve()),
              "seed": args.seed, "hand_vertex_samples": 3000, "object_registration_samples": 1000,
              "object_registration_iterations": 1200, "cd_units": "cm^2", "aggregation": "mean of sequence means",
              "num_sequences": len(sequence_means), "mean": mean}
    # Serialize before touching an existing report, and replace it atomically.
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(payload)
            handle.close()
            os.replace(temporary, args.output)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Overall ({len(sequence_means)} sequences):\n" +
          json.dumps(mean, indent=2), flush=True)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
