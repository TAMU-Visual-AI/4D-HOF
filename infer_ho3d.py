#!/usr/bin/env python3
"""Run 4D-HOF on prepared HO3D initialization results."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HO3D_SEQUENCES = (
    "ABF12", "ABF14", "GPMF12", "GPMF14", "MC1", "MC4", "MDF12", "MDF14",
    "ShSu10", "ShSu12", "SM2", "SM4", "SMu1", "SMu40", "GSF13",
    "GSF14", "BB12", "BB13",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/4dhof.yaml")
    parser.add_argument("--checkpoint", "--ckpt", type=Path, default=ROOT / "checkpoints/4dhof.pth",
                        help="Full model checkpoint (default: checkpoints/4dhof.pth)")
    parser.add_argument("--init-root", type=Path, default=ROOT / "_DATA/ho3dv3_init",
                        help="Prepared initialization root (default: _DATA/ho3dv3_init)")
    parser.add_argument("--ho3d-root", type=Path, required=True, help="HO3D_v3 root containing train/")
    parser.add_argument("--sequences", "--seq", nargs="+", default=list(HO3D_SEQUENCES))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/ho3d")
    parser.add_argument("--chunk-size", "--chunk_size", type=int, default=128)
    parser.add_argument("--stride-ratio", "--stride_ratio", type=float, default=0.5)
    parser.add_argument("--passes", type=int, default=2, help="Number of refinement passes (default: 2)")
    parser.add_argument("--seed", type=int, help="Override config seed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing prediction files")
    args = parser.parse_args(argv)
    if args.passes < 1:
        parser.error("--passes must be positive")
    if len(set(args.sequences)) != len(args.sequences):
        parser.error("--sequences must not contain duplicates")
    if any(Path(seq).name != seq or seq in {".", ".."} for seq in args.sequences):
        parser.error("--sequences must contain sequence names, not paths")
    from hof.inference.windows import inference_stride
    try:
        inference_stride(args.chunk_size, args.stride_ratio)
    except ValueError as error:
        parser.error(str(error))
    return args


def create_model(cfg, checkpoint_path, *, device="cuda:0"):
    from copy import deepcopy
    import torch
    from hof.model.builder import build_model

    model_cfg = deepcopy(cfg.model)
    mano_cfg = model_cfg.denoiser_cfg.cfg.mano_cfg
    path = ROOT / "_DATA/mano"
    if not (path / "MANO_RIGHT.pkl").is_file():
        raise FileNotFoundError(f"Place MANO_RIGHT.pkl at {path / 'MANO_RIGHT.pkl'}")
    mano_cfg.model_path = str(path)
    model = build_model(model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    # Includes DINOv2 and MANO buffers; never silently accept a partial checkpoint.
    model.load_state_dict(state, strict=True)
    model.normalize_trans = cfg.input.get("normalize_trans", True)
    return model.to(device).eval()


def main(argv=None):
    args = parse_args(argv)
    import pickle
    import random
    import numpy as np
    import yaml
    import torch
    from easydict import EasyDict
    from hof.inference.data import load_sequence
    from hof.inference.runner import infer_sequence

    with args.config.open() as handle:
        cfg = EasyDict(yaml.safe_load(handle))
    if not {"input", "model"}.issubset(cfg):
        raise ValueError("Expected an inference config with 'input' and 'model' sections")
    seed = cfg.get("seed", 3407) if args.seed is None else args.seed
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Inference requires CUDA (FlashAttention and spconv)")
    torch.cuda.set_device(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = create_model(cfg, args.checkpoint, device=device)

    for sequence in args.sequences:
        outputs = [args.output_dir / f"chunk_{args.chunk_size}" / f"re_{i}" / f"{sequence}.pkl"
                   for i in range(args.passes)]
        if not args.overwrite:
            for path in outputs:
                if path.exists():
                    raise FileExistsError(f"{path} already exists; use --overwrite to replace it")
        data = load_sequence(args.init_root, args.ho3d_root, sequence)
        print(f"{sequence}: validated {len(data['image_paths'])} frames", flush=True)
        previous = None
        for pass_index, path in enumerate(outputs):
            previous = infer_sequence(model, data, cfg.input, chunk_size=args.chunk_size,
                                      stride_ratio=args.stride_ratio, previous=previous,
                                      description=f"{sequence} pass {pass_index + 1}")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as handle:
                pickle.dump(previous, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved {path}", flush=True)
        model.clear_guidance_sdf_cache()


if __name__ == "__main__":
    main()
