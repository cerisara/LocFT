#!/usr/bin/env python3
"""
Compute the magnitude of LoRA weight differences between each epoch
and the initial epoch (0).

Magnitude = sqrt(sum(diff^2)) / num_elements
          = Frobenius_norm(diff) / num_elements
          = per-element RMS of the difference
"""

import os
import sys
import glob
import torch
from safetensors.torch import load_file
from pathlib import Path

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <checkpoint_dir>")
    sys.exit(1)

DIR = Path(sys.argv[1])

def main(checkpoint_dir):
    # Gather checkpoint dirs sorted by epoch
    epoch_dirs = sorted(
        glob.glob(str(checkpoint_dir / "checkpoint_epoch_*")),
        key=lambda p: int(Path(p).name.split("_")[-1]),
    )

    if len(epoch_dirs) < 2:
        print("Need at least epoch_0 and one more checkpoint.")
        sys.exit(1)

    # Load epoch 0 as baseline
    base_path = os.path.join(epoch_dirs[0], "adapter_model.safetensors")
    base_state = load_file(base_path)
    print(f"Loaded baseline from: {base_path}")
    print(f"  Keys: {len(base_state)}")

    # Verify all epochs have the same keys
    for ep_dir in epoch_dirs[1:]:
        ckpt_path = os.path.join(ep_dir, "adapter_model.safetensors")
        state = load_file(ckpt_path)
        if set(state.keys()) != set(base_state.keys()):
            print(f"WARNING: {ckpt_path} has different keys!")
            sys.exit(1)

    print(f"\n{'Epoch':>6} | {'Magnitude (per-dim)':>22} | {'Frobenius norm':>18} | {'Total dims':>12}")
    print("-" * 65)

    for ep_dir in epoch_dirs[1:]:
        epoch_num = int(Path(ep_dir).name.split("_")[-1])
        ckpt_path = os.path.join(ep_dir, "adapter_model.safetensors")
        state = load_file(ckpt_path)

        total_frobenius_sq = 0.0
        total_elements = 0

        for key in state:
            diff = state[key] - base_state[key]
            # Frobenius norm squared (sum of squared diffs)
            total_frobenius_sq += diff.pow(2).sum().item()
            total_elements += diff.numel()

        frobenius_norm = total_frobenius_sq ** 0.5
        magnitude_per_dim = frobenius_norm / total_elements

        print(f"{epoch_num:>6} | {magnitude_per_dim:>22.8f} | {frobenius_norm:>18.4f} | {total_elements:>12}")

if __name__ == "__main__":
    main(DIR)
