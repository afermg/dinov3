#!/usr/bin/env python3
"""End-to-end forward pass against the DINOv3 OCI container."""

import json
import os

os.environ.setdefault("NAHUAL_IPC_TIMEOUT_MS", "900000")

import numpy as np
from nahual.process import dispatch_setup_process


def main() -> None:
    address = os.environ.get("NAHUAL_ADDRESS", "tcp://127.0.0.1:5555")
    setup, process = dispatch_setup_process("dinov3")
    info = setup({"model_name": "dinov3_vits16", "pretrained": False}, address=address)
    pixels = np.random.default_rng(42).standard_normal((1, 3, 1, 224, 224)).astype(np.float32)
    result = process(pixels, address=address)
    assert result.shape == (1, 384), result.shape
    print(
        json.dumps(
            {
                "setup": info,
                "shape": list(result.shape),
                "finite": bool(np.isfinite(result).all()),
            }
        )
    )


if __name__ == "__main__":
    main()
