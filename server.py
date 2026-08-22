"""Nahual server for DINOv3.

Loads a DINOv3 backbone via the local `hubconf.py` (torch.hub.load with
`source='local'`), then runs it through the standard Nahual setup/process loop.

Run with:
    nix run . -- ipc:///tmp/dinov3.ipc
or:
    python server.py ipc:///tmp/dinov3.ipc
"""

import os
import sys
from functools import partial
from typing import Callable

import numpy
import pynng
import torch
import trio
from nahual.preprocess import channel_chunks_rigid3, validate_input_shape
from nahual.server import responder

address = sys.argv[1]

# Repo root containing hubconf.py — defaults to the directory of this file.
REPO_DIR = os.environ.get("DINOV3_REPO_DIR", os.path.dirname(os.path.abspath(__file__)))


def setup(
    model_name: str = "dinov3_vits16",
    weights: str | None = None,
    pretrained: bool = True,
    device: int | None = None,
    expected_tile_size: int = 16,
    expected_channels: int = 3,
) -> tuple[Callable, dict]:
    """Load a DINOv3 backbone via the local hubconf.

    Parameters
    ----------
    model_name : str
        Hub entrypoint, e.g. ``dinov3_vits16``, ``dinov3_vitb16``.
    weights : str | None
        Path or URL to a checkpoint. If None and ``pretrained`` is True the
        hub default URL is used (requires network + license access).
    pretrained : bool
        Forwarded to the hub entrypoint.
    device : int | None
        CUDA device index. None → cuda:0 if available, else cpu.
    """
    if device is None:
        device = 0
    if torch.cuda.is_available():
        torch_device = torch.device(int(device))
    else:
        torch_device = torch.device("cpu")

    kwargs = {"pretrained": bool(pretrained)}
    if weights is not None:
        kwargs["weights"] = weights

    # Skip torch.hub.load — it spawns a subprocess for the local repo lookup
    # which can stall under nix. Import the factory directly instead.
    from dinov3.hub import backbones, classifiers, segmentors, depthers, dinotxt
    factories = {}
    for mod in (backbones, classifiers, segmentors, depthers, dinotxt):
        for name in getattr(mod, "__all__", []) + [
            n for n in dir(mod) if n.startswith("dinov3_")
        ]:
            fn = getattr(mod, name, None)
            if callable(fn):
                factories[name] = fn
    if model_name not in factories:
        raise ValueError(
            f"Unknown model {model_name!r}; available: {sorted(factories)[:8]}..."
        )
    loaded_model = factories[model_name](**kwargs).to(torch_device)
    loaded_model.eval()

    info = {
        "device": str(torch_device),
        "model_name": model_name,
        "repo_dir": REPO_DIR,
    }

    processor = partial(
        process,
        processor=loaded_model,
        device=torch_device,
        expected_tile_size=expected_tile_size,
        expected_channels=expected_channels,
    )
    return processor, info


@torch.inference_mode()
def process(
    pixels: numpy.ndarray,
    processor: Callable,
    device: torch.device,
    expected_tile_size: int,
    expected_channels: int,
) -> numpy.ndarray:
    """Run DINOv3 on an NCZYX numpy array.

    DINOv3 is a rigid 3-channel ImageNet ViT. Inputs with C ≠ 3 are split
    into ``ceil(C/3)`` 3-channel chunks via
    :func:`nahual.preprocess.channel_chunks_rigid3` (recycling leading
    channels for the trailing chunk), the backbone is run on each chunk,
    and per-chunk cls tokens are concatenated along the feature axis.
    Final shape is ``(N, D · ceil(C/3))``.
    """
    if pixels.ndim != 5:
        raise ValueError(
            f"Expected NCZYX (5D) array, got shape {pixels.shape}"
        )
    _, _, _, *input_yx = pixels.shape
    validate_input_shape(input_yx, expected_tile_size)

    chunks = channel_chunks_rigid3(pixels)
    outs = []
    for chunk in chunks:
        torch_tensor = torch.from_numpy(chunk.copy()).float().to(device)
        result = processor(torch_tensor)
        # Some hub entrypoints return a dict / tuple — pick the cls token.
        if isinstance(result, dict):
            for k in ("x_norm_clstoken", "cls", "cls_token", "logits"):
                if k in result:
                    result = result[k]
                    break
        elif isinstance(result, (list, tuple)):
            result = result[0]
        if hasattr(result, "detach"):
            result = result.detach().cpu().numpy()
        outs.append(result)
    return numpy.concatenate(outs, axis=1)


async def main():
    with pynng.Rep0(listen=address, recv_timeout=300_000) as sock:
        print(f"DINOv3 server listening on {address}")
        async with trio.open_nursery() as nursery:
            responder_curried = partial(responder, setup=setup)
            nursery.start_soon(responder_curried, sock)


if __name__ == "__main__":
    try:
        trio.run(main)
    except KeyboardInterrupt:
        pass
