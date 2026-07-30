# DINOv3 Nahual OCI image

Build with `nix build .#oci-image`, then load `result` with `podman load` or
`docker load`. The resulting image is `nahual/dinov3:local` and defaults to
`tcp://0.0.0.0:5555`.

```console
podman run --rm --name nahual-dinov3 \
  --device nvidia.com/gpu=all -p 5555:5555 \
  -v nahual-dinov3-cache:/tmp/nahual nahual/dinov3:local
```

Use Docker's `--gpus all` instead of the Podman CDI option. CPU fallback works
when no GPU is exposed. Override the NNG endpoint with a container argument.

For a full forward-pass smoke test from any non-Nix Python environment:

```console
pip install 'nahual==0.0.8' numpy
NAHUAL_ADDRESS=tcp://127.0.0.1:5555 python oci/smoke_test.py
```

The smoke test uses an untrained ViT-S/16 because pretrained DINOv3 weights
require separate license acceptance and an access URL. It validates the complete
forward path and output shape; random initialization may produce non-finite values.
Mount downloaded weights under `/tmp/nahual` and pass their path through
`setup()` for meaningful production inference.
