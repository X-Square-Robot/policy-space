# Cosmos3 DROID Joint

The service loads `Cosmos3-Nano-Policy-DROID` directly, maps Policy Space observations to the model input, and returns the `(32, 8)` absolute-joint action chunk.

Current implementation files:

- `model.py`: model loading, input conversion, and inference.
- `deploy.yaml`: model and service configuration.

Set `COSMOS_ROOT` and optionally `COSMOS3_CHECKPOINT_PATH`, install Policy Space in the Cosmos environment, then run:

```bash
policy-space check policies/cosmos3/cosmos3_droid_joint/deploy.yaml
policy-space serve policies/cosmos3/cosmos3_droid_joint/deploy.yaml --port 8001
```

The current service does not provide built-in authentication or TLS. Restrict inbound sources at the network boundary when exposing it remotely.

[简体中文](README.zh-CN.md)
