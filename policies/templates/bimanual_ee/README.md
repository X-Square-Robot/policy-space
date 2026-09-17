# Bimanual EE Example

This no-GPU, no-checkpoint example validates the Policy Space v2 lifecycle, contract negotiation, and service transport. It consumes `bimanual_rgb_ee@1` and returns a 32-step, 14-D zero-action sequence.

```bash
policy-space check policies/templates/bimanual_ee/deploy.yaml
policy-space serve policies/templates/bimanual_ee/deploy.yaml --host 127.0.0.1 --port 8001
```

Copy this directory and replace the preprocessing and inference code in `model.py` when integrating a real model.

[简体中文](README.zh-CN.md)
