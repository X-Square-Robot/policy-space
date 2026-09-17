# Mobile Manipulation Example

This no-GPU, no-checkpoint Quanta X1 example returns a 20-D whole-body zero-action sequence with wire metadata. It validates transport for the mobile base, bimanual arms, and remaining whole-body fields.

```bash
policy-space check policies/templates/mobile_manipulation/deploy.yaml
policy-space serve policies/templates/mobile_manipulation/deploy.yaml \
  --host 127.0.0.1 --port 8001
```

All actions are zero, so this example is intended only for integration testing.

[简体中文](README.zh-CN.md)
