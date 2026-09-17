# Franka Joint Example

This no-GPU, no-checkpoint example consumes `single_arm_rgb_joint@1` and returns a 24-step, 8-D zero joint target.

```bash
policy-space check policies/templates/franka_joint/deploy.yaml
policy-space serve policies/templates/franka_joint/deploy.yaml \
  --host 127.0.0.1 --port 8002
```

Use it to validate a single-arm observation/action contract and client connection, not policy performance.

[简体中文](README.zh-CN.md)
