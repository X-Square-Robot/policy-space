# Franka Joint Example

这是无需 GPU 和 checkpoint 的 Franka 单臂关节动作示例。它接收 `single_arm_rgb_joint@1` 观测，并返回 24 步、8 维的零关节目标。

```bash
policy-space check policies/templates/franka_joint/deploy.yaml
policy-space serve policies/templates/franka_joint/deploy.yaml \
  --host 127.0.0.1 --port 8002
```

该示例适合检查单臂 observation/action contract 和 ManaEnv 客户端连接，不代表实际策略性能。

[English](README.md)
