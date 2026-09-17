# Bimanual EE Example

这是无需 GPU 和 checkpoint 的双臂末端动作示例，用于检查 Policy Space v2 生命周期、契约协商和服务通信。

它接收 `bimanual_rgb_ee@1` 观测，并返回 32 步、14 维的零动作序列。该示例只验证接入链路，不用于测量策略成功率。

```bash
policy-space check policies/templates/bimanual_ee/deploy.yaml
policy-space serve policies/templates/bimanual_ee/deploy.yaml --host 127.0.0.1 --port 8001
```

接入真实模型时，可复制此目录并替换 `model.py` 中的输入处理与推理逻辑。

[English](README.md)
