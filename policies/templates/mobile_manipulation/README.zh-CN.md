# Mobile Manipulation Example

这是无需 GPU 和 checkpoint 的 Quanta X1 移动操作示例。它返回带 wire metadata 的 20 维全身零动作序列，用于检查移动底盘、双臂和其他全身动作字段的完整通信链路。

```bash
policy-space check policies/templates/mobile_manipulation/deploy.yaml
policy-space serve policies/templates/mobile_manipulation/deploy.yaml \
  --host 127.0.0.1 --port 8001
```

所有动作值均为零，因此该示例只用于链路联调，不用于成功率评测。

[English](README.md)
