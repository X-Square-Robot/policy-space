# 策略模板

这些模板实现完整的 Policy Space v2 episode 生命周期，不加载训练权重，也不需要 GPU。它们用于接入开发和链路检查，不代表真实策略性能。

- [bimanual_ee](bimanual_ee/README.zh-CN.md)：双臂末端位姿，`bimanual_rgb_ee@1` → `bimanual_ee_absolute@1`。
- [franka_joint](franka_joint/README.zh-CN.md)：Franka 单臂关节，`single_arm_rgb_joint@1` → `single_arm_joint_position@1`。
- [mobile_manipulation](mobile_manipulation/README.zh-CN.md)：Quanta X1 移动操作，`mobile_bimanual_rgb@1` → `mobile_semantic_20d@1`。

首次使用建议从双臂模板开始：

```bash
policy-space check policies/templates/bimanual_ee/deploy.yaml
policy-space serve policies/templates/bimanual_ee/deploy.yaml \
  --host 127.0.0.1 --port 8001
```

[English](README.md)
