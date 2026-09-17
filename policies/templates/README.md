# Policy Templates

These templates implement the complete Policy Space v2 episode lifecycle without model weights or GPUs. They validate integration code and transport behavior; they do not represent policy performance.

- [bimanual_ee](bimanual_ee/README.md): `bimanual_rgb_ee@1` → `bimanual_ee_absolute@1`.
- [franka_joint](franka_joint/README.md): `single_arm_rgb_joint@1` → `single_arm_joint_position@1`.
- [mobile_manipulation](mobile_manipulation/README.md): `mobile_bimanual_rgb@1` → `mobile_semantic_20d@1`.

Start with the bimanual template:

```bash
policy-space check policies/templates/bimanual_ee/deploy.yaml
policy-space serve policies/templates/bimanual_ee/deploy.yaml \
  --host 127.0.0.1 --port 8001
```

[简体中文](README.zh-CN.md)
