# Wall-X Quanta X1 Whole Body

该目录是模型特定的部署适配器，将 Wall-X 原生 Quanta X1 序列化输出转换为公共 `manaenv_ex001_wholebody_policy_wire_20d_v1` wire，不包含仿真转换或模型权重。

原生 Wall-X 服务应只在推理主机内部访问。原生服务就绪后再启动网关：

```bash
WALLX_NATIVE_ADDRESS=127.0.0.1 WALLX_NATIVE_PORT=8002 \
  policy-space serve policies/wallx/wallx_quanta_x1_whole_body/deploy.yaml \
  --host 0.0.0.0 --port 8001
```

评测器通过运行参数注入服务地址。

20 维协议布局：

```text
follow1_pos[7] | follow2_pos[7] | velocity_decomposed_odom[3] | lift[1] | head_pos[2]
```

初始位姿组合、Euler 到 quaternion、夹爪缩放和轮速转换均保留在 ManaEnv。

[English](README.md)
