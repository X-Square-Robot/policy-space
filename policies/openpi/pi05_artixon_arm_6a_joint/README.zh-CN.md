# OpenPI (π0.5) ArtiXon Arm-6A Joint

该部署加载 `pi05_x2real_split280` JAX checkpoint，并提供 PolicySpace `bimanual_rgb_joint@1` 到 `bimanual_joint_position@1` 协议。模型输入和输出均为 14 维：每条机械臂包含六个关节和一个归一化夹爪通道。OpenPI 会在模型内部补齐到预训练模型使用的 32 维宽度。

适配器将 PolicySpace 的三个相机映射到 OpenPI 的前视、左腕和右腕图像字段，拼接 `follow1_pos` 与 `follow2_pos`，并将 episode 指令作为 OpenPI prompt。返回的 action chunk 必须为 `(T, 14)`，并且只能包含有限的绝对关节目标值。

## 环境

使用 OpenPI 的依赖环境运行该策略。以下环境变量可以覆盖部署 YAML 中的配置：

```bash
export OPENPI_ROOT=/path/to/openpi
export OPENPI_CHECKPOINT_PATH=/path/to/checkpoint
export OPENPI_TRAIN_CONFIG=pi05_x2real_split280
export OPENPI_DEVICE=cuda:0
export PYTHONPATH="$PWD/src:$PWD/packages/protocol/src:$PWD"
```

在新机器上部署时，需要将 Policy Space 以 editable 模式安装到 OpenPI 环境中。OpenPI 和 checkpoint 专用依赖不要加入 Policy Space 核心环境定义。

## 验证

```bash
python -m policy_space.cli check policies/openpi/pi05_artixon_arm_6a_joint/deploy.yaml
python -m pytest -q tests/test_openpi_pi05_x2real_joint_policy.py
```

## 启动

```bash
CUDA_VISIBLE_DEVICES=0 \
  python -m policy_space.cli serve policies/openpi/pi05_artixon_arm_6a_joint/deploy.yaml \
  --host 0.0.0.0 --port 8001
```

服务声明 `recommended_ingest_observation_each_step: false`，因为每次推理返回 32 步 action chunk。兼容的评测器因此可以执行向量化环境，同时服务端仍以串行方式执行模型推理。

[English](README.md)
