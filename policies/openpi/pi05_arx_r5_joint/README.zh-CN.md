# OpenPI (π0.5) ARX R5 Joint

该部署加载 `pi05_arx_r5_joint` OpenPI checkpoint，面向 `arx_r5@1` 本体提供 PolicySpace `bimanual_rgb_joint@1` -> `bimanual_joint_position@1` 契约。

## 契约

- 观测：三路 uint8 HWC RGB 相机（`camera_front`、`camera_left`、`camera_right`），分别映射到 OpenPI 的 `observation/image`、`observation/left_wrist_image`、`observation/right_wrist_image`。
- 状态：`follow1_pos` 与 `follow2_pos` 各 7 维，拼接为 14 维 `observation/state`，左臂在前。
- 动作：绝对关节目标，`(T, 14)` float32，`T` 最大 32，控制频率 20 Hz。每臂六个关节加一个归一化夹爪，夹爪采用 `normalized_01`。
- 必须提供非空指令。优先级为单帧观测 `instruction` > episode 指令 > 配置默认值。
- 返回的动作块会校验形状与有限性，并按 `action_horizon` 截断。

## 环境

需要在 OpenPI 依赖环境中运行。以下环境变量可覆盖 YAML 中的取值，无需修改被跟踪的配置文件：

```bash
export OPENPI_ROOT=/path/to/openpi
export OPENPI_CHECKPOINT_PATH=/path/to/checkpoint
export OPENPI_TRAIN_CONFIG=pi05_arx_r5_joint
export OPENPI_DEVICE=cuda:0
export PYTHONPATH="$PWD/src:$PWD/packages/protocol/src:$PWD"
```

相对路径的 `OPENPI_CHECKPOINT_PATH` 会基于 `OPENPI_ROOT` 解析。train config 的 `action_horizon` 必须为 32，其他取值会被加载器拒绝。

## 检查

```bash
python -m policy_space.cli check policies/openpi/pi05_arx_r5_joint/deploy.yaml
```

## 启动服务

```bash
CUDA_VISIBLE_DEVICES=0 \
  python -m policy_space.cli serve policies/openpi/pi05_arx_r5_joint/deploy.yaml \
  --host 0.0.0.0 --port 8001
```

客户端 schema 选择：

```yaml
observation_schema: bimanual_rgb_joint@1
action_schema: bimanual_joint_position@1
embodiment_schema: arx_r5@1
```

由于每次推理返回 32 步动作块，服务声明 `recommended_ingest_observation_each_step: false` 与 `recommended_connect_chunks: true`。握手与有限动作块只验证接入是否正确，不代表任务成功率，策略效果仍需 rollout 评测。

[English](README.md)
