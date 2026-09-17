# OpenPI (π0.5) ArtiXon Arm-6A EEF

该 Policy Space 服务在进程中直接加载面向 ArtiXon Arm-6A 桌面双臂的 OpenPI π0.5 checkpoint。ManaEnv 线上协议仍为 14 维双臂 episode-init-relative EE pose：

```text
[left_xyz, left_rpy, left_gripper, right_xyz, right_rpy, right_gripper]
```

适配器在 OpenPI 边界转换为训练 checkpoint 的 20 维模型表示：

```text
[left_xyz, left_rot6d, left_gripper, right_xyz, right_rot6d, right_gripper]
```

仿真客户端根据本体契约，将 14 维 Euler 轨迹转换成目标机器人可执行的控制命令。

## 准备环境

```bash
export OPENPI_ROOT=/path/to/openpi
cd "$OPENPI_ROOT"
GIT_LFS_SKIP_SMUDGE=1 uv sync
cd /path/to/policy-space
uv pip install --python "$OPENPI_ROOT"/.venv/bin/python -e .
```

OpenPI checkpoint 必须使用 20 维双臂 EE pose state/action 训练，并与 `pi05_x2real_lerobotv2_finetune` 的 PI0.5 模型形状兼容。若 manifest 准备出的 OpenPI 版本还没有这个训练工作区配置，适配器会回退到推理阶段等价的 `pi05_ex001_6r_joint_finetune`：两者使用相同的 PI0.5 模型、三相机 repack、ALOHA transform 和 `adapt_to_pi=false`。

OpenPI 当前的通用 `AlohaOutputs` 会固定只返回前 14 个通道。该策略加载时会验证 output transform 中恰好存在一个 `AlohaOutputs`，将它替换为保留完整 20 个 X2 EE 通道的输出层，再把 rotation-6D 投影为合法旋转并转回 Euler14。启动时还会验证 checkpoint 同时带有 20 维 state/action quantile normalization stats。若上游 transform 或 stats 结构变化，服务会拒绝启动，避免静默错位。

默认 X2Real checkpoint 保存的是旋转矩阵前两行，因此部署默认使用 `rotation_layout: row`。只有 checkpoint 明确使用标准的前两列表示训练时，才设置 `OPENPI_ROTATION_LAYOUT=column`。

PolicySpace 默认关闭 PI0.5 的 `torch.compile(max-autotune)`，避免共享文件系统上的 Triton 编译缓存导致首次请求耗时数分钟或出现 stale file handle。需要编译优化时可设置 `pytorch_compile_mode`，并将编译缓存放到节点本地磁盘。

## 启动

```bash
export OPENPI_ROOT=/path/to/openpi
OPENPI_CHECKPOINT_PATH=/path/to/openpi-pi05-artixon-arm-6a-checkpoint \
  "$OPENPI_ROOT"/.venv/bin/policy-space serve \
  policies/openpi/pi05_artixon_arm_6a_eef/deploy.yaml \
  --port 8001
```

也可以通过环境变量覆盖：

- `OPENPI_TRAIN_CONFIG`
- `OPENPI_CHECKPOINT_PATH`
- `OPENPI_DEVICE`
- `OPENPI_ROTATION_LAYOUT`（默认 `row`，可选 `column`）
- `OPENPI_PYTORCH_COMPILE_MODE`（默认关闭；可选 `default`、`reduce-overhead`、`max-autotune` 或 `max-autotune-no-cudagraphs`）

若 Policy Space 服务不在本机，在仿真客户端的运行配置中覆盖服务地址。

[English](README.md)
