# FastWAM ArtiXon Arm-6A Joint

该 bundle 在进程内加载 FastWAM joint-DiT checkpoint，并直接提供 Policy Space v2 lifecycle。它与 ManaEnv 独立，图像预处理、归一化和动作封装均由本 bundle 负责。

## 契约

- 输入：头部、左腕、右腕三路 HWC RGB `uint8` 图像，两组有限的 7 维关节/夹爪状态，以及当前任务指令。
- 预处理：头部缩放为 320x256，两路腕部各为 160x128，拼成训练使用的 320x384 RobotWin mosaic；默认 9 个 video frames、10 个 inference steps。
- 模型输出：严格有限的 `float32[32,14]` 绝对关节目标。
- 对外输出：严格有限的 `float32[32,26]`；0:14 为真实动作，14:26 为只用于记录的 master-EE 槽位并补零。
- 控制频率：20 Hz；夹爪使用归一化 `[0,1]`。

适配器与通用 runtime 都强制 H32，因此模型变化不会静默破坏 metadata 契约。

```bash
export FASTWAM_ROOT=/path/to/FastWAM
export FASTWAM_TRAINING_CONFIG_PATH=/path/to/training/config.yaml
export FASTWAM_CHECKPOINT_PATH=/path/to/weights/checkpoint.pt
export FASTWAM_DATASET_STATS_PATH=/path/to/dataset_stats.json
policy-space check policies/fastwam/fastwam_artixon_arm_6a_joint/deploy.yaml
policy-space serve policies/fastwam/fastwam_artixon_arm_6a_joint/deploy.yaml --port 8001
```

已验证部署曾连续返回有限的 `float32[32,26]`。更换 checkpoint、训练配置或统计文件后，必须在 FastWAM 模型环境中重新运行 `policy-space check`。

[English](README.md)
