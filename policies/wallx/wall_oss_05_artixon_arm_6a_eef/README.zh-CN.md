# Wall-X (wall-oss-0.5) ArtiXon Arm-6A EEF

该服务在一个进程中加载 Wall-X checkpoint，并向 ManaEnv 及其他兼容客户端暴露 Policy Space v2 协议。面向 ArtiXon Arm-6A 桌面末端契约。

```text
ManaEnv / 客户端 -> Policy Space service -> Wall-X policy
```

默认的 `standard` 后端使用 [Wall-X](https://github.com/X-Square-Robot/wall-x) 的推理接口。按 Wall-X README 准备推理环境和 CUDA 依赖，并在同一环境中安装 Policy Space；模型权重见 [wall-oss-0.5](https://huggingface.co/x-square-robot/wall-oss-0.5)。适配验证使用的 Wall-X 版本为 commit `ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f`。

对于较早的 Wall-X 运行环境，可通过 `WALLX_BACKEND=legacy` 或 `model.cfg.backend: legacy` 选择对应后端，并安装该运行环境所需的依赖。`standard` 后端不支持非默认 CFG 和 Wall-X RTC，传入这些选项时会明确报错。

## 启动

```bash
export WALLX_CHECKPOINT_PATH=/path/to/checkpoint
export WALLX_ROOT=/path/to/wall-x
export WALLX_BACKEND=standard
policy-space check policies/wallx/wall_oss_05_artixon_arm_6a_eef/deploy.yaml
policy-space serve policies/wallx/wall_oss_05_artixon_arm_6a_eef/deploy.yaml --port 8001
```

服务启动后接受 `initialize_episode`、`ingest_observation`、`infer_actions` 和 `finalize_episode`。

Wall-X 输出 33 行，其中首行为当前状态。ManaEnv 跳过首行，将 32 行预测动作按 1.5 倍时间插值为 48 个控制步。Observation mapping、动作适配和 RTC 调度位于 ManaEnv 客户端；模型预处理和推理位于 Policy Space 服务。

[English](README.md)
