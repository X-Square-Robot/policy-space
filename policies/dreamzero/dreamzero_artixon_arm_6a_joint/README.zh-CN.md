# DreamZero ArtiXon Arm-6A Joint

该接入在 Policy Space 服务进程中加载 DreamZero 双臂关节策略。模型预处理、因果视觉历史和内部状态均由 `model.py` 维护，ManaEnv 负责持续发送真实观测并执行返回的动作序列。

## 准备环境

DreamZero 源码和 checkpoint 保持在本仓库之外：

```bash
export DREAMZERO_ROOT=/path/to/dreamzero
export DREAMZERO_CHECKPOINT_PATH=/path/to/checkpoint
```

tokenizer、device、历史窗口和执行长度等默认值位于 `deploy.yaml`，需要时可通过对应的 `DREAMZERO_*` 环境变量覆盖。

## 会话模式

该策略维护因果历史和模型状态，使用 `session_mode: exclusive`。一个服务进程同一时间只运行一个客户端 episode；并行评测需要在不同端口启动多个服务进程。

## 检查与启动

```bash
pip install -e /path/to/policy-space
policy-space check policies/dreamzero/dreamzero_artixon_arm_6a_joint/deploy.yaml
policy-space serve policies/dreamzero/dreamzero_artixon_arm_6a_joint/deploy.yaml --port 8001
```

`initialize_episode` 清理旧历史，`ingest_observation` 加入真实观测，`infer_actions` 生成后续动作，`finalize_episode` 释放本 episode 的状态。

[English](README.md)
