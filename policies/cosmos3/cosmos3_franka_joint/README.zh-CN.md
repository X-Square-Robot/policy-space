# Cosmos3 Franka Joint

该 bundle 在进程内加载 Cosmos3 Franka checkpoint，仅暴露 Policy Space v2 lifecycle。它与 ManaEnv 独立，也不再需要单独维护模型专用 WebSocket server。

## 契约

- 输入：`wrist`、`exterior_1`、`exterior_2` 三路 HWC RGB `uint8` 图像，7 维关节、1 维夹爪，以及非空任务指令。
- 模型原始输出：严格有限的 `float32[32,8]` 绝对关节目标。
- Policy Space 输出：取前 24 行，严格为有限的 `float32[24,8]`。
- 控制频率：20 Hz；夹爪定义为原始 `0=打开`、`1=闭合`。

适配器同时校验模型 H32 和对外 H24。checkpoint 或模型改动导致原始 horizon 变化时会立即失败；通用 runtime 还会强制输出 horizon 等于 `metadata.action_horizon`。

## 配置与启动

```bash
export COSMOS3_FRAMEWORK=/path/to/cosmos-framework
export COSMOS3_CHECKPOINT_PATH=/path/to/checkpoint
export COSMOS3_QWEN_TOKENIZER_PATH=/path/to/Qwen3-VL-8B-Instruct
export COSMOS3_OUTPUT_DIR=/path/to/writable/output
policy-space check policies/cosmos3/cosmos3_franka_joint/deploy.yaml
policy-space serve policies/cosmos3/cosmos3_franka_joint/deploy.yaml --port 8001
```

`COSMOS3_INSTRUCTION` 可作为可选默认指令，正常请求仍应携带当前 episode 的真实指令。已验证探针使用三路 `uint8[540,640,3]` 图像，对外结果为有限的 `float32[24,8]`。

[English](README.md)
