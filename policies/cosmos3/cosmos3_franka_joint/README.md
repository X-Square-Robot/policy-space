# Cosmos3 Franka Joint

This bundle loads the Cosmos3 Franka checkpoint in-process and exposes only the Policy Space v2 lifecycle. It is independent of ManaEnv and does not require a second model-specific WebSocket server.

## Contract

- Observation: three HWC RGB `uint8` images (`wrist`, `exterior_1`, `exterior_2`), a 7D joint vector, a 1D gripper vector, and a non-empty task instruction.
- Model output: exact finite `float32[32,8]` absolute joint targets.
- Policy Space output: the first 24 rows as exact finite `float32[24,8]`.
- Rate: 20 Hz. Gripper convention: raw `0=open`, `1=closed`.

The adapter checks both H32 and H24. A checkpoint or model change that alters the source horizon fails immediately; the generic runtime also rejects any wire output whose horizon differs from `metadata.action_horizon`.

## Configuration and launch

Set all model-owned paths explicitly:

```bash
export COSMOS3_FRAMEWORK=/path/to/cosmos-framework
export COSMOS3_CHECKPOINT_PATH=/path/to/checkpoint
export COSMOS3_QWEN_TOKENIZER_PATH=/path/to/Qwen3-VL-8B-Instruct
export COSMOS3_OUTPUT_DIR=/path/to/writable/output
policy-space check policies/cosmos3/cosmos3_franka_joint/deploy.yaml
policy-space serve policies/cosmos3/cosmos3_franka_joint/deploy.yaml --port 8001
```

`COSMOS3_INSTRUCTION` is an optional fallback, but normal requests must carry the current episode instruction.

The verified probe used three `uint8[540,640,3]` images and returned finite `float32[24,8]` after the public H24 truncation.

[简体中文](README.zh-CN.md)
