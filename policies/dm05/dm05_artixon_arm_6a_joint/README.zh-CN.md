# DM0.5 ArtiXon Arm-6A Joint

这是独立的 Policy Space v2 bundle，通过轻量 HTTP adapter 连接 OpenDM 推理服务。只有 GPU backend 加载 OpenDM、CUDA、checkpoint 和 `norm_stats.json`；ManaEnv 不再需要模型专用注册。

## 契约

- 输入：头部、左腕、右腕三路 HWC RGB `uint8` 图像，两组有限的 7 维关节/夹爪状态，以及非空的当前任务指令。
- 后端请求：图像依次映射到 OpenDM 的 `1`、`2`、`3` 槽位；状态按左 7 维、右 7 维拼接；图像保持原始分辨率，编码为不带 `data:` 前缀的 PNG/base64。
- 后端输出：严格有限的 `float32[50,14]` 绝对关节目标。
- 对外输出：严格有限的 `float32[50,26]`；0:14 为真实动作，14:26 为只用于记录的 master-EE 槽位并补零。
- 控制频率：20 Hz；夹爪使用归一化 `[0,1]`。

adapter 在封装前强制 H50；通用 runtime 还会独立强制对外 horizon 等于 `metadata.action_horizon`。

## 后端配置

`backend_url` 必须是完整的 HTTP(S) `/v1/infer` 地址。以下环境变量覆盖 `deploy.yaml`：

- `DM05_X2REAL_BACKEND_URL`
- `DM05_X2REAL_CONNECT_TIMEOUT_SECONDS`
- `DM05_X2REAL_READ_TIMEOUT_SECONDS`
- `DM05_X2REAL_CHECKPOINT_ID`
- `DM05_X2REAL_SAMPLING_SEED`

网关不会自动重试 POST，也不会回退到旧动作。HTTP 错误、超时、非法 JSON、错误形状或非有限值都会让当前请求直接失败。

```bash
export DM05_X2REAL_BACKEND_URL=http://<gpu-host>:8002/v1/infer
export DM05_X2REAL_CHECKPOINT_ID=<opendm-checkpoint-id>
policy-space check policies/dm05/dm05_artixon_arm_6a_joint/deploy.yaml
policy-space serve policies/dm05/dm05_artixon_arm_6a_joint/deploy.yaml --port 8001
```

已验证探针从正式 checkpoint 得到有限的原始 `float32[50,14]` 和封装后的 `float32[50,26]` 动作。

[English](README.md)
