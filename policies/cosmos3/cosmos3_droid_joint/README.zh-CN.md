# Cosmos3 DROID Joint

该服务直接加载 `Cosmos3-Nano-Policy-DROID`，将 Policy Space 观测转换为模型输入，并返回 `(32, 8)` 的绝对关节动作序列。

- `model.py`：模型加载、输入转换与推理。
- `deploy.yaml`：模型和服务配置。

设置 `COSMOS_ROOT`，按需设置 `COSMOS3_CHECKPOINT_PATH`，在 Cosmos 环境中安装 Policy Space 后运行：

```bash
policy-space check policies/cosmos3/cosmos3_droid_joint/deploy.yaml
policy-space serve policies/cosmos3/cosmos3_droid_joint/deploy.yaml --port 8001
```

当前服务不内置认证或 TLS。远程部署时应在网络边界限制访问来源。

[English](README.md)
