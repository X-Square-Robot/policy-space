# Policy Space 文档

Policy Space 连接仿真客户端与策略服务。仿真客户端负责环境、机器人执行和评测；策略服务负责模型加载、模型输入转换、推理及策略内部状态。

## 第一次使用

1. 按照仓库 README 中的快速开始命令启动无 GPU 示例。
2. 按照[策略接入指南](policy-integration.zh-CN.md)接入自己的模型。
3. 按照[ManaEnv 评测指南](manaenv-evaluation.zh-CN.md)连接仿真客户端。

## 按问题查找

- 需要了解客户端、Policy Space 与策略服务的运行边界：阅读[架构说明](architecture.zh-CN.md)。
- 不清楚 observation、action 或 embodiment schema：阅读[契约说明](contracts.zh-CN.md)。
- 需要配置模型环境、端口或多客户端模式：阅读[部署说明](deployment.zh-CN.md)。
- 需要固定安装版本或发布 wheel：阅读[安装与发布](release.zh-CN.md)。
- 服务无法启动、连接或推理：阅读[排障指南](troubleshooting.zh-CN.md)。

## 仓库目录

```text
policy_space/
├── docs/                       # 用户文档
├── policies/                   # 正式策略与参考模板
│   └── templates/              # 无权重、无 GPU 的可运行模板
├── src/policy_space/           # 协议、服务端、客户端和公共运行时
│   └── contracts/schemas/      # 版本化的公共契约
└── tests/                      # 协议与兼容性测试
```

策略目录见 [`policies/README.zh-CN.md`](../policies/README.zh-CN.md)。每个正式策略通常包含 `model.py`、`deploy.yaml`、`requirements.txt` 和 `README.md`。

[English](README.md)
