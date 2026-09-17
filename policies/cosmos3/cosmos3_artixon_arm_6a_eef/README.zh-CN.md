# Cosmos3 ArtiXon Arm-6A EEF

该适配器连接原生 Cosmos3 X2Robot 推理服务，将三路相机、本体状态和语言指令转换为模型服务使用的输入，并返回双臂末端动作序列。

## 配置

通过环境变量覆盖原生服务地址：

```bash
export COSMOS3_X2ROBOT_SERVER_ADDRESS=127.0.0.1
export COSMOS3_X2ROBOT_SERVER_PORT=8002
```

## 启动

```bash
pip install -r policies/cosmos3/cosmos3_artixon_arm_6a_eef/requirements.txt
policy-space serve policies/cosmos3/cosmos3_artixon_arm_6a_eef/deploy.yaml
```

模型服务每次请求包含完整观测，Policy Space 服务采用 `multiplexed_serial` 会话模式并串行执行共享模型推理。

[English](README.md)
