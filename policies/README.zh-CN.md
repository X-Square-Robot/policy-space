# 策略目录

`policies/` 是 Policy Space 的策略接入入口。正式接入按策略家族组织，每个具体接入目录保存模型适配、部署配置、依赖和独立使用说明。

## 正式策略

- [Cosmos3](cosmos3/README.zh-CN.md)：[DROID 关节](cosmos3/cosmos3_droid_joint/README.zh-CN.md)、[Franka 关节](cosmos3/cosmos3_franka_joint/README.zh-CN.md)、[ArtiXon Arm-6A 末端](cosmos3/cosmos3_artixon_arm_6a_eef/README.zh-CN.md)
- [DreamZero](dreamzero/README.zh-CN.md)：[DROID 关节](dreamzero/dreamzero_droid_joint/README.zh-CN.md)、[ArtiXon Arm-6A 关节](dreamzero/dreamzero_artixon_arm_6a_joint/README.zh-CN.md)
- [DM0.5](dm05/README.zh-CN.md)：ArtiXon Arm-6A 关节与[末端策略](dm05/dm05_artixon_arm_6a_eef/README.zh-CN.md)。
- [FastWAM](fastwam/README.zh-CN.md)：[ArtiXon Arm-6A 关节策略](fastwam/fastwam_artixon_arm_6a_joint/README.zh-CN.md)
- [OpenPI](openpi/README.zh-CN.md)：π0.5 的 ArtiXon Arm-6A 末端 / 关节、Quanta X1 whole body、[ARX R5 末端](openpi/pi05_arx_r5_eef/README.zh-CN.md)及 [ARX R5 关节](openpi/pi05_arx_r5_joint/README.zh-CN.md)。
- [Wall-X](wallx/README.zh-CN.md)：wall-oss-0.5（ArtiXon Arm-6A 末端）及 Quanta X1 全身。
- [SmolVLA](smolvla/README.zh-CN.md)：[Franka 关节控制](smolvla/smolvla_franka_joint/README.zh-CN.md)。

## 参考模板

[templates/](templates/README.zh-CN.md) 提供三个不依赖 checkpoint 和 GPU 的可运行模板，用于检查契约、episode 生命周期和 ManaEnv 通信链路。

## 接入文件

一个常规策略接入包含：

```text
policies/my_family/my_policy/
├── model.py
├── deploy.yaml
├── requirements.txt
└── README.md
```

新策略应放入对应家族目录；新家族可以直接在 `policies/` 下建立。完整接入过程见[策略接入指南](../docs/policy-integration.zh-CN.md)。

## 模型与本体支持

| 模型 | 本体 / 控制模式 | 接入文档 |
|---|---|---|
| Cosmos3 | DROID / Franka / ArtiXon Arm-6A | [cosmos3](cosmos3/README.zh-CN.md) |
| DreamZero | DROID / ArtiXon Arm-6A | [dreamzero](dreamzero/README.zh-CN.md) |
| DM0.5 | ArtiXon Arm-6A 关节 / 末端 | [dm05](dm05/README.zh-CN.md) |
| FastWAM | ArtiXon Arm-6A 关节 | [fastwam](fastwam/README.zh-CN.md) |
| OpenPI π0.5 | ArtiXon Arm-6A / Quanta X1 / 关节；ARX R5 末端 / 关节 | [openpi](openpi/README.zh-CN.md) |
| Wall-X | ArtiXon Arm-6A / Quanta X1 | [wallx](wallx/README.zh-CN.md) |
| SmolVLA | Franka 关节 | [smolvla](smolvla/README.zh-CN.md) |

[English](README.md)
