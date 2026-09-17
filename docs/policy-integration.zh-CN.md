# 策略接入指南

一个策略接入负责两件事：把公共观测转换为模型输入，以及把模型输出整理为契约声明的动作。

## 1. 创建目录

下面以 OpenPI 家族中的自定义 π0.5 适配包为例。该命令只生成接入骨架，不下载模型权重，也不修改已有策略。

```bash
policy-space init pi05_custom \
  --family openpi \
  --observation bimanual_rgb_joint@1 \
  --action bimanual_joint_position@1
```

- `pi05_custom` 是新策略接入包的名称。
- `--family openpi` 将该策略归入 OpenPI 模型家族。
- `--observation` 指定策略接收的版本化观测契约。
- `--action` 指定策略输出的版本化动作契约。

生成内容如下：

```text
policies/openpi/pi05_custom/
├── model.py
├── deploy.yaml
├── requirements.txt
└── README.md
```

请根据策略家族和机器人配置选择最接近的已有实现：

- **OpenPI / π0.5：**[ArtiXon Arm-6A 末端](../policies/openpi/pi05_artixon_arm_6a_eef/)、[Quanta X1 whole body](../policies/openpi/pi05_quanta_x1_whole_body/)、[ArtiXon Arm-6A 关节](../policies/openpi/pi05_artixon_arm_6a_joint/)、[ARX R5 末端](../policies/openpi/pi05_arx_r5_eef/)、[ARX R5 关节](../policies/openpi/pi05_arx_r5_joint/)。
- **Wall-X：**[wall-oss-0.5 ArtiXon Arm-6A 末端](../policies/wallx/wall_oss_05_artixon_arm_6a_eef/)、[Quanta X1 whole body](../policies/wallx/wallx_quanta_x1_whole_body/)。
- **DreamZero：**[ArtiXon Arm-6A 关节](../policies/dreamzero/dreamzero_artixon_arm_6a_joint/)、[DROID 关节](../policies/dreamzero/dreamzero_droid_joint/)。
- **Cosmos3：**[ArtiXon Arm-6A 末端](../policies/cosmos3/cosmos3_artixon_arm_6a_eef/)、[DROID 关节](../policies/cosmos3/cosmos3_droid_joint/)、[Franka 关节](../policies/cosmos3/cosmos3_franka_joint/)。
- **DM0.5：**[ArtiXon Arm-6A 关节](../policies/dm05/dm05_artixon_arm_6a_joint/)、[ArtiXon Arm-6A 末端](../policies/dm05/dm05_artixon_arm_6a_eef/)。
- **FastWAM：**[ArtiXon Arm-6A 关节](../policies/fastwam/fastwam_artixon_arm_6a_joint/)。
- **SmolVLA：**[Franka 关节](../policies/smolvla/smolvla_franka_joint/)。

## 2. 实现 episode 生命周期

```python
from policy_space import PolicyModel


class Model(PolicyModel):
    def initialize_episode(self, episode_info=None):
        pass

    def ingest_observation(self, observation):
        self.observation = observation

    def infer_actions(self):
        return self.policy.predict(self.observation)

    def finalize_episode(self, episode_info=None):
        pass
```

- `initialize_episode`：准备一次新评测所需的状态。
- `ingest_observation`：接收当前观测，并按需更新历史或内部状态。
- `infer_actions`：返回一个动作或形状为 `[T, D]` 的动作序列。
- `finalize_episode`：清理本 episode 的状态。

## 3. 配置部署

在 `deploy.yaml` 中填写模型类、模型参数、监听地址、会话模式及支持的契约：

```yaml
policy_name: pi05_custom

model:
  class: Model
  cfg:
    checkpoint_path: /path/to/checkpoint

server:
  host: 0.0.0.0
  port: 8001
  session_mode: exclusive

metadata:
  supported_schema_pairs:
    - observation_schema: bimanual_rgb_joint@1
      action_schema: bimanual_joint_position@1
      embodiment_constraints: [ex001_6r@1]
```

`host` 是当前 Policy Space 服务实例的监听地址。ManaEnv 位于另一台机器时，应将服务绑定到 ManaEnv 可以访问的网络接口；`0.0.0.0` 会监听全部网络接口，是常见写法，也可以填写指定网卡地址以缩小监听范围。服务只允许同机连接时，可使用 `127.0.0.1`。`port` 是 WebSocket 服务端口。ManaEnv 连接地址写为 `ws://<policy-service-host>:<port>`，其中 `<policy-service-host>` 是运行当前策略对应 Policy Space 服务实例的机器可被 ManaEnv 访问的 IP 或主机名；在上面的示例中，`<port>` 为 `8001`。`0.0.0.0` 只用于服务端监听，不能作为客户端连接地址。

契约字段的含义见[契约说明](contracts.zh-CN.md)，会话模式见[部署说明](deployment.zh-CN.md)。

## 4. 检查并启动

```bash
policy-space check policies/openpi/pi05_custom/deploy.yaml
policy-space serve policies/openpi/pi05_custom/deploy.yaml
```

最后在 `policies/openpi/pi05_custom/README.md` 中记录模型源码、权重准备、环境变量、检查命令和启动命令。可参考 `policies/openpi/pi05_quanta_x1_whole_body/`，也可先从 `policies/templates/bimanual_ee/` 的最小实现开始。

[English](policy-integration.md)
