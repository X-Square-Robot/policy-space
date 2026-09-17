# 契约说明

契约让客户端和策略服务在推理前确认双方使用相同的观测、动作和机器人本体约定。

## 契约类型

- `src/policy_space/contracts/schemas/protocol/`：消息和 episode 生命周期版本。
- `src/policy_space/contracts/schemas/observations/`：观测字段、形状和语义。
- `src/policy_space/contracts/schemas/actions/`：动作维度、布局和表示方式。
- `src/policy_space/contracts/schemas/embodiments/`：机器人本体及其约束。

这些 YAML 文件作为 `policy_space.contracts` 的包数据安装，因此源码运行和安装后使用相同的内置契约。

契约标识采用 `name@version`，例如：

```yaml
observation_schema: bimanual_rgb_joint@1
action_schema: bimanual_joint_position@1
embodiment_constraints: [ex001_6r@1]
```

## 策略声明

策略在 `deploy.yaml` 的 `metadata.supported_schema_pairs` 中声明支持的组合。一个策略可以声明多组组合，客户端在 `initialize_episode` 时选择其中一组。

```yaml
metadata:
  supported_schema_pairs:
    - observation_schema: bimanual_rgb_joint@1
      action_schema: bimanual_joint_position@1
      embodiment_constraints: [ex001_6r@1]
```

## 修改原则

- 只增加可选字段且不改变原语义时，可在同一版本中扩展。
- 改变字段含义、动作布局或单位时，创建新版本。
- 不在模型代码和客户端中分别硬编码同一套约定；公共约定放入契约。
- 修改后运行 `policy-space check <deploy.yaml>`，并执行客户端联调。

[English](contracts.md)
