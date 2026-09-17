# OpenPI (π0.5) Quanta X1 Whole Body

该 Policy Space 适配器调用 Quanta X1 原生 π0.5 推理服务，并将具名输出封装为 `manaenv_ex001_wholebody_policy_wire_20d_v1`。

模型特定逻辑只保留在部署边界。初始末端位姿组合、Euler 到 quaternion、轮速和归一化夹爪转换仍由 ManaEnv 统一负责。

远程原生服务可先转发到本机，再通过 `PI05_NATIVE_ADDRESS`、`PI05_NATIVE_PORT` 或 `PI05_NATIVE_ENDPOINT` 配置适配器。

[English](README.md)
