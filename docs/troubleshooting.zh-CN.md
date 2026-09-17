# 排障指南

## `policy-space check` 失败

先看错误属于哪一层：

- 找不到模型源码或权重：检查策略 README 中的环境变量和路径。
- 找不到 schema：确认契约文件存在，名称和版本完全一致。
- 动作维度不匹配：核对 `deploy.yaml`、模型输出和 action contract。
- 输出包含 NaN 或 Inf：在模型适配层检查预处理、归一化和后处理。

## ManaEnv 无法连接

确认以下内容：

1. 服务进程仍在运行，并监听配置中的端口。
2. ManaEnv 的 `endpoint` 与服务地址一致。
3. 跨机器连接时，服务没有只监听 `127.0.0.1`。
4. 防火墙和代理没有拦截 WebSocket。

## 握手被拒绝

核对 ManaEnv 本体配置与服务端 `supported_schema_pairs`：

- observation schema；
- action schema；
- embodiment constraint；
- policy name 和协议版本。

不要通过关闭校验来掩盖不匹配，否则错误可能在机器人执行阶段才出现。

## 多客户端互相影响

如果模型维护历史帧、KV cache 或其他进程级状态，将 `session_mode` 设为 `exclusive`。只有在模型不保留客户端专属时序状态，或状态已按 episode 完整隔离时，才使用 `multiplexed_serial`。

## episode 之间状态残留

检查 `initialize_episode` 是否重建本次 episode 的历史和内部状态，以及 `finalize_episode` 是否释放 episode 专属资源。进程级资源由 `close()` 释放。

## 仍无法定位

保留以下信息再排查：策略名称与提交版本、`deploy.yaml`、服务启动日志、ManaEnv endpoint、失败的生命周期操作、episode id，以及最小可复现任务。不要提交模型权重、访问凭据或敏感路径。

[English](troubleshooting.md)
