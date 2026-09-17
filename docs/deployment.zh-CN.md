# 部署说明

## 模型环境

Policy Space 核心包只提供协议和服务运行时。Torch、JAX、模型源码与权重由每个策略自己的环境管理。外部源码通过环境变量或 `model.cfg` 显式指定，例如：

```bash
export DREAMZERO_ROOT=/path/to/dreamzero
export DREAMZERO_CHECKPOINT_PATH=/path/to/checkpoint
```

策略需要的全部变量应写入该策略目录的 `README.md`。

## 会话模式

`deploy.yaml` 的 `server.session_mode` 控制多个客户端如何使用模型：

- `exclusive`：同一时间只允许一个客户端使用服务，适合在模型进程中维护历史帧、KV cache 或其他客户端专属状态的策略。
- `multiplexed_serial`：按 episode 隔离客户端会话，并对共享模型串行调用；适合每次推理都携带所需输入、模型不保留客户端专属时序状态的策略。

不确定时先使用 `exclusive`。确认模型状态可以安全隔离后，再使用 `multiplexed_serial`。

## 本地与远程连接

本机联调建议监听 `127.0.0.1`：

```bash
policy-space serve policies/<family>/<policy_name>/deploy.yaml \
  --host 127.0.0.1 --port 8001
```

ManaEnv 位于另一台机器时，应将 Policy Space 服务绑定到 ManaEnv 可以访问的网络接口。在受控网络中，可以监听全部网络接口：

```bash
policy-space serve policies/<family>/<policy_name>/deploy.yaml \
  --host 0.0.0.0 --port 8001
```

ManaEnv 随后连接 `ws://<policy-service-host>:<port>`。其中，`<policy-service-host>` 是运行当前策略对应 Policy Space 服务实例的机器可被 ManaEnv 访问的 IP 或主机名，`<port>` 是该服务的端口（本例为 `8001`）。不要将 `0.0.0.0` 作为客户端连接地址。跨不可信网络时，应在服务前配置 WSS、认证、访问控制和连接限流。

消息大小、连接数、会话数、请求频率和数据嵌套深度可在 `server` 配置中设置上限。

[English](deployment.md)
