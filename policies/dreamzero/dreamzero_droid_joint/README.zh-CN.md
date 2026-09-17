# DreamZero DROID Joint

该适配器在 Policy Space 服务进程内加载 DreamZero。模型特定预处理、因果历史和推理均保留在 `model.py` 中。

## 配置

DreamZero 源码和 checkpoint 保持在仓库外：

```bash
export DREAMZERO_ROOT=/path/to/dreamzero
```

在 `deploy.yaml` 中设置 `model.cfg.checkpoint_path` 和 `model.cfg.tokenizer_path`。相对路径以 `DREAMZERO_ROOT` 为基准，也可使用绝对路径。

该适配器接收三路 RGB、七个关节位置、一个夹爪值和语言指令，返回可变长度的 `[T, 8]` 绝对关节动作序列。

## 会话与启动

DreamZero 维护因果历史和模型状态，因此使用 `session_mode: exclusive`。并行评测需在不同端口启动多个服务进程。

```bash
pip install -e /path/to/policy-space
policy-space check policies/dreamzero/dreamzero_droid_joint/deploy.yaml
policy-space serve policies/dreamzero/dreamzero_droid_joint/deploy.yaml --port 8001
```

[English](README.md)
