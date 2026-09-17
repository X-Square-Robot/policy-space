# DM0.5 ArtiXon Arm-6A EEF

此接入通过 HTTP 调用独立 OpenDM 后端，面向 ArtiXon Arm-6A 双臂末端控制。checkpoint 由后端加载；可通过 `DM05_X2REAL_EE_CHECKPOINT_ID` 覆盖适配器记录的 ID。

客户端使用 `bimanual_rgb_ee@1` / `bimanual_ee_absolute@1` / `ex001_6r@1`；每次返回 32×14 动作，20 Hz。14D XYZ/Euler 观测转换为模型 20D row-major rot6d；输出执行逆转换，保持相对 episode 初始末端的目标语义。夹爪为 [0,1] 归一化开度。

先启动 OpenDM HTTP 后端（默认 `http://127.0.0.1:8002/v1/infer`），再在安装了本目录 requirements 的 PolicySpace 环境执行：

```bash
python -m policy_space.cli check policies/dm05/dm05_artixon_arm_6a_eef/deploy.yaml
python -m policy_space.cli serve policies/dm05/dm05_artixon_arm_6a_eef/deploy.yaml --port 8001
```

通过 `DM05_X2REAL_EE_BACKEND_URL` 覆盖后端地址，`DM05_X2REAL_EE_TIMEOUT_SECONDS` 覆盖超时，`DM05_X2REAL_EE_CHECKPOINT_ID` 覆盖适配器记录的 checkpoint ID。该 gateway 不直接占用 GPU。episode 生命周期由 PolicySpace 管理，任务指令必须非空。

[English](README.md)
