# SmolVLA Franka Joint

模型原生输出为 50×8：七个绝对关节目标（弧度），以及夹爪归一化开度（0 关闭、1 张开）。Franka wire 使用夹爪闭合度（0 张开、1 关闭）。

适配器在观测归一化之前、动作反归一化之后，分别做 `1 - clip(g, 0, 1)`。部署声明 `gripper_convention: raw`，客户端不可再做一次夹爪缩放或反转。

当前部署按 1.5 倍进行时间插值：50×8 → 75×8，30 Hz。关节线性插值，夹爪保持上一条原生命令，避免创造额外开合过渡。客户端直接执行 75 行，不再进行第二次时间插值或四元数插值。

使用已有 SmolVLA / LeRobot 环境：

```bash
python -m policy_space.cli check policies/smolvla/smolvla_franka_joint/deploy.yaml
python -m policy_space.cli serve policies/smolvla/smolvla_franka_joint/deploy.yaml --port 8001
python -m unittest tests.test_smolvla_franka_model -v
```

在 `deploy.yaml` 中设置 `project_path` 与 `checkpoint_path`。模型依赖不加入 PolicySpace 核心包。观测和动作契约为 `single_arm_rgb_joint@1` / `single_arm_joint_position@1`，本体为 `franka@1`。

[English](README.md)
