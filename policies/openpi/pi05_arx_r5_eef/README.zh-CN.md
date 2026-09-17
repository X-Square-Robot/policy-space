# OpenPI (π0.5) ARX R5 EEF

该接入加载面向 ARX R5 末端契约的 JAX π0.5 checkpoint。通过 `OPENPI_CHECKPOINT_PATH` 指向 checkpoint 根目录。

## 动作、观测与训练一致性

- 本体：`arx_r5@1`；观测：`bimanual_rgb_ee@1`；动作：`bimanual_ee_absolute@1`。
- 三路 uint8 HWC RGB，相机 front / left / right 对应 faceImg / leftImg / rightImg，右腕 mask 开启。
- 每臂为 `[x,y,z,roll,pitch,yaw,gripper]`，先左臂后右臂，共 14 维。
- 位姿是相对 episode 初始末端坐标系的目标，平移米、XYZ Euler 弧度。不把它当作逐步增量或世界坐标；不做 rot6d 转换。
- 夹爪为归一化开度：0 关闭、1 张开。客户端按 ARX 本体换算原生单位；服务端不再乘源数据中的 4.5。预测夹爪仅裁剪到 [0,1]，位姿值不改变。
- 使用 checkpoint 的分位数归一化、π0.5 状态 tokenizer 和 32 维内部 padding；反归一化后截取前 14 维。
- 每次严格返回 `32 × 14`，20 Hz，不额外添加当前状态、不做时间插值或平滑。
- 指令不能为空。服务使用 exclusive 会话，episode 结束时清理观测和指令。

推理配置在 policy 包内构建，不依赖外部 ARX 启动脚本或训练数据。OpenPI 模型、tokenizer 和读取 checkpoint 的依赖仍使用原生环境。

## 环境与路径

通过 `OPENPI_ROOT`、`OPENPI_CHECKPOINT_PATH` 指向本机 OpenPI 源码和 checkpoint 根目录；通过 `POLICY_SPACE_PYTHON` 覆盖解释器。checkpoint 必须传根目录，不能传其 `params/` 子目录。推理不需要 base checkpoint 或优化器状态。已有离线 PaliGemma tokenizer 缓存须保留。

参考训练版本：JAX/JAXlib 0.5.3、Flax 0.10.2、Orbax 0.11.13、NumPy 1.26.4、Transformers 4.53.2。

## 检查与启动

在 Policy Space 项目根目录执行：

```bash
export CUDA_VISIBLE_DEVICES=<当前可用GPU编号>
export OPENPI_ROOT=/path/to/openpi
export OPENPI_CHECKPOINT_PATH=/path/to/checkpoint
bash policies/openpi/pi05_arx_r5_eef/serve.sh --check
bash policies/openpi/pi05_arx_r5_eef/serve.sh --host 0.0.0.0 --port 8001
python -m pytest -q tests/test_openpi_pi05_arx_ee_policy.py
```

`--check` 调用 `python -m policy_space.cli check`，会加载真实权重，用合成观测完成 episode 生命周期和推理。启动脚本禁用 JAX 预分配，默认内存比例 0.25；运行前按实时占用选择 GPU。

客户端选择 `bimanual_rgb_ee@1` / `bimanual_ee_absolute@1` / `arx_r5@1`，连接 `ws://127.0.0.1:8001`。本地客户端可先开 SSH 转发，再连接同一 URL：

```bash
ssh -N -L 8001:127.0.0.1:8001 my-server
```

端口冲突时用 `--port` 覆盖。接口检查和有限动作输出只证明适配可用，任务成功率仍需 rollout 评估。

[English](README.md)
