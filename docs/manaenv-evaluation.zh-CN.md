# 使用 ManaEnv 评测

ManaEnv 是评测客户端：它生成观测、执行机器人动作并计算评测结果。策略在独立的软件环境中通过 Policy Space 提供服务，ManaEnv 通过 `ws://` 或 `wss://` endpoint 请求动作。

评测网页是可选的管理界面，用于注册模型、选择任务、启动评测和查看结果。ManaEnv 不访问网页地址；无论是否使用网页，它连接的都是 Policy Space 策略服务 endpoint。

## 1. 启动策略服务

在策略自己的软件环境中检查并启动服务：

```bash
policy-space check policies/<policy_name>/deploy.yaml
policy-space serve policies/<policy_name>/deploy.yaml
```

首次联调可将路径换为 `policies/templates/bimanual_ee/deploy.yaml`。启动日志会给出策略服务监听的 endpoint，例如 `ws://127.0.0.1:8001`。

## 2. 选择评测方式

### 方式 A：通过网页评测

在 ManaEnv 评测网页中注册策略服务 endpoint，然后选择机器人本体、任务和 episode 数量。网页会检查服务声明的观测、动作和本体契约，并为受支持的契约生成 ManaEnv 客户端配置。

普通评测用户不需要修改 ManaEnv 仓库中的策略 YAML。

### 方式 B：直接启动 ManaEnv

网页不是运行评测的必要条件。ManaEnv 提供命令行评测入口，可以将策略服务地址直接传给评测进程：

```bash
python manaenv/scripts/x2real_eval.py \
  --task atom_drawer \
  --mode id \
  --num-episodes 1 \
  --timeout-s 30 \
  --server-address 127.0.0.1 \
  --server-port 8001 \
  --viz kit
```

正式批量运行前，可以先检查将要执行的任务和最终命令：

```bash
python manaenv/scripts/x2real_eval.py --list --mode id

python manaenv/scripts/x2real_eval.py \
  --dry-run --mode id \
  --server-address 127.0.0.1 \
  --server-port 8001
```

`--server-address` 和 `--server-port` 指向 Policy Space 策略服务，不是评测网页。评测入口使用 ManaEnv 已提供的模型无关 Policy Space 客户端配置，并在运行时替换 endpoint。

## 3. 配置由谁管理

策略服务的 `deploy.yaml` 声明模型身份、支持的观测与动作契约、本体约束以及必要的运行建议。ManaEnv 负责任务配置、本体映射、动作执行和自动评测。

对于 ManaEnv 已支持的契约，网页或命令行入口会选择已有客户端配置，用户只需提供策略服务 endpoint。只有接入新的观测结构、动作空间或机器人本体时，框架维护者才需要补充相应的契约或本体映射；这不是每次评测都要执行的用户步骤。

ManaEnv 在 episode 初始化时会与策略服务协商并检查 observation、action 和 embodiment schema。不匹配的服务不会进入正式评测。

## 4. 先做单 episode 联调

正式批量评测前，建议依次确认：

1. `policy-space check` 通过。
2. 策略服务成功加载模型并监听目标 endpoint。
3. ManaEnv 完成 metadata 握手和 episode 初始化。
4. 单 episode 中观测、动作维度和控制模式正确。
5. 再扩大到并行环境和完整任务集。

策略服务只接收评测允许的观测。用于成功判定和阶段评分的特权仿真状态保留在 ManaEnv evaluator 中，不发送给策略。

[English](manaenv-evaluation.md)
