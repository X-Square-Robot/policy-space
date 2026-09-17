# Evaluate with ManaEnv

ManaEnv is the evaluation client: it produces observations, executes robot actions, and computes evaluation results. A policy runs in its own software environment as a Policy Space service, and ManaEnv requests actions from its `ws://` or `wss://` endpoint.

The evaluation website is an optional management interface for registering models, selecting tasks, starting evaluations, and viewing results. ManaEnv does not connect to the website URL; with or without the website, it connects directly to the Policy Space service endpoint.

## 1. Start the policy service

In the policy's native software environment:

```bash
policy-space check policies/<family>/<policy_name>/deploy.yaml
policy-space serve policies/<family>/<policy_name>/deploy.yaml
```

For an initial smoke test, use `policies/templates/bimanual_ee/deploy.yaml`. The startup log reports the listening endpoint, such as `ws://127.0.0.1:8001`.

## 2. Choose an evaluation entry point

### Option A: evaluation website

Register the policy-service endpoint in the ManaEnv evaluation website, then select the robot embodiment, tasks, and episode count. The website checks the declared observation, action, and embodiment contracts and selects a supported ManaEnv client configuration.

Evaluation users do not need to edit policy YAML files in the ManaEnv repository.

### Option B: ManaEnv command line

The website is not required. Pass the policy-service address directly to ManaEnv:

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

Before a batch run, inspect the selected tasks and generated command:

```bash
python manaenv/scripts/x2real_eval.py --list --mode id

python manaenv/scripts/x2real_eval.py \
  --dry-run --mode id \
  --server-address 127.0.0.1 \
  --server-port 8001
```

`--server-address` and `--server-port` identify the Policy Space endpoint, not the evaluation website. ManaEnv uses its model-independent Policy Space client configuration and replaces the endpoint at runtime.

## 3. Configuration ownership

The service `deploy.yaml` declares policy identity, supported observation and action contracts, embodiment constraints, and runtime recommendations. ManaEnv owns task configuration, embodiment mapping, action execution, and automatic evaluation.

For contracts already supported by ManaEnv, the website or CLI selects an existing client configuration; users only provide the service endpoint. Framework maintainers add a contract or embodiment mapping only when introducing a new observation structure, action space, or robot embodiment.

During episode initialization, ManaEnv negotiates and validates the observation, action, and embodiment schemas. An incompatible service does not enter evaluation.

## 4. Validate one episode first

Before a batch evaluation, confirm that:

1. `policy-space check` passes.
2. The policy service loads the model and listens on the expected endpoint.
3. ManaEnv completes metadata negotiation and episode initialization.
4. Observation fields, action dimensions, and control mode are correct for one episode.
5. Only then expand to parallel environments and the full task set.

The policy receives only evaluation-approved observations. Privileged simulator state used for success checks and stage scoring remains in the ManaEnv evaluator.

[简体中文](manaenv-evaluation.zh-CN.md)
