# DreamZero ArtiXon Arm-6A Joint

This integration loads the DreamZero bimanual joint policy inside the Policy Space service. `model.py` owns model preprocessing, causal visual history, and internal state; ManaEnv continuously supplies real observations and executes the returned action sequence.

## Environment

```bash
export DREAMZERO_ROOT=/path/to/dreamzero
export DREAMZERO_CHECKPOINT_PATH=/path/to/checkpoint
```

Tokenizer, device, history window, and execution defaults are declared in `deploy.yaml` and can be overridden with the corresponding `DREAMZERO_*` environment variables.

## Session mode

DreamZero maintains causal history and model state, so this service uses `session_mode: exclusive`. Run multiple service processes on different ports for parallel evaluation.

## Validate and serve

```bash
pip install -e /path/to/policy-space
policy-space check policies/dreamzero/dreamzero_artixon_arm_6a_joint/deploy.yaml
policy-space serve policies/dreamzero/dreamzero_artixon_arm_6a_joint/deploy.yaml --port 8001
```

`initialize_episode` clears old history, `ingest_observation` appends real observations, `infer_actions` predicts future actions, and `finalize_episode` releases episode state.

[简体中文](README.zh-CN.md)
