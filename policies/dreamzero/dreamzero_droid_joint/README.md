# DreamZero DROID Joint

This adapter loads DreamZero in the Policy Space service process. DreamZero-specific preprocessing, causal history, and inference stay inside `model.py`.

## Configure

Keep the DreamZero source checkout and checkpoint outside this repository:

```bash
export DREAMZERO_ROOT=/path/to/dreamzero
```

Set `model.cfg.checkpoint_path` and `model.cfg.tokenizer_path` in `deploy.yaml`. Relative paths are resolved from `DREAMZERO_ROOT`; absolute paths are also accepted.

The adapter expects three RGB cameras, seven joint positions, one gripper value, and a language instruction. It returns a variable-length `[T, 8]` absolute-joint action sequence.

## Session behavior

DreamZero maintains causal history and model state across requests. Its deployment therefore uses `session_mode: exclusive`: one service process owns one active client episode at a time. Run additional service processes on separate ports for parallel evaluation.

## Check and serve

Install Policy Space in the DreamZero environment, then run:

```bash
pip install -e /path/to/policy-space
policy-space check policies/dreamzero/dreamzero_droid_joint/deploy.yaml
policy-space serve policies/dreamzero/dreamzero_droid_joint/deploy.yaml --port 8001
```

`initialize_episode` clears old temporal state, `ingest_observation` adds real observations to the history, `infer_actions` performs prediction, and `finalize_episode` releases episode-specific state.

[简体中文](README.zh-CN.md)
