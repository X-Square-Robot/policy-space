# Wall-X (wall-oss-0.5) ArtiXon Arm-6A EEF

This service loads a Wall-X checkpoint in-process and exposes the Policy Space v2 protocol to ManaEnv and other compatible clients. It targets the ArtiXon Arm-6A desktop end-effector contract.

```text
ManaEnv / client -> Policy Space service -> Wall-X policy
```

The default `standard` backend uses the [Wall-X](https://github.com/X-Square-Robot/wall-x) serving implementation. Prepare the Wall-X inference environment and install Policy Space in that same environment; see the Wall-X README for its CUDA and model dependencies. The checkpoint is available at [wall-oss-0.5](https://huggingface.co/x-square-robot/wall-oss-0.5). The integration was verified against Wall-X commit `ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f`.

For older Wall-X runtimes, select `WALLX_BACKEND=legacy` or `model.cfg.backend: legacy` and install the dependencies required by that runtime. Non-default CFG options and Wall-X RTC are not supported by the standard backend and are rejected instead of being silently ignored.

## Serve

```bash
export WALLX_CHECKPOINT_PATH=/path/to/checkpoint
export WALLX_ROOT=/path/to/wall-x
export WALLX_BACKEND=standard
policy-space check policies/wallx/wall_oss_05_artixon_arm_6a_eef/deploy.yaml
policy-space serve policies/wallx/wall_oss_05_artixon_arm_6a_eef/deploy.yaml --port 8001
```

Wall-X returns 33 rows: the first echoes the current state and the remaining 32 are predicted actions. ManaEnv skips the echo and interpolates the predictions into 48 control steps. Observation mapping, action adaptation, and RTC scheduling remain in the client; model preprocessing and inference remain in the policy service.

[简体中文](README.zh-CN.md)
