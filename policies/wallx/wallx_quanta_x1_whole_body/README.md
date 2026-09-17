# Wall-X Quanta X1 Whole Body

This directory is a model-specific deployment adapter. It converts the native WallX serialized Quanta X1 response into the public `manaenv_ex001_wholebody_policy_wire_20d_v1` wire. It does not contain simulator conversions or model weights.

The native WallX service must remain private on the inference host. Start the gateway after the native service is ready:

```bash
WALLX_NATIVE_ADDRESS=127.0.0.1 WALLX_NATIVE_PORT=8002 \
  policy-space serve policies/wallx/wallx_quanta_x1_whole_body/deploy.yaml \
  --host 0.0.0.0 --port 8001
```

The evaluator injects the host with `--server-address` / `--server-port`.

Protocol layout:

```text
follow1_pos[7] | follow2_pos[7] | velocity_decomposed_odom[3] | lift[1] | head_pos[2]
```

This is exactly 20 values. Home-pose composition, Euler-to-quaternion conversion, gripper scaling, and wheel conversion remain in ManaEnv.

[简体中文](README.zh-CN.md)
