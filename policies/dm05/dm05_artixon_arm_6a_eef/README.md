# DM0.5 ArtiXon Arm-6A EEF

This adapter calls the native OpenDM EE HTTP server on `127.0.0.1:8002`. It converts the trained 20-D `rot6d_row` representation to the public ArtiXon Arm-6A 14-D `euler_xyz` PolicySpace wire. DM0.5 predicts 50 rows internally; the registered x2real wire contract executes at most 32 rows, so this adapter exposes the first 32 rows. The Policy Space endpoint is port `8001`.

```bash
policy-space check policies/dm05/dm05_artixon_arm_6a_eef/deploy.yaml
policy-space serve policies/dm05/dm05_artixon_arm_6a_eef/deploy.yaml --host 0.0.0.0 --port 8001
```

The checkpoint and normalization statistics remain in the native OpenDM server; the adapter does not copy or rewrite model files.
