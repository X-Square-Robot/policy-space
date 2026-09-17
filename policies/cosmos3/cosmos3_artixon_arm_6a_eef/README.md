# Cosmos3 ArtiXon Arm-6A EEF

This adapter connects to the native Cosmos3 X2Robot inference service. It converts three camera streams, proprioceptive state, and the language instruction into native model inputs and returns a bimanual end-effector action sequence.

## Configure

```bash
export COSMOS3_X2ROBOT_SERVER_ADDRESS=127.0.0.1
export COSMOS3_X2ROBOT_SERVER_PORT=8002
```

## Serve

```bash
pip install -r policies/cosmos3/cosmos3_artixon_arm_6a_eef/requirements.txt
policy-space serve policies/cosmos3/cosmos3_artixon_arm_6a_eef/deploy.yaml
```

Each request carries a complete observation. The Policy Space service therefore uses `multiplexed_serial` and serializes inference through the shared model service.

[简体中文](README.zh-CN.md)
