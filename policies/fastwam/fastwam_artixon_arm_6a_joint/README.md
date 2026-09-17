# FastWAM ArtiXon Arm-6A Joint

This bundle loads the FastWAM joint-DiT checkpoint in-process and serves the Policy Space v2 lifecycle. It is independent of ManaEnv and owns all FastWAM preprocessing, normalization, and action packing.

## Contract

- Input: front, left-wrist, and right-wrist HWC RGB `uint8` images; two finite 7D joint/gripper state vectors; and the current task instruction.
- Preprocessing: front 320x256 above two 160x128 wrists, forming the trained 320x384 RobotWin mosaic; 9 video frames and 10 inference steps by default.
- Model output: exact finite `float32[32,14]` absolute joint targets.
- Wire output: exact finite `float32[32,26]`; columns 0:14 are physical actions and columns 14:26 are zero-filled record-only master-EE slots.
- Rate: 20 Hz. Grippers use normalized `[0,1]` values.

Both the adapter and generic runtime require H32, so a model change cannot silently invalidate the metadata contract.

## Configuration and launch

```bash
export FASTWAM_ROOT=/path/to/FastWAM
export FASTWAM_TRAINING_CONFIG_PATH=/path/to/training/config.yaml
export FASTWAM_CHECKPOINT_PATH=/path/to/weights/checkpoint.pt
export FASTWAM_DATASET_STATS_PATH=/path/to/dataset_stats.json
policy-space check policies/fastwam/fastwam_artixon_arm_6a_joint/deploy.yaml
policy-space serve policies/fastwam/fastwam_artixon_arm_6a_joint/deploy.yaml --port 8001
```

The verified deployment produced repeated finite `float32[32,26]` responses from the H32 checkpoint. Re-run `policy-space check` in the FastWAM model environment after changing the checkpoint, config, or statistics.

[简体中文](README.zh-CN.md)
