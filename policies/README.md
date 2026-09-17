# Policy Catalog

`policies/` is the entry point for Policy Space integrations. Production integrations are grouped by model family; each leaf directory contains the model adapter, deployment configuration, dependencies, and usage notes.

## Production policies

- [Cosmos3](cosmos3/README.md): DROID joint, Franka joint, and ArtiXon Arm-6A EE integrations.
- [DreamZero](dreamzero/README.md): DROID and ArtiXon Arm-6A joint integrations.
- [DM0.5](dm05/README.md): ArtiXon Arm-6A joint and [EE](dm05/dm05_artixon_arm_6a_eef/README.md) integrations.
- [FastWAM](fastwam/README.md): ArtiXon Arm-6A joint-policy integration.
- [OpenPI](openpi/README.md): π0.5 for ArtiXon Arm-6A EE / joint, Quanta X1 whole body, [ARX R5 EE](openpi/pi05_arx_r5_eef/README.md), and [ARX R5 joint](openpi/pi05_arx_r5_joint/README.md).
- [Wall-X](wallx/README.md): wall-oss-0.5 (ArtiXon Arm-6A EE) and Quanta X1 whole-body integrations.
- [SmolVLA](smolvla/README.md): [Franka](smolvla/smolvla_franka_joint/README.md) joint control.

## Templates

[`templates/`](templates/README.md) contains three runnable examples that require no checkpoint or GPU. Use them to validate contracts, the episode lifecycle, and client communication.

## Integration package

A typical integration contains:

```text
policies/my_family/my_policy/
├── model.py
├── deploy.yaml
├── requirements.txt
└── README.md
```

Place a new policy under its model family, or create a family directory directly under `policies/`. See [Policy Integration](../docs/policy-integration.md) for the complete workflow.

## Model and embodiment coverage

| Model | Embodiment | Integration |
|---|---|---|
| Cosmos3 | DROID / Franka / ArtiXon Arm-6A | [cosmos3](cosmos3/README.md) |
| DreamZero | DROID / ArtiXon Arm-6A | [dreamzero](dreamzero/README.md) |
| DM0.5 | ArtiXon Arm-6A joint / EE | [dm05](dm05/README.md) |
| FastWAM | ArtiXon Arm-6A joint | [fastwam](fastwam/README.md) |
| OpenPI π0.5 | ArtiXon Arm-6A / Quanta X1 / joint; ARX R5 EE / joint | [openpi](openpi/README.md) |
| Wall-X | ArtiXon Arm-6A / Quanta X1 | [wallx](wallx/README.md) |
| SmolVLA | Franka joint | [smolvla](smolvla/README.md) |

[简体中文](README.zh-CN.md)
