<div align="center">

# Policy Space

**A benchmark-neutral runtime contract for robot policy inference.**

Run each policy in its native software stack, then connect it to evaluation clients through versioned observation, action, and episode-lifecycle contracts.

[简体中文](README.zh-CN.md) · [Quick Start](#quick-start) · [Integrate a Policy](#integrate-a-policy) · [ManaEnv Evaluation](docs/manaenv-evaluation.md) · [Contracts](docs/contracts.md)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white) ![Version](https://img.shields.io/badge/version-0.3.0-6F5BD3) ![Protocol](https://img.shields.io/badge/protocol-Policy%20Space%20v2-8B5CF6) ![Transport](https://img.shields.io/badge/transport-WebSocket%20%2B%20msgpack-EF5DA8)

</div>

Robot policies often differ in runtime dependencies, observation formats, history and state requirements, action representations, and inference schedules. Policy Space provides one explicit boundary for those differences: the client keeps control of the environment, robot, task, and evaluation, while each policy service keeps its own model dependencies and inference logic.

## Runtime overview

<p align="center">
  <a href="docs/architecture.md">
    <img src="docs/assets/policy-space-runtime-overview.svg" width="100%" alt="Policy Space runtime overview">
  </a>
</p>

Policy Space standardizes the interaction boundary without forcing every policy into the same framework or process. An editable Mermaid version is available in the [architecture notes](docs/architecture.md).

## What it gives you

- **Dependency isolation.** Policies run in separate environments with their own frameworks, CUDA stacks, and checkpoints.
- **Versioned contracts.** Observation, action, and embodiment schemas make interface assumptions visible and checkable.
- **Episode-aware interaction.** `initialize_episode`, `ingest_observation`, `infer_actions`, and `finalize_episode` cover stateless policies, recurrent policies, and action-chunk inference through one lifecycle.
- **Composable adaptation.** Observation mapping and action adaptation can handle policy- and embodiment-specific representations without moving model code into the client.
- **Explicit session behavior.** Services can use exclusive sessions or serialized multiplexing according to their history and state requirements.
- **Client independence.** The protocol can serve ManaEnv evaluation as well as other compatible evaluation clients.

## Quick start

Clone the repository and install Policy Space:

```bash
git clone <repository-url> policy-space
cd policy-space
pip install -e .
policy-space check policies/templates/bimanual_ee/deploy.yaml
policy-space serve policies/templates/bimanual_ee/deploy.yaml
```

The template starts without a checkpoint or GPU, so it is useful for validating installation, transport, and lifecycle behavior.

The following scaffolding example creates a new custom π0.5 adapter package under the OpenPI family. It does not download model weights or replace the OpenPI integrations already included in this repository.

```bash
policy-space init pi05_custom \
  --family openpi \
  --observation bimanual_rgb_joint@1 \
  --action bimanual_joint_position@1

policy-space check policies/openpi/pi05_custom/deploy.yaml
```

- `pi05_custom` is the new policy package name.
- `--family openpi` groups the package under `policies/openpi/`.
- `--observation bimanual_rgb_joint@1` selects the versioned observation contract expected by the policy.
- `--action bimanual_joint_position@1` selects the versioned action contract produced by the policy.
- `policy-space check ...` validates the generated deployment configuration without starting a service.

The command creates:

```text
policies/openpi/pi05_custom/
├── deploy.yaml
├── model.py
├── requirements.txt
└── README.md
```

Continue with [Integrate a policy](#integrate-a-policy) to implement and configure the generated package. For working references, browse the [OpenPI policy catalog](policies/openpi/README.md): [`pi05_artixon_arm_6a_eef`](policies/openpi/pi05_artixon_arm_6a_eef) provides the ArtiXon Arm-6A desktop integration, while [`pi05_quanta_x1_whole_body`](policies/openpi/pi05_quanta_x1_whole_body) provides the Quanta X1 mobile whole-body integration.

## Integrate a policy

Each policy integration follows the self-contained package structure shown in the scaffold example above. Implement the lifecycle required by the policy in `model.py`:

```python
from policy_space import PolicyModel


class Model(PolicyModel):
    def initialize_episode(self, episode_info=None):
        ...

    def ingest_observation(self, observation):
        ...

    def infer_actions(self):
        ...  # Return one action or an action sequence with shape [T, D].

    def finalize_episode(self, episode_info=None):
        ...
```

Declare the model entry point, observation contract, action contract, embodiment, and session mode in `deploy.yaml`; then validate it with `policy-space check`. The [Policy Integration Guide](docs/policy-integration.md) covers the full workflow, and [`policies/templates`](policies/templates) provides working starting points.

## Policy catalog

The repository includes integrations for several policy families and robot setups:

| Policy / model | Embodiment | Integration |
|---|---|---|
| Wall-X (wall-oss-0.5) | ArtiXon Arm-6A | [`wall_oss_05_artixon_arm_6a_eef`](policies/wallx/wall_oss_05_artixon_arm_6a_eef) |
| Wall-X | Quanta X1 | [`wallx_quanta_x1_whole_body`](policies/wallx/wallx_quanta_x1_whole_body) |
| OpenPI (π0.5) | ArtiXon Arm-6A | [`pi05_artixon_arm_6a_eef`](policies/openpi/pi05_artixon_arm_6a_eef) |
| OpenPI (π0.5) | Quanta X1 | [`pi05_quanta_x1_whole_body`](policies/openpi/pi05_quanta_x1_whole_body) |
| OpenPI (π0.5) | ArtiXon Arm-6A | [`pi05_artixon_arm_6a_joint`](policies/openpi/pi05_artixon_arm_6a_joint) |
| OpenPI (π0.5) | ARX R5 | [`pi05_arx_r5_eef`](policies/openpi/pi05_arx_r5_eef) |
| OpenPI (π0.5) | ARX R5 | [`pi05_arx_r5_joint`](policies/openpi/pi05_arx_r5_joint) |
| DreamZero | ArtiXon Arm-6A | [`dreamzero_artixon_arm_6a_joint`](policies/dreamzero/dreamzero_artixon_arm_6a_joint) |
| DreamZero | DROID | [`dreamzero_droid_joint`](policies/dreamzero/dreamzero_droid_joint) |
| Cosmos3 | ArtiXon Arm-6A | [`cosmos3_artixon_arm_6a_eef`](policies/cosmos3/cosmos3_artixon_arm_6a_eef) |
| Cosmos3 | DROID | [`cosmos3_droid_joint`](policies/cosmos3/cosmos3_droid_joint) |
| Cosmos3 | Franka | [`cosmos3_franka_joint`](policies/cosmos3/cosmos3_franka_joint) |
| DM0.5 | ArtiXon Arm-6A | [`dm05_artixon_arm_6a_joint`](policies/dm05/dm05_artixon_arm_6a_joint) |
| DM0.5 | ArtiXon Arm-6A | [`dm05_artixon_arm_6a_eef`](policies/dm05/dm05_artixon_arm_6a_eef) |
| FastWAM | ArtiXon Arm-6A | [`fastwam_artixon_arm_6a_joint`](policies/fastwam/fastwam_artixon_arm_6a_joint) |
| SmolVLA | Franka | [`smolvla_franka_joint`](policies/smolvla/smolvla_franka_joint) |

Some integrations require external model source trees and checkpoints. Each integration README is the source of truth for its dependencies and launch command. See the full [policy catalog](policies/README.md).

## Connect an evaluation client

A client connects to a Policy Space service using wire protocol v2, negotiates metadata, starts an episode, sends observations, requests actions, and finalizes the episode. The client remains responsible for environment stepping, robot control, task configuration, safety, and scoring.

For ManaEnv, follow the [ManaEnv Evaluation Guide](docs/manaenv-evaluation.md). The core server remains independent of ManaEnv and can be used by any client that implements the protocol.

Clients that only need handshake validation and contract negotiation can install the lightweight `policy-space-protocol` distribution from [`packages/protocol`](packages/protocol), without installing the policy-service runtime or policy catalog.

## Documentation

| Guide | Use it for |
|---|---|
| [Policy Integration](docs/policy-integration.md) | Add a model adapter and deployment configuration |
| [Contract Reference](docs/contracts.md) | Understand observation, action, and embodiment schemas |
| [Deployment](docs/deployment.md) | Configure endpoints, sessions, concurrency, and runtime isolation |
| [ManaEnv Evaluation](docs/manaenv-evaluation.md) | Connect a policy service to closed-loop benchmark evaluation |
| [Troubleshooting](docs/troubleshooting.md) | Diagnose startup, schema, transport, and inference failures |
| [Release Guide](docs/release.md) | Pin versions and publish reproducible releases |

## Versioning

Policy Space versions its software packages, wire protocol, and contract schemas independently:

- Package version: `0.3.0`
- Wire protocol version: `2`
- Contract schemas: versioned individually as `name@version`

For reproducible deployments, pin a release tag or commit. Run the test suite before contributing:

```bash
pytest -q
```

The runtime boundary is intentionally small: policies own model-specific inference; clients own interaction and evaluation; Policy Space makes the contract between them explicit.
