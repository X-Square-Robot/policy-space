# Policy Integration

A policy integration has two responsibilities: convert common observations into model-native inputs, and convert model outputs into actions declared by the contract.

## 1. Create a package

The following example scaffolds a custom π0.5 adapter under the OpenPI family. It creates integration files only; it does not download weights or modify an existing policy.

```bash
policy-space init pi05_custom \
  --family openpi \
  --observation bimanual_rgb_joint@1 \
  --action bimanual_joint_position@1
```

- `pi05_custom`: new integration package name.
- `--family openpi`: groups the package under the OpenPI family.
- `--observation`: versioned observation contract consumed by the policy.
- `--action`: versioned action contract produced by the policy.

The command creates:

```text
policies/openpi/pi05_custom/
├── model.py
├── deploy.yaml
├── requirements.txt
└── README.md
```

Choose the closest working reference by policy family and robot setup:

- **OpenPI / π0.5:** [ArtiXon Arm-6A EE](../policies/openpi/pi05_artixon_arm_6a_eef/), [Quanta X1 whole body](../policies/openpi/pi05_quanta_x1_whole_body/), [ArtiXon Arm-6A joint](../policies/openpi/pi05_artixon_arm_6a_joint/), [ARX R5 EE](../policies/openpi/pi05_arx_r5_eef/), and [ARX R5 joint](../policies/openpi/pi05_arx_r5_joint/).
- **Wall-X:** [wall-oss-0.5 ArtiXon Arm-6A EE](../policies/wallx/wall_oss_05_artixon_arm_6a_eef/) and [Quanta X1 whole body](../policies/wallx/wallx_quanta_x1_whole_body/).
- **DreamZero:** [ArtiXon Arm-6A joint](../policies/dreamzero/dreamzero_artixon_arm_6a_joint/) and [DROID joint](../policies/dreamzero/dreamzero_droid_joint/).
- **Cosmos3:** [ArtiXon Arm-6A EE](../policies/cosmos3/cosmos3_artixon_arm_6a_eef/), [DROID joint](../policies/cosmos3/cosmos3_droid_joint/), and [Franka joint](../policies/cosmos3/cosmos3_franka_joint/).
- **DM0.5:** [ArtiXon Arm-6A joint](../policies/dm05/dm05_artixon_arm_6a_joint/) and [ArtiXon Arm-6A EE](../policies/dm05/dm05_artixon_arm_6a_eef/).
- **FastWAM:** [ArtiXon Arm-6A joint](../policies/fastwam/fastwam_artixon_arm_6a_joint/).
- **SmolVLA:** [Franka joint](../policies/smolvla/smolvla_franka_joint/).

## 2. Implement the episode lifecycle

```python
from policy_space import PolicyModel


class Model(PolicyModel):
    def initialize_episode(self, episode_info=None):
        pass

    def ingest_observation(self, observation):
        self.observation = observation

    def infer_actions(self):
        return self.policy.predict(self.observation)

    def finalize_episode(self, episode_info=None):
        pass
```

- `initialize_episode`: prepare state for a new episode.
- `ingest_observation`: receive the current observation and update history or internal state when needed.
- `infer_actions`: return one action or an action sequence with shape `[T, D]`.
- `finalize_episode`: clear episode-specific state.

## 3. Configure deployment

Declare the model class, model settings, listening address, session mode, and supported contracts in `deploy.yaml`:

```yaml
policy_name: pi05_custom

model:
  class: Model
  cfg:
    checkpoint_path: /path/to/checkpoint

server:
  host: 0.0.0.0
  port: 8001
  session_mode: exclusive

metadata:
  supported_schema_pairs:
    - observation_schema: bimanual_rgb_joint@1
      action_schema: bimanual_joint_position@1
      embodiment_constraints: [ex001_6r@1]
```

`host` is the address on which this Policy Space service instance listens. When ManaEnv runs on another machine, bind the service to a network interface reachable from ManaEnv; `0.0.0.0` is a common choice because it listens on all interfaces, while a specific interface address is more restrictive. Use `127.0.0.1` when the service should accept same-machine connections only. `port` is the WebSocket service port. ManaEnv connects to `ws://<policy-service-host>:<port>`, where `<policy-service-host>` is the reachable IP address or hostname of the machine running the Policy Space service for the selected policy. In the example above, `<port>` is `8001`. `0.0.0.0` is a bind address and is not a client endpoint.

See [Contracts](contracts.md) for contract fields and [Deployment](deployment.md) for session behavior.

## 4. Validate and serve

```bash
policy-space check policies/openpi/pi05_custom/deploy.yaml
policy-space serve policies/openpi/pi05_custom/deploy.yaml
```

Record model source, checkpoint preparation, environment variables, validation, and startup commands in `policies/openpi/pi05_custom/README.md`. Start from `policies/openpi/pi05_artixon_arm_6a_eef/` or the minimal `policies/templates/bimanual_ee/` implementation.

[简体中文](policy-integration.zh-CN.md)
