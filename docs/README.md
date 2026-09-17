# Policy Space Documentation

Policy Space connects evaluation clients to policy services. The client owns the environment, robot execution, and evaluation; the policy service owns model loading, model-specific input conversion, inference, and optional internal state.

## Start here

1. Run the no-GPU example from the Quick Start on the repository homepage.
2. Follow [Policy Integration](policy-integration.md) to add a model.
3. Follow [ManaEnv Evaluation](manaenv-evaluation.md) to connect the simulation client.

## Find a topic

- Runtime boundaries: [Architecture](architecture.md)
- Observation, action, or embodiment schemas: [Contracts](contracts.md)
- Model environments, ports, and multi-client behavior: [Deployment](deployment.md)
- Version pinning and wheel publication: [Release](release.md)
- Startup, connection, or inference failures: [Troubleshooting](troubleshooting.md)

## Repository layout

```text
policy_space/
├── docs/                       # user documentation
├── policies/                   # production integrations and templates
│   └── templates/              # runnable examples without weights or GPUs
├── src/policy_space/           # protocol, server, client, and shared runtime
│   └── contracts/schemas/      # versioned public contracts
└── tests/                      # protocol and compatibility tests
```

See the [policy catalog](../policies/README.md). A production integration normally contains `model.py`, `deploy.yaml`, `requirements.txt`, and `README.md`.

[简体中文](README.zh-CN.md)
