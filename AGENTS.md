# Policy Space deployment guide for coding agents

Help developers run their models as Policy Space services and connect them to compatible evaluation clients. Use this file as a starting map; adapt the workflow, environment manager, and implementation to the user's model and infrastructure.

## Start with the selected integration

- Find the closest model and robot setup in the [policy catalog](policies/README.md), then read that integration's README and `deploy.yaml`. Its dependencies, checkpoint preparation, and launch instructions take precedence over the generic example below.
- Use [Policy Integration](docs/policy-integration.md) for adapters, [Deployment](docs/deployment.md) for sessions and networking, [Contracts](docs/contracts.md) for interface semantics, and [Troubleshooting](docs/troubleshooting.md) for failures. Chinese equivalents are linked from each guide.
- Establish the target model/checkpoint, robot embodiment, action representation, inference environment, and client endpoint. Inspect existing configuration first; ask when missing information would change the deployment.

## Prepare and adapt

- Use the model's native inference environment and install Policy Space into the same Python environment. Model frameworks, CUDA dependencies, source trees, and weights follow the selected integration's requirements. Hosting a policy service alone does not require a simulator or the X2Real dataset.
- Reuse a compatible integration where practical, or start from `policies/templates/` or `policy-space init --help`. Keep model-specific loading, preprocessing, output conversion, and temporal state in `policies/<family>/<policy>/model.py`.
- Configure the model entry point, model settings, service address, session mode, and `metadata.supported_schema_pairs` in `deploy.yaml`. Declared capabilities should match the adapter's actual behavior and the client's supported contracts.
- Check camera order and image layout, state ordering, normalization, action dimensions, units, coordinate frames, absolute/relative conventions, gripper scaling, and action horizon. Reuse contracts with matching semantics; changes to existing semantics need a new version and matching client support.
- Use `initialize_episode`, `ingest_observation`, `infer_actions`, and `finalize_episode` for episode state. Keep history and caches isolated between episodes; reserve `close` for process cleanup. Prefer `exclusive` for stateful models; use `multiplexed_serial` when shared-model calls can safely isolate session state.
- The shared service runtime lives in `src/policy_space/`; portable protocol definitions and built-in schemas live in `packages/protocol/src/policy_space_protocol/`. Extend these when the task calls for it, keeping model-specific dependencies in the integration.

## Check and serve

After activating the intended inference environment, run from the repository root. Replace the template path with the chosen integration and apply its required environment variables or native-service setup first.

```bash
python -m pip install -e .
POLICY_DEPLOY=policies/templates/bimanual_ee/deploy.yaml
python -m policy_space.cli check "$POLICY_DEPLOY"
python -m policy_space.cli serve "$POLICY_DEPLOY" --host 127.0.0.1 --port 8001
```

- The template needs no checkpoint or GPU. For a real integration, `check` loads the model and runs inference on synthetic observations, so it may need weights, a GPU, or an upstream service. A pass verifies interface conformance, not task success.
- If several schema combinations are declared, select the observation, action, and embodiment together using the options shown by `python -m policy_space.cli check --help`.
- For a remote client, bind to a reachable interface (or `0.0.0.0`) and give the client the actual `ws://` or `wss://` endpoint, including any path. `0.0.0.0` is a bind address; `127.0.0.1` only reaches the client's own machine. Choose network access controls appropriate to the deployment; see the deployment guide.
- Verify connectivity from the client side and try a small rollout when an evaluation environment is available. For X2Real, follow the [ManaEnv Evaluation guide](docs/manaenv-evaluation.md). Keep the service running throughout evaluation.

## Validate and hand off

- Match validation to the change: check documentation links for documentation edits; run relevant existing tests for adapter, contract, or runtime edits. Broaden testing when shared behavior changes.
- Report the environment, deployment configuration, startup command, endpoint, and checks performed. If weights, hardware, credentials, or client access are unavailable, state what remains unverified and the next concrete step.
- Keep reusable setup notes with the integration. Preserve unrelated local changes and keep credentials, private deployment details, and model weights out of tracked files. Use the developer's preferred tools and deployment approach while preserving the declared interface behavior.
