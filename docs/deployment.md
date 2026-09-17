# Deployment

## Model environments

The Policy Space core package provides the protocol and service runtime. Torch, JAX, model source code, and checkpoints remain in each policy's native environment. External source trees are specified through environment variables or `model.cfg`, for example:

```bash
export DREAMZERO_ROOT=/path/to/dreamzero
export DREAMZERO_CHECKPOINT_PATH=/path/to/checkpoint
```

Document every required variable in the integration's `README.md`.

## Session modes

`server.session_mode` in `deploy.yaml` controls how clients share a model service:

- `exclusive`: allows one client at a time. Use it when the process maintains history frames, a KV cache, or other client-specific state.
- `multiplexed_serial`: isolates sessions by episode and serializes calls into a shared model. Use it when each request carries the required input and the model keeps no client-specific temporal state.

Use `exclusive` by default. Switch to `multiplexed_serial` only after verifying that model state is safely isolated.

## Local and remote endpoints

For local development, listen on `127.0.0.1`:

```bash
policy-space serve policies/<family>/<policy_name>/deploy.yaml \
  --host 127.0.0.1 --port 8001
```

When ManaEnv runs on another machine, bind the Policy Space service to an interface reachable from ManaEnv. Listening on all interfaces is convenient in a controlled network:

```bash
policy-space serve policies/<family>/<policy_name>/deploy.yaml \
  --host 0.0.0.0 --port 8001
```

ManaEnv then connects to `ws://<policy-service-host>:<port>`. Here, `<policy-service-host>` is the reachable IP address or hostname of the machine running the Policy Space service for the selected policy, and `<port>` is the service port (`8001` in this example). Do not use `0.0.0.0` as the client endpoint. Across untrusted networks, place WSS, authentication, access control, and connection rate limits in front of the service.

The `server` configuration can also limit message size, connections, sessions, request rate, and data nesting depth.

[简体中文](deployment.zh-CN.md)
