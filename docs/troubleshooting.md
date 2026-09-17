# Troubleshooting

## `policy-space check` fails

Identify the failing layer first:

- Model source or checkpoint not found: verify environment variables and paths in the policy README.
- Schema not found: confirm that the contract file exists and that name and version match exactly.
- Action dimension mismatch: compare `deploy.yaml`, model output, and the action contract.
- NaN or Inf output: inspect preprocessing, normalization, and postprocessing in the model adapter.

## ManaEnv cannot connect

Confirm that:

1. The service process is running and listening on the configured port.
2. The ManaEnv endpoint matches the service address.
3. A cross-machine service is not listening only on `127.0.0.1`.
4. Firewalls and proxies allow WebSocket traffic.

## Handshake rejected

Compare the ManaEnv embodiment configuration with `supported_schema_pairs`:

- observation schema;
- action schema;
- embodiment constraint;
- policy name and protocol version.

Do not disable validation to hide a mismatch; otherwise the error may appear only during robot execution.

## Clients affect each other

Use `exclusive` when the model maintains history frames, a KV cache, or other process-level state. Use `multiplexed_serial` only when the model keeps no client-specific temporal state or the state is fully isolated by episode.

## State leaks across episodes

Check that `initialize_episode` rebuilds episode history and internal state, and that `finalize_episode` releases episode-specific resources. Use `close()` for process-level resources.

## Still unresolved

Collect the policy name and commit, `deploy.yaml`, service startup logs, ManaEnv endpoint, failing lifecycle operation, episode ID, and a minimal reproduction. Do not share model weights, credentials, or sensitive paths.

[简体中文](troubleshooting.zh-CN.md)
