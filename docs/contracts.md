# Contract Reference

Contracts let a client and policy service confirm that they use the same observation, action, and robot-embodiment conventions before inference begins.

## Contract types

- `src/policy_space/contracts/schemas/protocol/`: messages and episode-lifecycle versions.
- `src/policy_space/contracts/schemas/observations/`: observation fields, shapes, and semantics.
- `src/policy_space/contracts/schemas/actions/`: action dimensions, layouts, and representations.
- `src/policy_space/contracts/schemas/embodiments/`: robot embodiments and constraints.

These YAML files are installed as package data for `policy_space.contracts`, so source checkouts and installed packages use the same built-in contracts.

Contract identifiers use `name@version`, for example:

```yaml
observation_schema: bimanual_rgb_joint@1
action_schema: bimanual_joint_position@1
embodiment_constraints: [ex001_6r@1]
```

## Policy declarations

A policy lists supported combinations under `metadata.supported_schema_pairs` in `deploy.yaml`. A policy may declare multiple combinations; the client selects one during `initialize_episode`.

```yaml
metadata:
  supported_schema_pairs:
    - observation_schema: bimanual_rgb_joint@1
      action_schema: bimanual_joint_position@1
      embodiment_constraints: [ex001_6r@1]
```

## Versioning rules

- Optional fields that preserve existing semantics may extend the current version.
- Changes to field meaning, action layout, or units require a new version.
- Put shared conventions in contracts instead of hard-coding separate copies in model and client code.
- After a change, run `policy-space check <deploy.yaml>` and verify it with a client.

[简体中文](contracts.zh-CN.md)
