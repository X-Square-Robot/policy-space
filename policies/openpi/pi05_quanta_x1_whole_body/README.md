# OpenPI (π0.5) Quanta X1 Whole Body

This PolicySpace deployment adapter calls the native Quanta X1 Pi0.5 inference service and packs its named output into `manaenv_ex001_wholebody_policy_wire_20d_v1`.

It is intentionally model-specific only at the deployment boundary. ManaEnv remains the single owner of init-EE composition, Euler-to-quaternion, wheel, and normalized-gripper conversion.

For a remote native service, forward it locally and configure the adapter with `PI05_NATIVE_ADDRESS`, `PI05_NATIVE_PORT`, or `PI05_NATIVE_ENDPOINT`.

[简体中文](README.zh-CN.md)
