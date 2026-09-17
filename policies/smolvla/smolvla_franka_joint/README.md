# SmolVLA Franka Joint

The checkpoint uses 8 dimensions: seven absolute arm joint targets in radians, followed by normalized finger opening (0 closed, 1 open). Franka's `single_arm_rgb_joint@1` / `single_arm_joint_position@1` wire uses gripper closure (0 open, 1 closed).

`model.py` converts the observation gripper before checkpoint preprocessing and the action gripper after checkpoint postprocessing. Both conversions are `1 - clip(g, 0, 1)`. Packed state aliases use the same wire convention. The adapter copies inputs/outputs. Gripper conversion leaves the arm values unchanged; the optional temporal interpolation below resamples their time axis. `gripper_convention: raw` means the emitted values already match Franka; do not also enable client-side gripper normalization/inversion.

For an open-loop test through the corrected service, convert the dataset's state gripper to closure before sending it. Compare returned actions against GT with its gripper also converted to closure, or convert the response back to opening before comparing to the original GT. Direct model-only open-loop tests continue to use the original training values.

Validation from the PolicySpace root in the existing model environment:

```sh
python -m unittest tests.test_smolvla_franka_model -v
python -m policy_space.cli check policies/smolvla/smolvla_franka_joint/deploy.yaml
```

The metadata field `smolvla_gripper_adapter` must be `franka_closed01_to_model_open01_v1` after the service reloads this adapter.

## 1.5x temporal interpolation

The checkpoint still predicts 50x8 actions. `model.cfg.interpolation_multiplier: 1.5` resamples them to 75x8 in this adapter, after physical-unit postprocessing and gripper conversion. Seven joints use independent linear interpolation at native ticks `output_tick / 1.5`; the final keyframe is held to cover the last output tick. The gripper uses the previous native command so interpolation does not invent partial closure or advance an open/close transition.

The service declares `action_horizon: 75`, `action_fps: 30`, and `recommended_execute_horizon: 75`. Clients must consume those 75 rows directly, with no additional temporal interpolation. Do not run this 8-D joint vector through a pose interpolator that normalizes dimensions3:7 as a quaternion.

For dataset open-loop comparisons, compare against GT resampled by the same timing rule, or set multiplier1.0 and matching wire horizon50 for a native-grid service. This is an explicitly requested evaluation setting, not evidence that dataset FPS matches the original simulator collection-control rate.
