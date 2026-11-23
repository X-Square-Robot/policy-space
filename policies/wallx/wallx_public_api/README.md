# Wall-X public API / ArtiXon Arm-6A

This integration connects Policy Space v2 to the authenticated Wall-X API.
Use `deploy.yaml` for G grippers and `gripper_h/deploy.yaml` for H grippers.
The public embodiment is `quanta_x1_pro/gripper-g` or `quanta_x1_pro/gripper-h`.
Provide `WALLX_API_KEY` or `WALLX_API_KEY_FILE` outside version control.

The named-arm codec was adapted from the private comparison's
`public_runner.py` and `public_rtc_codec.py`. Position/RPY remain in the
existing bimanual EE contract. Three RGB images are encoded as JPEG95.
Policy Space uses normalized grippers; this gateway maps them to API G
`[0,1.526]` or H `[0,5.5]`. ManaEnv performs native simulation scaling.

RTC uses the existing asynchronous client scheduler, six executed rows,
four committed overlapping rows, no temporal interpolation, and an explicit
source-row offset. The `WallXApiRtc` hook preserves that offset after cropping
the committed prefix. The current upstream requires the private dense RTC
fields `s_model`, `source_phase_ns=0` and the command-anchor prefix to avoid
an observed one-row shift. Every RTC response is checked for acknowledgement
and matching committed positions, rotations and grippers. The padding at
the end of the 32-prediction API response is never executed by this schedule.

History uses the upstream's advertised `server_auto_history`, not the
private simulator's per-physical-step motion-axis selector. Each episode
gets a new connection and UUID and explicitly resets upstream history.
Actual G/H API tests verify history length 4 and RTC delay 4. This is not a
claim of numerical parity with the private simulator's client-history run.

The simulation's existing 30 Hz timestep is preserved; the API advertises
33.7 ms rows. This small rate difference is not corrected by interpolation.

Run from the Policy Space checkout:

```bash
python -m policy_space.cli check policies/wallx/wallx_public_api/deploy.yaml
python -m policy_space.cli serve policies/wallx/wallx_public_api/deploy.yaml --host 127.0.0.1 --port 18861
```

ManaEnv profile: `policy/wallx_api_rtc@v1.0.0.yaml`. For H use port 18862.
The evaluator retains the repository's scenarios, scoring and video recording.
