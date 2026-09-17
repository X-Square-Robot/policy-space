from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from policies.dreamzero.dreamzero_artixon_arm_6a_joint.model import (
    ACTION_MODEL_KEYS,
    DEFAULT_CAMERA_MAP,
    DEFAULT_RELATIVE_OFFSETS,
    Model,
    _env_csv_ints_or_cfg,
    _env_flag,
    _env_int_or_cfg,
    _env_or_cfg,
)


class _FakeTorch:
    @staticmethod
    def inference_mode():
        return nullcontext()

    @staticmethod
    def is_tensor(_value):
        return False

    @staticmethod
    def as_tensor(value):
        return _FakeVideoTensor(value)


class _FakeVideoTensor:
    def __init__(self, value):
        self.value = value
        self.device = None

    def detach(self):
        return self

    def to(self, device):
        self.device = device
        return self


class _FakePolicy:
    def __init__(self, left, right, video_pred=None):
        self._left = left
        self._right = right
        self._video_pred = video_pred
        self.batches = []
        self.trained_model = SimpleNamespace(
            action_head=SimpleNamespace(current_start_frame=17, language="cached")
        )

    def lazy_joint_forward_causal(self, batch):
        self.batches.append(batch)
        return (
            SimpleNamespace(
                act={
                    ACTION_MODEL_KEYS[0]: self._left,
                    ACTION_MODEL_KEYS[1]: self._right,
                }
            ),
            self._video_pred,
        )


def _obs(value=0.0):
    return {
        "images": {
            name: np.full((4, 5, 3), index, dtype=np.uint8)
            for index, name in enumerate(DEFAULT_CAMERA_MAP)
        },
        "state": {
            "follow1_pos": np.arange(7, dtype=np.float32) + value,
            "follow2_pos": np.arange(7, dtype=np.float32) + value + 10,
        },
        "instruction": "put the spoon in the bowl",
    }


def _bare_model():
    model = Model.__new__(Model)
    model._camera_map = dict(DEFAULT_CAMERA_MAP)
    model._relative_offsets = list(DEFAULT_RELATIVE_OFFSETS)
    model._history_window = 25
    model._image_resolution = (4, 5)
    model._instruction = ""
    model._require_instruction = True
    model._action_horizon = 24
    model._output_action_dim = 26
    model._relative_actions = False
    model._inference_method = "lazy_joint_forward_causal"
    model._replan_context_mode = "causal"
    model._execute_horizon = 24
    model._save_video_pred = False
    model._video_output_dir = None
    model._video_fps = 15.0
    model._video_pred_latents = []
    model._video_episode_index = 0
    model._gripper_debounce_enabled = False
    model._gripper_filter_mode = "binary_slew"
    model._gripper_close_threshold = 0.15
    model._gripper_open_threshold = 0.35
    model._gripper_min_stable_steps = 3
    model._gripper_closed_target = 0.0
    model._gripper_open_target = 1.0
    model._gripper_max_delta_per_step = 0.06
    model._gripper_deadband = 0.02
    model._gripper_ema_alpha = 0.65
    model._gripper_max_close_delta_per_step = 0.20
    model._gripper_max_open_delta_per_step = 0.10
    model._gripper_release_threshold = 0.75
    model._gripper_release_stable_steps = 4
    model._gripper_open_state = {}
    model._gripper_accepted_target = {}
    model._gripper_pending_state = {}
    model._gripper_pending_steps = {}
    model._gripper_closed_latched = {}
    model._gripper_release_pending_steps = {}
    model._action_smoothing_enabled = False
    model._action_interpolation_multiplier = 2
    model._action_smoothing_window = 7
    model._action_smoothing_polyorder = 2
    model._action_boundary_blend_steps = 6
    model._last_arm_command = None
    model._language_key = "annotation.task"
    model._torch = _FakeTorch()
    model._Batch = lambda obs: SimpleNamespace(obs=obs)
    model._frame_history = {}
    model._request_index = 0
    model._latest_obs = None
    model._observed_since_predict = False
    model._session_id = "episode"
    return model


def test_upstream_absolute_actions_are_not_anchored_twice_and_are_padded():
    model = _bare_model()
    observation = _obs()
    model._ingest_observation_owned(observation)
    left = np.full((24, 7), 0.25, dtype=np.float32)
    right = np.full((24, 7), -0.5, dtype=np.float32)
    model._policy = _FakePolicy(left, right)

    actions = model._infer_actions_owned()

    assert actions.shape == (24, 26)
    np.testing.assert_allclose(actions[:, :7], 0.25)
    np.testing.assert_allclose(actions[:, 7:14], -0.5)
    np.testing.assert_array_equal(actions[:, 14:], 0.0)


def test_predicted_video_is_only_buffered_when_export_is_enabled():
    model = _bare_model()
    observation = _obs()
    model._ingest_observation_owned(observation)
    left = np.zeros((24, 7), dtype=np.float32)
    right = np.zeros((24, 7), dtype=np.float32)
    video_pred = np.ones((3, 4, 2, 3, 5), dtype=np.float32)
    model._policy = _FakePolicy(left, right, video_pred=video_pred)

    model._infer_actions_owned()

    assert model._video_pred_latents == []

    model._save_video_pred = True
    model._observed_since_predict = True
    model._infer_actions_owned()

    assert len(model._video_pred_latents) == 1
    assert model._video_pred_latents[0].device == "cpu"


def test_prediction_video_env_flag_overrides_yaml_default(monkeypatch):
    monkeypatch.setenv("DREAMZERO_SAVE_VIDEO_PRED", "1")
    assert _env_flag("DREAMZERO_SAVE_VIDEO_PRED", False) is True

    monkeypatch.setenv("DREAMZERO_SAVE_VIDEO_PRED", "off")
    assert _env_flag("DREAMZERO_SAVE_VIDEO_PRED", True) is False


def test_launcher_path_override_takes_precedence_over_checked_in_default(monkeypatch):
    cfg = {"checkpoint_path": "/checkpoints/default-6000"}
    monkeypatch.setenv("DREAMZERO_CHECKPOINT_PATH", "/checkpoints/selected-10000")

    resolved = _env_or_cfg(
        cfg,
        env_name="DREAMZERO_CHECKPOINT_PATH",
        cfg_key="checkpoint_path",
    )

    assert resolved == "/checkpoints/selected-10000"


def test_execute_horizon_env_override_is_parsed_as_int(monkeypatch):
    monkeypatch.setenv("DREAMZERO_EXECUTE_HORIZON", "12")

    resolved = _env_int_or_cfg(
        {"execute_horizon": 24},
        env_name="DREAMZERO_EXECUTE_HORIZON",
        cfg_key="execute_horizon",
    )

    assert resolved == 12


def test_runtime_metadata_reports_actual_execute_horizon() -> None:
    model = Model.__new__(Model)
    model._execute_horizon = 8

    assert model.runtime_metadata() == {"recommended_execute_horizon": 8}


def test_action_boundary_blend_steps_env_override_takes_precedence(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("DREAMZERO_ACTION_BOUNDARY_BLEND_STEPS", "3")
    monkeypatch.setattr(Model, "_run", lambda self, fn: None)

    model = Model(
        {
            "action_smoothing": {"boundary_blend_steps": 6},
            "checkpoint_path": "/tmp/checkpoint",
            "dreamzero_root": str(tmp_path),
        }
    )

    assert model._action_boundary_blend_steps == 3
    model._executor.shutdown(wait=False)


def test_replan_context_mode_env_override_takes_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("DREAMZERO_REPLAN_CONTEXT_MODE", "single")
    monkeypatch.setattr(Model, "_run", lambda self, fn: None)

    model = Model(
        {
            "replan_context_mode": "causal",
            "checkpoint_path": "/tmp/checkpoint",
            "dreamzero_root": str(tmp_path),
        }
    )

    assert model._replan_context_mode == "single"
    model._executor.shutdown(wait=False)


def test_history_offsets_env_override_is_parsed_as_csv_ints(monkeypatch):
    monkeypatch.setenv("DREAMZERO_RELATIVE_OFFSETS", "-24, -18, -12, -6, 0")

    resolved = _env_csv_ints_or_cfg(
        {"relative_offsets": DEFAULT_RELATIVE_OFFSETS},
        env_name="DREAMZERO_RELATIVE_OFFSETS",
        cfg_key="relative_offsets",
    )

    assert resolved == [-24, -18, -12, -6, 0]


def test_history_offsets_env_override_rejects_empty_values(monkeypatch):
    monkeypatch.setenv("DREAMZERO_RELATIVE_OFFSETS", ", ,")

    with np.testing.assert_raises_regex(ValueError, "at least one integer"):
        _env_csv_ints_or_cfg(
            {"relative_offsets": DEFAULT_RELATIVE_OFFSETS},
            env_name="DREAMZERO_RELATIVE_OFFSETS",
            cfg_key="relative_offsets",
        )


def test_prediction_video_flush_decodes_each_view_and_writes_metadata(
    tmp_path, monkeypatch
):
    class _FakeVAE:
        @staticmethod
        def decode(_latents, **_kwargs):
            return torch.zeros((3, 3, 4, 2, 2), dtype=torch.float32)

    model = _bare_model()
    model._torch = torch
    model._device = "cpu"
    model._save_video_pred = True
    model._video_output_dir = tmp_path
    model._video_pred_latents = [torch.zeros((3, 4, 2, 1, 1), dtype=torch.float32)]
    model._policy = _FakePolicy(np.zeros((24, 7)), np.zeros((24, 7)))
    head = model._policy.trained_model.action_head
    head.vae = _FakeVAE()
    head.tiled = True
    head.tile_size_height = 2
    head.tile_size_width = 2
    head.tile_stride_height = 1
    head.tile_stride_width = 1
    written = []
    monkeypatch.setattr(
        "imageio.v2.mimsave", lambda path, _frames, **_kwargs: written.append(path.name)
    )

    model._flush_prediction_video_owned()

    assert written == ["pred_face.mp4", "pred_left_wrist.mp4", "pred_right_wrist.mp4"]
    episode_dirs = list(tmp_path.iterdir())
    assert len(episode_dirs) == 1
    assert (episode_dirs[0] / "metadata.json").is_file()
    assert model._video_pred_latents == []


def test_raw_relative_backend_can_opt_in_to_joint_state_anchoring():
    model = _bare_model()
    model._relative_actions = True
    observation = _obs()
    model._ingest_observation_owned(observation)
    left = np.full((24, 7), 0.25, dtype=np.float32)
    right = np.full((24, 7), -0.5, dtype=np.float32)
    model._policy = _FakePolicy(left, right)

    actions = model._infer_actions_owned()

    np.testing.assert_allclose(
        actions[:, :7],
        np.broadcast_to(observation["state"]["follow1_pos"] + 0.25, (24, 7)),
    )
    np.testing.assert_allclose(
        actions[:, 7:14],
        np.broadcast_to(observation["state"]["follow2_pos"] - 0.5, (24, 7)),
    )


def test_second_chunk_uses_nine_stride_three_history_frames():
    model = _bare_model()
    for frame_index in range(25):
        observation = _obs()
        for camera in observation["images"]:
            observation["images"][camera].fill(frame_index)
        model._append_frames(observation)
    model._request_index = 1

    request = model._build_request(_obs())

    expected = np.arange(0, 25, 3, dtype=np.uint8)
    assert model._instruction == "put the spoon in the bowl"
    for model_key in DEFAULT_CAMERA_MAP.values():
        assert request[model_key].shape == (9, 4, 5, 3)
        np.testing.assert_array_equal(request[model_key][:, 0, 0, 0], expected)


def test_single_replan_resets_causal_cursor_and_uses_latest_real_frame():
    model = _bare_model()
    model._replan_context_mode = "single"
    model._request_index = 3
    policy = _FakePolicy(np.zeros((24, 7)), np.zeros((24, 7)))
    model._policy = policy
    for frame_index in range(25):
        observation = _obs()
        for camera in observation["images"]:
            observation["images"][camera].fill(frame_index)
        model._ingest_observation_owned(observation)

    model._infer_actions_owned()

    assert policy.trained_model.action_head.current_start_frame == 0
    assert policy.trained_model.action_head.language == "cached"
    for model_key in DEFAULT_CAMERA_MAP.values():
        frames = policy.batches[-1].obs[model_key]
        assert frames.shape == (1, 4, 5, 3)
        np.testing.assert_array_equal(frames[0], 24)


def test_gripper_debounce_suppresses_short_pulse_and_accepts_stable_transition():
    model = _bare_model()
    model._execute_horizon = 6
    model._gripper_debounce_enabled = True
    current = np.zeros(14, dtype=np.float32)
    actions = np.zeros((24, 14), dtype=np.float32)
    actions[:2, 6] = 1.0
    actions[:3, 13] = 1.0

    filtered = model._debounce_grippers(actions, current)

    np.testing.assert_array_equal(filtered[:6, 6], 0.0)
    np.testing.assert_array_equal(filtered[:2, 13], 0.0)
    np.testing.assert_allclose(filtered[2, 13], 0.06)
    np.testing.assert_array_equal(filtered[6:], actions[6:])


def test_gripper_debounce_preserves_transition_state_across_chunks():
    model = _bare_model()
    model._execute_horizon = 4
    model._gripper_debounce_enabled = True
    current = np.zeros(14, dtype=np.float32)

    # The open intent reaches its three-row threshold on the final executed
    # row. The physical joint is still closed when the next refill arrives.
    first = np.zeros((24, 14), dtype=np.float32)
    first[1:4, 6] = 1.0
    filtered_first = model._debounce_grippers(first, current)
    np.testing.assert_allclose(filtered_first[:4, 6], [0.0, 0.0, 0.0, 0.06])

    # Hold the accepted open target while the opposite intent is pending. If
    # this were reinitialized from ``current``, the output would immediately
    # fall to zero and reproduce the observed one-frame gripper pulse.
    second = np.zeros((24, 14), dtype=np.float32)
    filtered_second = model._debounce_grippers(second, current)
    np.testing.assert_allclose(filtered_second[:4, 6], [0.06, 0.06, 0.0, 0.0])


def test_gripper_debounce_counts_pending_intent_across_chunks():
    model = _bare_model()
    model._execute_horizon = 2
    model._gripper_debounce_enabled = True
    current = np.zeros(14, dtype=np.float32)
    opening = np.ones((24, 14), dtype=np.float32)

    filtered_first = model._debounce_grippers(opening, current)
    np.testing.assert_array_equal(filtered_first[:2, 6], 0.0)

    filtered_second = model._debounce_grippers(opening, current)
    np.testing.assert_allclose(filtered_second[:2, 6], [0.06, 0.12])


def test_gripper_debounce_holds_canonical_target_in_same_state_and_deadband():
    model = _bare_model()
    model._execute_horizon = 6
    model._gripper_debounce_enabled = True
    current = np.zeros(14, dtype=np.float32)
    current[13] = 1.0
    actions = np.zeros((24, 14), dtype=np.float32)
    actions[:6, 6] = [0.02, 0.12, 0.20, 0.14, 0.08, 0.22]
    actions[:6, 13] = [0.98, 0.80, 0.34, 0.75, 0.99, 0.25]

    filtered = model._debounce_grippers(actions, current)

    np.testing.assert_array_equal(filtered[:6, 6], 0.0)
    np.testing.assert_array_equal(filtered[:6, 13], 1.0)


def test_gripper_debounce_rate_limits_full_stroke_transition():
    model = _bare_model()
    model._execute_horizon = 24
    model._gripper_debounce_enabled = True
    current = np.zeros(14, dtype=np.float32)
    opening = np.ones((24, 14), dtype=np.float32)

    filtered = model._debounce_grippers(opening, current)

    commanded = filtered[:24, 6]
    assert commanded[0] == 0.0
    assert commanded[1] == 0.0
    assert commanded[-1] == 1.0
    assert np.abs(np.diff(commanded)).max() <= 0.060001


def test_continuous_gripper_filter_preserves_graded_close_without_confirmation_delay():
    model = _bare_model()
    model._execute_horizon = 8
    model._gripper_debounce_enabled = True
    model._gripper_filter_mode = "continuous_slew"
    current = np.ones(14, dtype=np.float32)
    actions = np.ones((24, 14), dtype=np.float32)
    actions[:8, 6] = 0.25

    filtered = model._debounce_grippers(actions, current)

    commanded = filtered[:8, 6]
    assert commanded[0] < 1.0
    assert np.all(np.diff(commanded) <= 0)
    assert commanded[-1] == 0.25
    assert np.abs(np.diff(np.r_[1.0, commanded])).max() <= 0.200001


def test_continuous_gripper_filter_holds_small_noise_and_persists_across_chunks():
    model = _bare_model()
    model._execute_horizon = 3
    model._gripper_debounce_enabled = True
    model._gripper_filter_mode = "continuous_slew"
    current = np.zeros(14, dtype=np.float32)
    first = np.zeros((24, 14), dtype=np.float32)
    first[:3, 6] = [0.50, 0.50, 0.50]

    filtered_first = model._debounce_grippers(first, current)
    second = np.zeros((24, 14), dtype=np.float32)
    second[:3, 6] = [0.31, 0.30, 0.29]
    filtered_second = model._debounce_grippers(second, current)

    np.testing.assert_allclose(filtered_first[:3, 6], [0.10, 0.20, 0.30])
    np.testing.assert_allclose(filtered_second[:3, 6], 0.30)


def test_continuous_gripper_filter_closes_faster_than_it_opens():
    closing = _bare_model()
    closing._execute_horizon = 8
    closing._gripper_debounce_enabled = True
    closing._gripper_filter_mode = "continuous_slew"
    close_actions = np.zeros((24, 14), dtype=np.float32)
    close_current = np.ones(14, dtype=np.float32)

    close_filtered = closing._debounce_grippers(close_actions, close_current)

    opening = _bare_model()
    opening._execute_horizon = 12
    opening._gripper_debounce_enabled = True
    opening._gripper_filter_mode = "continuous_slew"
    open_actions = np.ones((24, 14), dtype=np.float32)
    open_current = np.zeros(14, dtype=np.float32)
    open_filtered = opening._debounce_grippers(open_actions, open_current)

    np.testing.assert_allclose(
        close_filtered[:5, 6], [0.8, 0.6, 0.4, 0.2, 0.0], atol=1e-6
    )
    np.testing.assert_allclose(
        open_filtered[:5, 6], [0.1, 0.2, 0.3, 0.4, 0.5], atol=1e-6
    )


def test_continuous_hysteresis_latches_grasp_against_partial_reopen():
    model = _bare_model()
    model._execute_horizon = 8
    model._gripper_debounce_enabled = True
    model._gripper_filter_mode = "continuous_hysteresis"
    current = np.ones(14, dtype=np.float32)
    closing = np.zeros((24, 14), dtype=np.float32)

    closed = model._debounce_grippers(closing, current)
    partial_reopen = np.full((24, 14), 0.60, dtype=np.float32)
    held = model._debounce_grippers(partial_reopen, np.zeros(14, dtype=np.float32))

    np.testing.assert_allclose(closed[:5, 6], [0.8, 0.6, 0.4, 0.2, 0.0], atol=1e-6)
    np.testing.assert_array_equal(held[:8, 6], 0.0)


def test_continuous_hysteresis_requires_sustained_explicit_release_across_chunks():
    model = _bare_model()
    model._execute_horizon = 2
    model._gripper_debounce_enabled = True
    model._gripper_filter_mode = "continuous_hysteresis"
    model._gripper_closed_latched = {6: True, 13: True}
    current = np.zeros(14, dtype=np.float32)
    opening = np.ones((24, 14), dtype=np.float32)

    first = model._debounce_grippers(opening, current)
    second = model._debounce_grippers(opening, current)
    third = model._debounce_grippers(opening, current)

    np.testing.assert_array_equal(first[:2, 6], 0.0)
    np.testing.assert_allclose(second[:2, 6], [0.0, 0.1], atol=1e-6)
    np.testing.assert_allclose(third[:2, 6], [0.2, 0.3], atol=1e-6)


def test_arm_smoothing_reduces_curvature_and_connects_consecutive_chunks():
    model = _bare_model()
    model._action_smoothing_enabled = True
    current = np.zeros(14, dtype=np.float32)
    first = np.zeros((24, 14), dtype=np.float32)
    first[:, 0] = np.where(np.arange(24) % 2, 0.5, -0.5)
    first[:, 7] = np.linspace(0.0, 1.0, 24)
    first[:, 6] = np.linspace(0.0, 1.0, 24)

    filtered_first = model._smooth_arm_actions(first, current)
    first_last = filtered_first[23, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]].copy()
    second = first.copy()
    second[:, 0] += 1.0
    filtered_second = model._smooth_arm_actions(second, current)

    raw_curvature = np.abs(np.diff(first[:, 0], n=2)).mean()
    filtered_curvature = np.abs(np.diff(filtered_first[:, 0], n=2)).mean()
    assert filtered_curvature < raw_curvature
    np.testing.assert_allclose(filtered_first[:, 6], first[:, 6])
    np.testing.assert_allclose(
        filtered_second[0, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]],
        first_last,
    )


def test_reset_rearms_causal_action_head():
    model = _bare_model()
    model._policy = _FakePolicy(np.zeros((24, 7)), np.zeros((24, 7)))
    model._latest_obs = _obs()
    model._frame_history = {"video.face": object()}
    model._request_index = 3
    model._gripper_open_state = {6: True}
    model._gripper_accepted_target = {6: 1.0}
    model._gripper_pending_state = {6: False}
    model._gripper_pending_steps = {6: 2}
    model._last_arm_command = np.ones(12, dtype=np.float32)

    model._initialize_episode_owned({"instruction": "new instruction"})

    head = model._policy.trained_model.action_head
    assert head.current_start_frame == 0
    assert head.language is None
    assert model._request_index == 0
    assert model._frame_history == {}
    assert model._instruction == "new instruction"
    assert model._gripper_open_state == {}
    assert model._gripper_accepted_target == {}
    assert model._gripper_pending_state == {}
    assert model._gripper_pending_steps == {}
    assert model._last_arm_command is None
