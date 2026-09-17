"""DreamZero DROID episode-state regression tests."""

from __future__ import annotations

from types import SimpleNamespace

from policies.dreamzero.dreamzero_droid_joint.model import Model


def test_initialize_episode_clears_wrapper_and_causal_model_state() -> None:
    head = SimpleNamespace(current_start_frame=17, language=object())
    model = Model.__new__(Model)
    model._policy = SimpleNamespace(trained_model=SimpleNamespace(action_head=head))
    model._instruction = ""

    model._initialize_episode_owned({"instruction": "put the banana in the bowl"})

    assert head.current_start_frame == 0
    assert head.language is None
    assert model._instruction == "put the banana in the bowl"
    assert model._step == 0
    assert model._frame_history == {}
    assert model._latest_obs is None
