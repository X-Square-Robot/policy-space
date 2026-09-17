from pathlib import Path
import importlib.util
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("dm05_ee_test", Path(__file__).parents[1] / "policies/dm05/dm05_artixon_arm_6a_eef/model.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_ee_adapter_preserves_row_rot6d_pose_and_grippers():
    state = np.array([.1,.2,.3,.2,-.4,.6,.7, -.2,.4,.1,-.3,.2,-.5,.1], np.float32)
    native = M._wire_to_model(state)
    assert native.shape == (20,)
    wire = M._model_to_wire(np.tile(native, (32,1)))
    np.testing.assert_allclose(wire, np.tile(state,(32,1)), atol=1e-6)
    with pytest.raises(ValueError, match="20"):
        M._model_to_wire(np.zeros((32,14), np.float32))


def test_finalize_clears_ee_observation():
    model = M.Model.__new__(M.Model)
    model._latest_obs = {"instruction":"previous episode"}
    model.finalize_episode({"episode_id":"finished"})
    with pytest.raises(RuntimeError, match="buffered"):
        model.infer_actions()
