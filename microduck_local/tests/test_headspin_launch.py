"""Foot-driven launch contract and finite-turn state machine."""
import numpy as np
import pytest

from microduck_local import behaviors as B
from microduck_local.behaviors import headstand as H


@pytest.fixture
def env():
    e = B.BehaviorEnv('headspin_launch', obs_noise=False, domain_rand=False,
                      action_delay=False, random_yaw=False, seed=123)
    e.reset(seed=123)
    yield e
    e.close()


def test_standing_zero_momentum_and_task_observations(env):
    assert B.match_behavior('撑地启动头转').id == 'headspin_launch'
    assert env.last_spawn == 'standing'
    assert np.all(env.data.qvel == 0)
    obs = env._get_obs()
    assert obs.shape == (61,)
    assert obs[55] == 0 and obs[56] == 1
    assert len(env.behavior.curriculum) == 4 and env.behavior.spotter_fn is None
    assert env.behavior.warm_start_behavior == 'headspin'
    for _ in range(20):
        obs, reward, _, _, _ = env.step(np.zeros(14))
        assert np.isfinite(obs).all() and np.isfinite(reward)
        assert obs[55] == env._ls_phase / 2
        for t in env.behavior.terms:
            if t.is_penalty:
                assert t.fn(env) <= 0
    env.reset(seed=123)
    assert env._ls_turn == env._ls_quality == env._ls_foot_impulse == 0


def test_no_foot_drive_no_launch_credit(env, monkeypatch):
    monkeypatch.setattr(H, '_launch_lz', lambda e: .03)
    monkeypatch.setattr(H, '_launch_foot_torque', lambda e: 0.)
    env.foot_contact_state = {'left': True, 'right': True}
    H._launch_update(env)
    assert H._launch_push(env) == 0
    env.foot_contact_state = {'left': False, 'right': False}
    H._launch_update(env)
    assert env._ls_quality == 0


def test_foot_push_is_bounded_and_takeoff_quality_is_retained(env, monkeypatch):
    monkeypatch.setattr(H, '_launch_lz', lambda e: .015)
    monkeypatch.setattr(H, '_launch_foot_torque', lambda e: .75)
    env.foot_contact_state = {'left': True, 'right': False}
    H._launch_update(env)
    assert H._launch_push(env) == pytest.approx(1)
    H._launch_update(env)
    assert H._launch_push(env) == 0
    env.foot_contact_state = {'left': False, 'right': False}
    H._launch_update(env)
    assert env._ls_quality == pytest.approx(1)


def test_neck_twist_cannot_complete_but_one_whole_turn_succeeds(env, monkeypatch):
    # Per the user-facing contract, a continuous turn is success even when
    # the separate 50 Hz foot-impulse proxy does not certify launch quality.
    env._ls_phase, env._ls_quality = 1, 0.
    env._ls_target_turns = 4.0
    monkeypatch.setattr(H, '_headspin_supported', lambda e: True)
    monkeypatch.setattr(H, '_headspin_world_rate', lambda e: 4.)
    def head_still(*args):
        args[-2][:] = 0.
    monkeypatch.setattr(H.mujoco, 'mj_objectVelocity', head_still)
    for _ in range(160):
        H._launch_update(env)
    assert env._ls_phase == 1
    assert env._ls_head_turn == 0 and H._launch_spin(env) == 0
    env._ls_turn = 0
    def head_rotates(*args):
        args[-2][:] = [0, 0, 4, 0, 0, 0]
    monkeypatch.setattr(H.mujoco, 'mj_objectVelocity', head_rotates)
    for _ in range(79):
        H._launch_update(env)
    assert env._ls_phase == 1
    assert min(env._ls_turn, env._ls_head_turn) >= 2 * np.pi
    assert env._ls_success

    for _ in range(236):
        H._launch_update(env)
    assert env._ls_phase == 2
    assert min(env._ls_turn, env._ls_head_turn) >= 8 * np.pi
    assert env._ls_success
    assert all(t.fn(env) == 0 for t in env.behavior.terms[:len(B.BEHAVIORS['headstand'].terms)])


def test_later_turns_pay_more_without_a_settle_reward(env):
    env._ls_phase, env._ls_quality, env._ls_spin_gain = 1, 1.0, 0.5
    env._ls_turn = env._ls_head_turn = 0.25 * 2 * np.pi
    first = H._launch_spin(env)
    env._ls_turn = env._ls_head_turn = 3.25 * 2 * np.pi
    fourth = H._launch_spin(env)
    assert fourth > first > 0
    assert 'launch_settle' not in {t.key for t in env.behavior.terms}


def test_lost_support_does_not_bank_disconnected_turns(env, monkeypatch):
    env._ls_phase = 1
    env._ls_turn = env._ls_head_turn = np.pi
    monkeypatch.setattr(H, '_headspin_supported', lambda e: False)
    env._ls_support_gap_max = .1
    for _ in range(7):
        H._launch_update(env)
    assert env._ls_phase == 2 and not env._ls_success


def test_reverse_curriculum_exposes_assist_and_preserves_spin_rewards():
    stage = B.BEHAVIORS['headspin_launch'].curriculum[0]
    e = B.BehaviorEnv('headspin_launch', obs_noise=False, domain_rand=False,
                      action_delay=False, random_yaw=False, seed=7,
                      spawn_overrides=stage.env)
    try:
        e.reset(seed=7)
        assert e.last_spawn == 'inverted'
        assert e._ls_phase == 1 and e._ls_quality == 1
        assert e._ls_target_turns == .5
        assert e.data.qvel[5] > 1.0
        # The headstand donor's terms remain active during the spin phase.
        assert any(t.fn(e) != 0 for t in e.behavior.terms[:len(B.BEHAVIORS['headstand'].terms)])
    finally:
        e.close()


def test_final_stage_is_motionless_and_unassisted():
    stage = B.BEHAVIORS['headspin_launch'].curriculum[-1]
    e = B.BehaviorEnv('headspin_launch', obs_noise=False, domain_rand=False,
                      action_delay=False, random_yaw=False, seed=9,
                      spawn_overrides=stage.env)
    try:
        e.reset(seed=9)
        assert e.last_spawn == 'standing'
        assert np.all(e.data.qvel == 0)
        assert e._ls_phase == 0 and e._ls_quality == 0
        assert e._ls_target_turns == 4
    finally:
        e.close()
