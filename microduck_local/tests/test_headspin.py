"""Headspin: observable world-axis rotation, support gates, and donor selection."""
import json

import numpy as np
import pytest

from microduck_local import behaviors as B
from microduck_local import viz_server as V


def test_keywords_and_recipe():
    for text in ('headspin', 'do a head spin', '头顶接地旋转'):
        assert B.match_behavior(text).id == 'headspin'
    b = B.BEHAVIORS['headspin']
    assert b.warm_start_behavior == 'headstand'
    assert not B.is_symmetric(b)
    assert len(b.curriculum) == 3
    assert sum(s.steps for s in b.curriculum) == b.default_steps
    assert b.spotter_fn is None


def test_world_up_rate_inverts_body_z_and_matches_physics():
    env = B.BehaviorEnv('headspin', obs_noise=False, domain_rand=False,
                        action_delay=False, random_yaw=False, seed=0)
    try:
        for quat in ([1, 0, 0, 0], [0, 1, 0, 0], [2**-.5, 0, 2**-.5, 0]):
            env.reset(seed=0)
            env.data.qpos[3:7] = quat
            env.data.qvel[:] = 0
            env.data.qvel[3:6] = [.2, -.3, .6]
            B.mujoco.mj_forward(env.model, env.data)
            velocity = np.zeros(6)
            B.mujoco.mj_objectVelocity(env.model, env.data, B.mujoco.mjtObj.mjOBJ_BODY,
                                      env.trunk_body_id, velocity, 0)
            assert B._headspin_world_rate(env) == pytest.approx(velocity[2], abs=1e-6)
    finally:
        env.close()


def test_signed_reward_and_support_gate(monkeypatch):
    # Run the real env for geometry/contact gates; substitute only the speed
    # and the final gate to isolate the reward's signed, bounded arithmetic.
    from microduck_local.behaviors import headstand as H
    env = B.BehaviorEnv('headspin', standing_spawns=True, obs_noise=False,
                        domain_rand=False, action_delay=False, seed=0)
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (61,)
        assert obs[50] == pytest.approx(0.0)
        assert not H._headspin_supported(env)
        assert H._headspin_turn(env) == 0
        monkeypatch.setattr(H, '_headspin_supported', lambda e: True)
        for rate, expected in [(0, 0), (.3, .5), (-.3, -.5), (6, 1), (-6, -1)]:
            monkeypatch.setattr(H, '_headspin_world_rate', lambda e, r=rate: r)
            assert H._headspin_turn(env) == pytest.approx(expected)
            assert -1 <= H._headspin_overspeed(env) <= 0
        env._headspin_net = 99
        env.reset(seed=1)
        assert env._headspin_net == 0
        assert env._headspin_best == 0
    finally:
        env.close()


def test_donor_requires_completed_final_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(V, 'RUNS_DIR', tmp_path)
    b = B.BEHAVIORS['headspin']
    with pytest.raises(ValueError, match='completed headstand'):
        V.resolve_prerequisite_init(b)
    for name, done in [('teach-headstand-old-s1', True),
                       ('teach-headstand-active-s5', False),
                       ('teach-headstand-ready-s5', True)]:
        d = tmp_path / name
        d.mkdir()
        for f in ('model.zip', 'vecnormalize.pkl', 'policy.onnx'):
            (d / f).touch()
        (d / 'behavior.json').write_text(json.dumps({'behavior': 'headstand'}))
        (d / 'progress.jsonl').write_text(json.dumps({'done': done})+'\n')
    assert V.resolve_prerequisite_init(b).name == 'teach-headstand-ready-s5'


def test_browser_job_can_inherit_headstand(tmp_path, monkeypatch):
    # Exercise the actual /teach handler, including library hot reload, without
    # starting processes. Reuse the lab's fake-process fixture via a local fake.
    import asyncio
    from types import SimpleNamespace
    monkeypatch.setattr(B, 'reload_library', lambda: None)
    monkeypatch.setattr(V, 'RUNS_DIR', tmp_path)
    monkeypatch.setattr(V, 'teach_weights_path', lambda: tmp_path / 'weights.json')
    monkeypatch.setenv('LAB_STATE_PATH', str(tmp_path / 'lab-state.json'))
    d = tmp_path / 'teach-headstand-ready-s5'
    d.mkdir()
    for f in ('model.zip', 'vecnormalize.pkl', 'policy.onnx'):
        (d / f).touch()
    (d / 'behavior.json').write_text(json.dumps({'behavior': 'headstand'}))
    (d / 'progress.jsonl').write_text('{"done":true}\n')
    launches = []
    def launch(self, init_from):
        launches.append(init_from)
        return SimpleNamespace(pid=4194304, poll=lambda: None)
    monkeypatch.setattr(V.TrainingJob, '_launch', launch)
    app = V.make_app([])
    handler = next(r.endpoint for r in app.routes if getattr(r, 'path', '') == '/teach'
                   and 'POST' in getattr(r, 'methods', set()))
    result = asyncio.run(handler(V.TeachReq(text='headspin', steps=300000)))
    assert result.get('matched'), result
    assert launches == [d]


def test_visible_spin_prefers_target_over_still_or_fast(monkeypatch):
    from microduck_local.behaviors import headstand as H
    env = B.BehaviorEnv('headspin', standing_spawns=True, obs_noise=False,
                        domain_rand=False, action_delay=False, seed=0)
    try:
        env.reset(seed=0)
        assert H._headspin_visible(env) == 0
        monkeypatch.setattr(H, '_headspin_supported', lambda e: True)
        scores = {}
        for rate in (-1.2, 0, .3, .6, 1.2, 3.0, 10.0):
            monkeypatch.setattr(H, '_headspin_world_rate', lambda e, r=rate: r)
            scores[rate] = H._headspin_visible(env)
            assert 0 <= scores[rate] <= 1
        assert scores[-1.2] == scores[0] == scores[10.0] == 0
        assert scores[1.2] == pytest.approx(1)
        assert scores[.3] < scores[.6] < scores[1.2]
        assert scores[3.0] < scores[.6]
        monkeypatch.setattr(H, '_headspin_world_rate', lambda e: 1.2)
        assert H._headspin_overspeed(env) == 0
        env.weight_overrides['headspin_visible'] = 0
        assert H._headspin_overspeed(env) < 0
    finally:
        env.close()
