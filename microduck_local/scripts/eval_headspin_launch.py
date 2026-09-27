"""Evaluate deterministic foot-driven headspin from standing, without assists."""
import argparse
import json
from pathlib import Path
from microduck_local.render_rollout import build_env, load_driver
from microduck_local.behaviors import _launch_lz


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--policy', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--seed0', type=int, default=200)
    args = ap.parse_args()
    driver = load_driver(args.policy)
    rows = []
    knobs = {'MICRODUCK_ACTUATOR': 'bam', 'MICRODUCK_BAM_CURRENT_SCALE': '1.0',
             'MICRODUCK_INVERTED_SPAWN_PROB': '0', 'MICRODUCK_MID_FLIP_SPAWN_PROB': '0',
             'MICRODUCK_INV_SPAWN_KICK': '0', 'MICRODUCK_HS_GATE': '0.8',
             'MICRODUCK_LAUNCH_YAW_KICK': '0', 'MICRODUCK_LAUNCH_TURNS': '4',
             'MICRODUCK_LAUNCH_SUPPORT_GAP': '0.6', 'MICRODUCK_EPISODE_S': '12'}
    for seed in range(args.seed0, args.seed0 + args.seeds):
        env = build_env('headspin_launch', knobs, seed)
        try:
            obs, _ = env.reset(seed=seed)
            initial_lz = _launch_lz(env)
            trace = []
            success = False
            for step in range(700):
                obs, reward, done, truncated, _ = env.step(driver.fn(obs, env))
                success |= env._ls_success
                if step % 10 == 0 or done or truncated:
                    trace.append({'t': round((step + 1) * .02, 2), 'phase': env._ls_phase,
                                  'trunk_turns': env._ls_turn / 6.283185307179586,
                                  'head_turns': env._ls_head_turn / 6.283185307179586,
                                  'launch_quality': env._ls_quality, 'lz': _launch_lz(env),
                                  'foot_impulse_proxy': env._ls_foot_impulse})
                if done or truncated:
                    break
            row = {'seed': seed, 'spawn': env.last_spawn, 'initial_lz': initial_lz,
                   'launch_lz': env._ls_takeoff_lz, 'launch_quality': env._ls_quality,
                   'foot_impulse_proxy': env._ls_foot_impulse,
                   'trunk_turns': env._ls_turn / 6.283185307179586,
                   'head_turns': env._ls_head_turn / 6.283185307179586,
                   'peak_supported_rate': env._ls_peak_rate,
                   'settled_s_at_end': env._ls_settled,
                   'proxy_success': success, 'trace': trace}
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'trace'}), flush=True)
        finally:
            env.close()
    Path(args.out).write_text(json.dumps({'policy': args.policy,
        'limits': 'World-up rate integrals and 50Hz foot impulse are proxies, not exact yaw/impulse. Visual confirmation required. Simulator-derived task signals require estimators for hardware.',
        'physics': knobs, 'results': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main()
