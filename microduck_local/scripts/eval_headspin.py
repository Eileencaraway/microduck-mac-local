"""Deterministic, unassisted headspin diagnostics; no rendering required.

Run from microduck_local:
  .venv/bin/python scripts/eval_headspin.py --policy runs/NAME/policy.onnx --out runs/NAME/headspin-eval.json
Use render-rollout and inspect its sheets before claiming visual success.
"""
import argparse
import json
import math
from pathlib import Path

from microduck_local.behaviors import _headspin_supported, _headspin_world_rate
from microduck_local.render_rollout import Probe, build_env, load_driver


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--policy', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--seed0', type=int, default=100)
    ap.add_argument('--seconds', type=float, default=12)
    args = ap.parse_args()
    driver = load_driver(args.policy)
    results = []
    for start, prob in [('standing', '0'), ('inverted', '1')]:
        for seed in range(args.seed0, args.seed0 + args.seeds):
            env = build_env('headspin', {
                'MICRODUCK_INVERTED_SPAWN_PROB': prob,
                'MICRODUCK_MID_FLIP_SPAWN_PROB': '0',
                'MICRODUCK_ACTUATOR': 'bam',
                'MICRODUCK_BAM_CURRENT_SCALE': '1.0',
                'MICRODUCK_HS_GATE': '0.8',
                'MICRODUCK_EPISODE_S': str(args.seconds),
            }, seed)
            try:
                obs, _ = env.reset(seed=seed)
                probe = Probe(env)
                streak = longest = 0
                rates = []
                for step in range(math.ceil(args.seconds / .02)):
                    obs, _, terminated, truncated, _ = env.step(driver.fn(obs, env))
                    good = _headspin_supported(env)
                    streak = streak + 1 if good else 0
                    longest = max(longest, streak)
                    if good:
                        rates.append(_headspin_world_rate(env))
                    if terminated or truncated:
                        break
                frame = probe.sample(step, step + 1, driver.label, False)
                row = dict(start=start, seed=seed, spawn=env.last_spawn,
                           elapsed_s=round((step + 1) * .02, 2),
                           supported_s=round(env._headspin_supported_s, 3),
                           longest_supported_s=round(longest * .02, 3),
                           supported_net_deg=round(math.degrees(env._headspin_net), 2),
                           best_continuous_deg=round(math.degrees(env._headspin_best), 2),
                           full_turn_proxy=env._headspin_best >= 2 * math.pi,
                           mean_supported_rate=sum(rates) / len(rates) if rates else None,
                           final_tilt_deg=frame.tilt_deg,
                           final_ground_bodies=frame.ground_bodies)
                results.append(row)
                print(json.dumps(row), flush=True)
            finally:
                env.close()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        'policy': args.policy, 'reference_world_up_rate': 1.2,
        'physics': 'BAM, 1.0 current scale; no assistance or handoff; observation noise and domain randomization off',
        'limitation': 'Support/crown is an orientation and contact proxy. Inspect video; inverted starts do not demonstrate entry. These few seeds are a diagnostic, not a robust success-rate estimate.',
        'results': results,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
