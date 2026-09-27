# Microduck Headspin Experiment

[中文说明](README.zh-CN.md)

This branch adds two experimental reinforcement-learning behaviors to
[Microduck Lab](https://github.com/jonathanhawkins/microduck-lab):

- `headspin`: enter a headstand and learn a directed, visible rotation while
  supported on the head.
- `headspin_launch`: build yaw momentum from a standing start, transition to
  head support, and sustain a fast one-to-four-turn spin. One continuous turn
  counts as success; the second through fourth turns earn progressively more.

## Status

**Experimental simulation prototype.** This repository contains a new behavior
definition, curriculum, training-transfer logic, tests, and deterministic
evaluation tools. It does not contain a validated physical-robot controller.

The current results do not demonstrate a reliable full headspin:

| Policy | Deterministic setting | Result |
| --- | --- | --- |
| `teach-headspin-f95723` | 3 standing + 3 inverted starts | 0/6 uninterrupted full turns; best observed proxy angle about 225° |
| `teach-headspin_launch-e743c8-s4` | 6 motionless standing starts, no yaw assist | 0/6 complete launch-turn-settle successes; best trunk/head proxy rotation about 0.38/0.54 turns |
| `teach-headspin_launch-9406bd` | 20 motionless standing starts, no yaw assist | 11/20 exceeded one continuous turn; 1/20 exceeded two; best head-and-trunk proxy 2.11 turns |

These are contact- and angular-rate-based simulation proxies. Visual rollout
review is still required. The launch experiment also exposed a limitation in
its 50 Hz foot-contact impulse proxy: positive takeoff angular momentum can be
paired with a negative sampled impulse, which clears launch quality. Treat that
measurement as an open issue, not evidence that the foot launch is solved.

## What changed

The implementation preserves Microduck's shared 61-observation / 14-action
policy contract.

- `behaviors/headstand.py` defines both behaviors, their rewards, state
  machines, curricula, reports, and success proxies.
- `behaviors/core.py` lets a behavior declare a prerequisite donor policy.
- `behaviors/env.py` refreshes task command slots after launch-state changes.
- `training_init.py` adapts a donor network to the six task-command inputs
  without changing its initial output.
- `train_behavior.py` records the donor, seed, and curriculum environment in
  run metadata and applies donor input initialization.
- `viz_server.py` resolves completed prerequisite runs for browser training.
- `scripts/eval_headspin*.py` run deterministic, assistance-free batteries.
- `tests/test_headspin*.py` lock the reward, curriculum, observation, and
  finite-turn contracts.

Detailed design notes are in [headspin.md](../headspin.md) and
[headspin-launch.md](../headspin-launch.md).

## Reproduce the source environment

This prototype was developed from Microduck Lab commit `54989df` and is kept
on the branch `headspin-prototype-2026-09-23` so the source that produced the
experiment remains reproducible.

```bash
git clone --branch headspin-prototype-2026-09-23 \
  https://github.com/Eileencaraway/microduck-mac-local.git
cd microduck-mac-local
./scripts/setup.sh
```

Start the lab as described in the parent project README, open the Teach panel,
and enter `headspin` or `headspin_launch`. A prerequisite run must contain
`model.zip`, `vecnormalize.pkl`, and `policy.onnx`; the browser reports an
actionable error when no eligible donor exists.

The local `runs/` directory is intentionally ignored. Existing run names in
this document identify the recorded experiments on the author's machine; they
are not bundled checkpoints.

## Tests

From `microduck_local/`:

```bash
uv run --with pytest pytest -q \
  tests/test_headspin.py \
  tests/test_headspin_launch.py \
  tests/test_behaviors.py \
  tests/test_lab.py
```

The focused suite passed 212 tests when the reverse curriculum was introduced.
This statement applies to this prototype branch and the recorded dependency
environment; it is not a claim that every upstream test was run.

## Evaluate a trained policy

Continuous headspin:

```bash
.venv/bin/python scripts/eval_headspin.py \
  --policy runs/NAME/policy.onnx \
  --seeds 6 --seed0 100 \
  --out /tmp/headspin-eval.json
```

Foot-driven finite headspin:

```bash
.venv/bin/python scripts/eval_headspin_launch.py \
  --policy runs/NAME/policy.onnx \
  --seeds 6 --seed0 300 \
  --out /tmp/headspin-launch-eval.json
```

Do not infer success from reward curves. A completion claim requires fresh
standing-start seeds, the deterministic ONNX battery, and visual rollout
inspection.

## Limitations and intended use

- The policies were trained in the local CPU MuJoCo prototype stack, with a
  smaller domain-randomization subset than the official GPU stack.
- `headspin_launch` uses simulator-derived momentum, contact, phase, and
  progress signals in existing command slots. A real robot needs estimators
  and runtime support for those signals.
- Assisted inverted starts in early curriculum stages are training aids. Only
  the final standing-only, zero-kick stage is relevant to autonomous launch.
- The current goal was reduced from two turns to one because two turns created
  an exploration gap. Even one complete turn and recovery is not yet reliable.
- Do not deploy these policies on physical hardware. Port the behavior to the
  official [`microduck_rl`](https://github.com/pollen-robotics/microduck_rl)
  stack, retrain with its sim-to-real randomization, and validate safely first.

## Repository hygiene

This branch includes source, tests, and documentation. It deliberately excludes
virtual environments, tokens, browser state, local training runs, raw
checkpoints, and generated captures. A deployable ONNX and its model card can
be published separately on the Hugging Face Hub after an evaluation artifact
and intended runtime are selected.

## License and attribution

The code is distributed under the parent repository's Apache-2.0 license. It
builds on Microduck Lab by Jonathan Hawkins and on the open-source Microduck
and `microduck_rl` projects by Pollen Robotics. Modified files are recorded in
Git history.

Development and documentation were assisted by OpenAI Codex. Experimental
decisions, evaluation interpretation, and publication remain the repository
owner's responsibility.
