# Foot-driven headspin experiment — 2026-09-23

Browser command: `headspin_launch` / `撑地启动头转`. This is a new recipe,
separate from continuous `headspin`. First run: `teach-headspin_launch-5c5536-s1`.

Goal: start motionless on the feet, build positive world-z angular momentum
through foot-ground contact, carry it into two head-supported turns, then
settle lying flat for 0.5 s. Success is NOT established by implementing this.

One 3M-step training stage, BAM current scale 1.0, zero inverted/mid-entry
spawns and zero spin kicks, 12 s episodes. Explicit donor:
`teach-headstand-44d48a-s5`. No spotter, no injected external forces.

Within each episode:
- Push (up to 3 s): retain headstand entry terms; reward new bounded records
  of the smaller of positive total Lz and net foot-ground angular impulse.
- Spin (until 2 turns, support loss >0.1 s, or 10 s total): remove entry
  salaries. Rotation/balance pay is scaled by launch quality, measured at
  last foot contact. A foot impulse proxy below 60% of launch Lz disqualifies
  it. Head AND trunk must accumulate two turns; neck-only twisting fails.
- Recover: turn off headstand/body-drag rewards, pay calm flat grounded
  recovery in proportion to earned rotation and launch quality. Success
  additionally requires both two turns and launch quality >=0.5.

Initial experimental references: 2 turns, 4 rad/s, Lz=0.015 kg m²/s,
7 rad/s overspeed threshold. They are recipe constants, not demonstrated
physical capabilities or additional UI speed controls. UI sliders control
weights: launch_push=60, launch_spin=16, launch_balance=3,
launch_settle=20, launch_overspeed=2.

Observation/action dimensions remain 61/14. Body command slots 55:61 carry
phase/2, remaining fraction, elapsed/12, launch quality, Lz/reference,
and foot contact. These include simulator-derived controller signals;
a hardware deployment would require estimators. Fresh transfer into this
recipe clears the donor actor/critic input columns and Adam moments for
those six slots and sets their normalizer mean/variance to 0/1. Same-recipe
fine-tunes/resumes retain learned command weights. Actor output was verified
unchanged at transfer despite nonzero new commands. Observations are refreshed
after phase transitions without resampling the other observation noise.

Limitations: contact torque is sampled at 50Hz, not integrated every physics
substep. Head/trunk rotation uses signed world-up angular-rate integrals,
which are proxies under tilt/precession. Contact/crown classification is
also a proxy. Neither accumulated reward nor this success flag substitutes
for watching a deterministic rollout. No claim of robust success or of
feasibility of 2–5 turns on hardware.

Evaluate after export (or use a copy of live.onnx for early diagnostics):
```
.venv/bin/python scripts/eval_headspin_launch.py --policy runs/NAME/policy.onnx --out /tmp/launch-eval.json
```
If rollouts never produce foot-driven rotation, change exploration/physics
curriculum and document it, rather than only increasing reward weights.

## Reverse-curriculum revision — 2026-09-23

The first three standing-only runs converged on launch shaping without a
usable turn. The final two runs had good launch-quality proxies in several
deterministic seeds but zero complete turns. A headstand donor could still
reach head support, so the main problem was the exploration gap and the reward
transition, not simply too few steps.

The revised recipe warm-starts from the completed `headspin` policy and uses
four 750k-step stages:

| stage | inverted / mid / standing | yaw kick | target |
|---|---:|---:|---:|
| preserve supported turn | 85 / 10 / 5% | 1.5 rad/s | 0.25 turn |
| carry through entry | 65 / 25 / 10% | 1.0 rad/s | 0.5 turn |
| transfer momentum | 30 / 30 / 40% | 0.5 rad/s | 0.75 turn |
| honest foot launch | 0 / 0 / 100% | 0 | 1 turn |

The assist is declared in `MICRODUCK_LAUNCH_YAW_KICK` and appears through the
observed launch-quality slot. It exists only in the first three training
stages. The final stage and evaluation remain motionless, standing-only and
unassisted. The target also rides in the remaining-turn observation via
`MICRODUCK_LAUNCH_TURNS`; the observation/action dimensions stay 61/14.

Headstand rewards now remain active during both entry and supported rotation,
and switch off only for recovery. The support-loss tolerance is 0.6 s instead
of 0.1 s. Reward weights changed from push/spin/balance/settle = 60/16/3/20
to 24/24/8/12. This reduces the attractive launch-only solution and protects
the inherited balance skill. The first target is one full turn; 2–5 turns
remain a later extension after one-turn standing starts pass deterministic and
visual evaluation.

Revision run: `teach-headspin_launch-e743c8`, explicitly initialized from
`teach-headspin-f95723`. Do not compare its assisted early-stage reward as a
success rate for the final standing task.

## Fast multi-turn revision — 2026-09-27

The user's desired finish is now unconstrained: falling flat, remaining
inverted, or returning to the feet are all acceptable after the spin. Success
means at least one continuous, positive turn by both head and trunk. Foot-launch
quality remains a separately reported diagnostic and reward multiplier, rather
than a second success gate. There is no recovery-pose or settling requirement
and the `launch_settle` reward was removed.

The curriculum now targets 0.5, 1, 2, and 4 turns. Its final stage is still an
honest motionless standing start with no yaw injection. The dense
`launch_spin` reward is normalized at 5 rad/s and grows from a 1x multiplier
near takeoff to 2x during the fourth turn; therefore one turn passes, while
continuing through turns two to four remains the better solution. The
overspeed penalty begins above 8 rad/s.

Training chain `teach-headspin_launch-5cc39d` warm-starts stage 1 from
`teach-headspin_launch-e743c8-s4`, using 750k steps per stage and
`launch_spin=32`. Deterministic unassisted evaluation must use the final `s4`
policy and the revised four-turn evaluation horizon before any success claim.

That chain was followed by a 1.5M-step, final-stage-only fine-tune named
`teach-headspin_launch-9406bd`. Across deterministic seeds 300–319 from
motionless standing starts with no yaw assist, 11/20 runs exceeded one
continuous head-and-trunk turn, 1/20 exceeded two, and the best reached 2.110
turns. Median turns were 1.253; median supported peak speed was 7.137 rad/s.
Only 2/20 runs had foot-launch quality at least 0.5, so this result demonstrates
more rotation under the simulator proxy, not reliable foot-dominant launch.
Visual rollout inspection is still required.
