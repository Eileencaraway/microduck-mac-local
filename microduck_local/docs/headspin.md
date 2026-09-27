# Headspin experiment — 2026-09-23

This is an experimental training recipe, not a validated headspin controller.
The first milestone is sustained head-supported rotation. Stopping after exactly
one turn and recovering to standing are not implemented in this recipe.

## Browser

Open the existing viewer, open **teach**, enter `headspin` (also accepts
`head spin`, `头顶旋转`, `头顶接地旋转`, `头转`). The behavior library reloads on
submission. No frontend build is required. The trainer also resolves the donor,
so an already-running lab can use this feature without being restarted.

The default is three stages of 1,000,000 steps each, with 32 environments.
An explicit user-selected step budget or saved headspin budget takes precedence.
Do not submit a second job while one is running.

The first stage inherits the newest **completed final-stage headstand** run with
`model.zip`, `vecnormalize.pkl`, and `policy.onnx`. A completed un-staged
headstand run is also eligible. Completion is not a skill certificate: inspect
the donor. No suitable donor produces an actionable error; an explicit
`initFrom`/`--init-from` overrides auto-selection. Existing other recipes are
unchanged. The donor used in the first experiment was
`teach-headstand-44d48a-s5`.

## What is optimized

All headstand reward terms are retained. Two terms are added:

- `headspin_turn`, weight **3.0**: signed world-up angular speed divided by
  **0.6 rad/s**, clipped to [-1, 1], paid only during a supported inverted stack.
  No turn earns zero; a backwards turn subtracts; wobbling back and forth cannot
  earn an absolute-speed bonus. Above-target speed earns no extra turn reward.
- `headspin_overspeed`, weight **1.0**: bounded quadratic penalty for supported
  rotation faster than **0.9 rad/s** in magnitude.

World-up speed is `dot(body_gyro, -projected_gravity)`, using existing observable
signals. Body gyro Z alone has the wrong interpretation when inverted. The
rotation direction is fixed; mirror regularization is disabled. The 61-observation
/14-action interface and the headstand donor's zero command slots are preserved.
Motor strengths, body geometry and contact friction are not modified.

The support gate requires head contact, no trunk/hip or foot contact, projected
inversion >0.8, and a nonzero existing headstand shape score (>0.05). This is a
proxy: `jaw_soft` is one body, so inspect the actual contact/pose visually.
No assisted showcase torque or policy handoff is used.

## Curriculum

| Stage | Inverted starts | Mid-entry starts | Standing starts | Initial kick |
| --- | ---: | ---: | ---: | ---: |
| Slow turns | 85% | 10% | 5% | 0.05 |
| Disturbances | 70% | 20% | 10% | 0.15 |
| Entry practice | 45% | 40% | 15% | 0.15 |

All stages use BAM motors, the same reward terms, headstand gate 0.8, and 12 s
episodes. Warm starts use the trainer's existing cool learning-rate schedule
(2e-4 to 3e-5 by default). Initializing inverted is a training aid and does not
prove the policy can enter the trick from standing. The final stage deliberately
retains many balancing starts; it does not establish standing-start reliability.

## Logs and evaluation

`runs/teach-headspin-<id>-s1..s3/` contains `train.log`, `progress.jsonl`,
`behavior.json`, and policy/checkpoint files. New run metadata also records donor
path, random seed, and relevant stage environment settings. Rewards remain
training diagnostics, not success rates.

From `microduck_local/`, evaluate a finished checkpoint separately:

```sh
.venv/bin/python scripts/eval_headspin.py \
  --policy runs/NAME/policy.onnx --seeds 3 --seed0 100 \
  --out runs/NAME/headspin-eval.json
```

This runs both standing-only and inverted-only starts, without assistance or
randomization, reports supported signed net rotation, longest support, and best
uninterrupted directed rotation. `full_turn_proxy` requires 360 degrees in one
supported bout, not several disconnected partial turns. Seeds 100–102 are an
initial diagnostic, not a statistically robust acceptance battery.

Render and inspect before asserting the visible trick works:

```sh
.venv/bin/render-rollout --behavior headspin --policy runs/NAME/policy.onnx \
  --env MICRODUCK_INVERTED_SPAWN_PROB=0 \
  --env MICRODUCK_MID_FLIP_SPAWN_PROB=0 \
  --episodes 2 --seconds 12 --out /tmp/headspin-review
```

The agent environment could run physics but could not create a macOS CoreGraphics
rendering context. Browser visuals remain available. Keep this limitation explicit.

## Scope and next experiment

First verify balance retention and supported turns against the donor on identical
seeds. If support degrades, decrease fine-tuning learning rate or turn incentive
and repeat a paired comparison. Do not declare a win from reward alone. A later
stop-and-recover task needs its own observable target/phase and evaluation.

## Visible-spin revision — 2026-09-23

The later fine-tuned model `teach-headspin-62778e` held its headstand, but the
standing-start battery (seeds 100–102) measured only 0.055/0.038/0.111 rad/s
supported mean rotation. The user's observation that the turn was barely visible
was confirmed. The previous bounded directional reward saturated early and did
not specify a preferred visible speed.

Added **headspin visible**, default weight **6.0**. During valid head support it
pays a Gaussian speed band centered on **1.2 rad/s**, width **0.8 rad/s**, with
the rest-state baseline subtracted and the result bounded to [0, 1]. Rest,
reverse rotation and excessive speed earn zero from this term. The slider changes
the incentive's strength, not the target speed. One turn at the target takes
about 5.24 seconds; this is a goal, not a guarantee.

When this slider is nonzero, the overspeed penalty starts at **1.8 rad/s** so it
does not fight the new target. Setting it to zero restores the original **0.9
rad/s** penalty knee. Original headstand rewards and the directional term remain.
No policy input, actuator, friction, symmetry rule or support metric changed.

Use **fine-tune the result**, not a fresh headstand initialization, to retain the
current model's entry and balance. The first comparison uses a 1M-step budget and
the same six baseline episodes. Judge visible motion and support together.
