"""Shared prerequisite selection for browser and command-line training."""
import json
import re
from pathlib import Path


def resolve_prerequisite(behavior, runs_dir: Path) -> Path | None:
    """Pick a completed final-stage donor; explicit --init-from bypasses this.

    Completion is provenance, not a success certificate. Inspect and evaluate
    the donor and the new skill independently.
    """
    from .behaviors import BEHAVIORS

    prerequisite = behavior.warm_start_behavior
    if not prerequisite:
        return None
    donor = BEHAVIORS[prerequisite]
    candidates = []
    for run in runs_dir.glob('*'):
        if (not run.is_dir() or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,95}', run.name)
                or not all((run / f).is_file() for f in
                           ('model.zip', 'vecnormalize.pkl', 'policy.onnx'))):
            continue
        try:
            meta = json.loads((run / 'behavior.json').read_text())
            lines = (run / 'progress.jsonl').read_text().splitlines()
            if (meta.get('behavior') != prerequisite or not lines
                    or not json.loads(lines[-1]).get('done')):
                continue
            chain = re.fullmatch(r'teach-.+-s(\d+)', run.name)
            if chain and int(chain.group(1)) != len(donor.curriculum):
                continue
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        candidates.append(run)
    if not candidates:
        raise ValueError(f'{behavior.title} needs a completed {prerequisite} model first. '
                         f'Train {prerequisite}, then ask for {behavior.id} again.')
    return max(candidates, key=lambda run: (run / 'model.zip').stat().st_mtime)


def initialize_launch_commands(model, normalizer):
    """Preserve donor behavior while introducing six previously unused inputs.

    Apply only on transfer INTO headspin_launch, never when resuming it.
    Zero first-layer columns and corresponding Adam moments, then permit
    learning normally; no freezing. Both actor and critic paths are covered.
    """
    import torch
    normalizer.obs_rms.mean[55:61] = 0.0
    normalizer.obs_rms.var[55:61] = 1.0
    count = 0
    with torch.no_grad():
        for layer in model.policy.mlp_extractor.modules():
            if isinstance(layer, torch.nn.Linear) and layer.in_features == 61:
                layer.weight[:, 55:61] = 0.0
                for value in model.policy.optimizer.state.get(layer.weight, {}).values():
                    if torch.is_tensor(value) and value.shape == layer.weight.shape:
                        value[:, 55:61] = 0.0
                count += 1
    if count == 0:
        raise ValueError('Cannot identify 61-input donor layers; refusing unsafe command transfer')
    return count
