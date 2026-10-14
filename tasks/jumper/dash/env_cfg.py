"""jumper.dash's environment: jumper.jump's, playing `ref/dash_flat.npz`.

The reference is the one thing that differs, and it is named in two places:
the jump's command term loads the file this config names (every reward and
observation reads the reference through that term, so that is the whole of
training), and the contract the export writes loads a recording of its own
(`reference_contract`, which the jump's config closes over its own file). The
first dash export forgot the second and shipped `high_jump_flat` under the
dash's policy -- the bundler refused it only because the jump ships a file
of the same name, so the contract here is built from the dash's recording
by the same call. Resume from the jump's checkpoint:

    python scripts/train.py --task jumper.dash --headless --max-iterations 1500 --num_envs 2048 \\
        --checkpoint tasks/jumper/jump/out/example/model_3100.pt
"""
from __future__ import annotations

from pathlib import Path

from ..jump import env_cfg as _jump

DASH_REF_NPZ = Path(__file__).resolve().parent / "ref" / "dash_flat.npz"


def env_cfg(*args, **kwargs):
    cfg = _jump.env_cfg(*args, **kwargs)
    cfg.commands["jump"].motion_file = str(DASH_REF_NPZ)

    def _reference_contract() -> dict:
        # The jump's, with the dash's recording: see `_jump.env_cfg` for why the
        # joint order is this robot's rather than the file's.
        ref = _jump.jump_mdp.reference.JumpReference(DASH_REF_NPZ, "cpu", list(_jump.HOME))
        return ref.deploy_contract(
            _jump._FUTURE_OFFSETS, _jump._SIM_TIMESTEP * _jump._DECIMATION,
            starts_on_entry=_jump._STARTS_ON_ENTRY,
        )

    cfg.reference_contract = _reference_contract
    return cfg
