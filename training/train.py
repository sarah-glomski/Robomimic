"""
Training entry point for diffusion policy on xArm teleop data.

Adds the UMI codebase and dt_ag to sys.path, registers Hydra resolvers,
then delegates to the standard TrainDiffusionUnetImageWorkspace.

Usage:
    python train.py --config-name=train_diffusion_unet_timm_xarm
    python train.py --config-name=train_diffusion_unet_timm_xarm training.debug=True
    python train.py --config-name=train_diffusion_unet_timm_xarm logging.mode=disabled
"""

import os
import sys

# ── Path setup (before any diffusion_policy imports) ──────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_UMI_ROOT = os.path.join(_THIS_DIR, "..", "dt_ag-main")
_UMI_DP = os.path.join(_UMI_ROOT, "universal_manipulation_interface")

for p in [_UMI_DP, _UMI_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Hydra / OmegaConf setup ──────────────────────────────────────────────
import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, use_cache=True)

from diffusion_policy.workspace.train_diffusion_unet_image_workspace import (
    TrainDiffusionUnetImageWorkspace,
)


@hydra.main(
    version_base=None,
    config_path=os.path.join(_THIS_DIR, "config"),
)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
