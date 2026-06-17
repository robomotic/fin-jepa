"""Standalone runner for Experiment 5: Yield Curve Sanity Check."""
import sys
from pathlib import Path
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

from data.pipeline import build_pipeline, load_config
from model.jepa import JEPA, JEPAConfig
from experiments.exp5_yield_curve_sanity import run_experiment_5

config_path = "config/variables.yaml"
checkpoint  = "checkpoints/best.pt"
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")
config = load_config(config_path)

print("Loading cached splits (no rebuild) ...")
splits = build_pipeline(config_path=config_path, force_rebuild=False)

ckpt = torch.load(checkpoint, map_location=device)
jepa_cfg = JEPAConfig(
    n_features=ckpt["n_features"],
    **{k: v for k, v in ckpt["config"].items() if k in JEPAConfig.__dataclass_fields__},
)
jepa = JEPA(jepa_cfg).to(device)
jepa.load_state_dict(ckpt["model"])
jepa.eval()
print(f"Loaded checkpoint: n_features={ckpt['n_features']}")

val_test_panel = pd.concat([splits["val"], splits["test"]]).sort_index()
val_test_panel = val_test_panel[~val_test_panel.index.duplicated(keep="last")]
print(f"Panel: {val_test_panel.shape}  columns: {list(val_test_panel.columns)[:5]} ...")

result = run_experiment_5(
    jepa, val_test_panel, config, device,
    output_dir=Path("results/exp5"),
)

print("\n=== Experiment 5 Result ===")
for k, v in result.items():
    print(f"  {k}: {v}")
