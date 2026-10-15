import torch
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

import train_unetpp_spine_topology_finetune_512 as train
from unetpp_spine import UNetPPSpine

ROOT = Path.cwd()
CHECKPOINT = ROOT / "benchmark_models_512" / "unetpp_spine_topology_finetune_cldice" / "best_unetpp_spine_topology_512.pth"
OUT_DIR = ROOT / "benchmark_models_512" / "unetpp_spine_topology_finetune_cldice" / "predictions_all142"

OUT_DIR.mkdir(parents=True, exist_ok=True)

_, val_records = train.load_split()
dataset = train.FilamentSpineDataset(val_records)
loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

device = train.DEVICE
model = UNetPPSpine(in_channels=1, base_channels=16).to(device)

checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()

saved = 0
with torch.no_grad():
    for batch in loader:
        images = batch["image"].to(device)
        outputs = model(images)
        filament_logits, spine_logits = train.get_outputs(outputs)

        probabilities = torch.sigmoid(filament_logits).cpu().numpy()
        spine_probabilities = torch.sigmoid(spine_logits).cpu().numpy()
        input_images = batch["image"].cpu().numpy()
        names = batch["name"]

        for i, name in enumerate(names):
            filament = (probabilities[i, 0] >= train.THRESHOLD).astype(np.uint8) * 255
            spine = (spine_probabilities[i, 0] >= train.THRESHOLD).astype(np.uint8) * 255
            img = np.clip(input_images[i, 0] * 255, 0, 255).astype(np.uint8)

            Image.fromarray(filament).save(OUT_DIR / f"{name}_filament.png")
            Image.fromarray(spine).save(OUT_DIR / f"{name}_spine.png")
            Image.fromarray(img).save(OUT_DIR / f"{name}_input.png")

            overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            overlay[filament > 0] = (0, 255, 0)
            overlay[spine > 0] = (0, 0, 255)
            Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)).save(
                OUT_DIR / f"{name}_overlay.png"
            )
            saved += 1

print(f"\nValidation records: {len(val_records)}")
print(f"Prediction sets saved: {saved}")
print(f"Output folder: {OUT_DIR}")
