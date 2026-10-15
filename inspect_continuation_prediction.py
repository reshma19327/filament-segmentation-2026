
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

import train_unetpp_spine_topology_finetune_512 as base
from unetpp_spine import UNetPPSpine

ROOT = Path(__file__).resolve().parent
STEM = "20181102104630Ch"

CHECKPOINT = (
    ROOT / "benchmark_models_512"
    / "unetpp_spine_topology_full10"
    / "best_by_cldice.pth"
)
MASK_PATH = ROOT / "data" / "full_masks" / f"{STEM}.png"
OUT_PATH = ROOT / "benchmark_models_512" / "checkpoint_comparison"
OUT_PATH.mkdir(parents=True, exist_ok=True)


def main():
    device = base.DEVICE
    _, records = base.load_split()
    record = next(
        (r for r in records if Path(r["image"]).stem.lower() == STEM.lower()),
        None,
    )
    if record is None:
        raise ValueError(f"{STEM} not found in validation split.")

    dataset = base.FilamentSpineDataset([record])
    sample = dataset[0]
    image = sample["image"].unsqueeze(0).to(device)

    model = UNetPPSpine(in_channels=1, base_channels=16).to(device)
    checkpoint = torch.load(
        CHECKPOINT, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    with torch.no_grad():
        logits, _ = base.get_outputs(model(image))
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    pred = prob >= 0.10
    gt = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(MASK_PATH)
    gt = cv2.resize(
        gt, (pred.shape[1], pred.shape[0]),
        interpolation=cv2.INTER_NEAREST
    ) > 127

    img = sample["image"][0].numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    overlay[gt & ~pred] = [255, 60, 60]
    overlay[pred & ~gt] = [50, 140, 255]
    overlay[gt & pred] = [50, 220, 100]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, data, title, cmap in [
        (axes[0], img, "Input", "gray"),
        (axes[1], gt, "Ground truth", "gray"),
        (axes[2], pred, "Continuation prediction (0.10)", "gray"),
        (axes[3], overlay, "Red=missed, blue=extra, green=match", None),
    ]:
        ax.imshow(data, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    out = OUT_PATH / f"{STEM}_continuation_threshold010.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()


