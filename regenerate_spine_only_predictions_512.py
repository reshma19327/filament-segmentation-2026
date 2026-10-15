from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader

import train_benchmark_unetpp_spine_only_512 as training

def main():
    checkpoint_path = Path(
        "benchmark_models_512/unetpp_spine_only/"
        "best_unetpp_spine_only_512.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    train_records, val_records = training.load_split()
    dataset = training.FilamentSpineDataset(val_records)
    loader = DataLoader(
        dataset,
        batch_size=training.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(training.DEVICE.type == "cuda"),
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=training.DEVICE,
        weights_only=False,
    )

    model = training.UNetPPSpine(
        in_channels=1,
        base_channels=16,
    ).to(training.DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded best checkpoint epoch:", checkpoint["epoch"])
    print("Best validation Dice:", checkpoint["best_dice"])
    print("Validation images:", len(dataset))

    training.save_predictions(model, loader)

    # Verify one filament prediction per validation image.
    output_dir = training.PREDICTION_DIR
    expected_names = [record["stem"] for record in val_records]
    missing = [
        name for name in expected_names
        if not (output_dir / f"{name}_filament.png").exists()
    ]

    print("Expected validation images:", len(expected_names))
    print("Saved filament predictions:", len(expected_names) - len(missing))
    print("Missing predictions:", len(missing))

    if missing:
        print("First missing names:", missing[:10])
        raise RuntimeError("Some validation predictions are missing.")

    print("Prediction regeneration and verification complete.")

if __name__ == "__main__":
    main()
