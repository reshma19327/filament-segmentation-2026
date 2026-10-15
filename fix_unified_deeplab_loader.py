
from pathlib import Path

path = Path("unified_evaluation_7_models_512.py")
source = path.read_text(encoding="utf-8")

start_marker = '    if model_type == "deeplabv3":'
end_marker = '    model.to(DEVICE)'

start = source.find(start_marker, source.find("def load_torchvision_model("))
end = source.find(end_marker, start)

if start < 0 or end < 0:
    raise RuntimeError("Could not locate the expected loader section. No changes made.")

replacement = '''    if model_type == "deeplabv3":
        model = deeplabv3_resnet50(
            weights=None,
            weights_backbone=None,
            num_classes=1,
        )

        # Match the grayscale architecture used during training.
        old_conv = model.backbone.conv1
        model.backbone.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

        model.load_state_dict(state_dict, strict=True)

        print(
            "DeepLabV3 checkpoint loaded strictly; "
            f"input channels={model.backbone.conv1.in_channels}"
        )

    else:
        attempts = [
            lambda: maskrcnn_resnet50_fpn(
                weights=None,
                weights_backbone=None,
                num_classes=2,
            ),
            lambda: maskrcnn_resnet50_fpn(
                weights=None,
                weights_backbone=None,
                num_classes=1,
            ),
        ]

        best = None

        for make_model in attempts:
            try:
                candidate = make_model()
                candidate_state = candidate.state_dict()

                compatible = {
                    key: value
                    for key, value in state_dict.items()
                    if key in candidate_state
                    and tuple(value.shape) == tuple(candidate_state[key].shape)
                }

                score = len(compatible) / max(1, len(candidate_state))

                if best is None or score > best[0]:
                    best = (score, candidate, compatible)

            except Exception:
                continue

        if best is None or best[0] < 0.80:
            raise RuntimeError(
                "Mask R-CNN checkpoint is incompatible with the reconstructed model."
            )

        _, model, compatible = best
        model.load_state_dict(compatible, strict=False)
        print("Mask R-CNN compatible checkpoint weights loaded.")

'''

path.write_text(source[:start] + replacement + source[end:], encoding="utf-8")
print("Patched DeepLabV3 loader.")
print("Original evaluator backup should remain untouched.")
