# ================================================================
# 512x512 UNET++ + SPINE + clDice
# IMPROVED THIN-FILAMENT TRAINING
#
# IMPORTANT:
# - Self-contained dataset implementation
# - Does NOT use dataset_unetpp_spine.py
# - Keeps real image filenames including extensions
# - Uses existing unetpp_spine.py
# - Uses UNetPPSpine(in_channels=1, base_channels=16)
# - No sklearn
# - No tqdm
# ================================================================




import os
import csv
import time
import random
from pathlib import Path




import cv2
import numpy as np




import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader




from unetpp_spine import UNetPPSpine








# ================================================================
# PROJECT CONFIGURATION
# ================================================================




PROJECT_ROOT = Path(
    r"C:\Users\Kevin\Downloads\filament-segmentation-2026"
)




IMAGE_DIR = PROJECT_ROOT / (
    "MAGFiLO_1.0_Kaggle_2026"
    r"\train\train_images"
)




MASK_DIR = PROJECT_ROOT / r"data\full_masks"




SPINE_DIR = PROJECT_ROOT / r"data\full_spine_masks"








# ================================================================
# OUTPUT DIRECTORIES
# ================================================================




OUTPUT_DIR = (
    PROJECT_ROOT / "checkpoints_512_cldice_smoothfix"
)




OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)




BEST_CHECKPOINT = (
    OUTPUT_DIR / "best_unetpp_spine_512_cldice_improved_dice.pth"
)
BEST_CLDICE_CHECKPOINT = (
    OUTPUT_DIR / "best_unetpp_spine_512_cldice_improved_cldice.pth"
)




LAST_CHECKPOINT = (
    OUTPUT_DIR /
    "last_unetpp_spine_512_cldice_improved.pth"
)




CSV_LOG = (
    OUTPUT_DIR /
    "training_history.csv"
)








# ================================================================
# TRAINING CONFIGURATION
# ================================================================




IMAGE_SIZE = (512, 512)




TRAIN_RATIO = 0.80




SEED = 42




BATCH_SIZE = 2




NUM_WORKERS = 0




EPOCHS = 60




LEARNING_RATE = 5e-5




WEIGHT_DECAY = 1e-4




GRADIENT_CLIP = 1.0




USE_AMP = True




ACCUMULATION_STEPS = 2








# ================================================================
# LOSS WEIGHTS
# ================================================================




FILAMENT_BCE_WEIGHT = 0.40




FILAMENT_DICE_WEIGHT = 1.00




CLDICE_WEIGHT = 0.80




SPINE_BCE_WEIGHT = 0.20




SPINE_DICE_WEIGHT = 0.25








# ================================================================
# clDice CONFIGURATION
# ================================================================




CLDICE_ITERATIONS = 10




CLDICE_SMOOTH = 1e-6








# ================================================================
# TRAINING BEHAVIOR
# ================================================================




USE_AUGMENTATION = True




# Fine-tune the already trained 512x512 model.
#
# This is intentional:
# your previous 512x512 model reached approximately 0.6825
# validation Dice before threshold tuning.
#
# Starting from that checkpoint lets clDice focus on improving
# the thin filament structure rather than starting from zero.
#
# Set to None if you want completely fresh training.
RESUME_CHECKPOINT = (
    PROJECT_ROOT /
    r"checkpoints_512_improved\best_unetpp_spine_512_improved.pth"
)




# Start the optimizer at a lower learning rate when resuming.
RESUME_LEARNING_RATE = 2e-5








# ================================================================
# REPRODUCIBILITY
# ================================================================




def seed_everything(seed=42):




    random.seed(seed)




    np.random.seed(seed)




    torch.manual_seed(seed)




    torch.cuda.manual_seed_all(seed)




    os.environ["PYTHONHASHSEED"] = str(seed)




    # Deterministic behavior.
    # This may be slightly slower but keeps the experiment
    # reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False








# ================================================================
# DEVICE
# ================================================================




def get_device():




    if torch.cuda.is_available():




        device = torch.device("cuda")




        print("=" * 70)
        print("CUDA GPU ENABLED")
        print("=" * 70)




        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )




        print(
            "CUDA:",
            torch.version.cuda
        )




        print("=" * 70)




        return device




    print("=" * 70)
    print("CUDA NOT AVAILABLE")
    print("Using CPU")
    print("=" * 70)




    return torch.device("cpu")








# ================================================================
# FILE DISCOVERY
# ================================================================




IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}




MASK_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}








def get_files(directory):




    directory = Path(directory)




    if not directory.exists():




        raise FileNotFoundError(
            f"Directory does not exist:\n{directory}"
        )




    files = []




    for path in directory.iterdir():




        if not path.is_file():
            continue




        if path.suffix.lower() in MASK_EXTENSIONS:
            files.append(path)




    return sorted(files)








def discover_dataset():




    print()
    print("=" * 70)
    print("DATASET DISCOVERY")
    print("=" * 70)




    print("Images:", IMAGE_DIR)
    print("Masks :", MASK_DIR)
    print("Spines:", SPINE_DIR)




    image_files = get_files(IMAGE_DIR)




    mask_files = get_files(MASK_DIR)




    spine_files = get_files(SPINE_DIR)




    print()
    print("Image files:", len(image_files))
    print("Mask files :", len(mask_files))
    print("Spine files:", len(spine_files))




    # Map masks by stem.
    mask_lookup = {
        path.stem: path
        for path in mask_files
    }




    spine_lookup = {
        path.stem: path
        for path in spine_files
    }




    valid_records = []




    for image_path in image_files:




        stem = image_path.stem




        if stem not in mask_lookup:
            continue




        if stem not in spine_lookup:
            continue




        valid_records.append(
            {
                "image": image_path,
                "mask": mask_lookup[stem],
                "spine": spine_lookup[stem],
                "name": image_path.name,
                "stem": stem,
            }
        )




    print()
    print(
        "Images with both masks:",
        len(valid_records)
    )




    if len(valid_records) == 0:




        raise RuntimeError(
            "No images with both filament and spine masks were found."
        )




    print()
    print("First 5 matched files:")




    for record in valid_records[:5]:




        print(
            "Image:",
            record["image"].name
        )




        print(
            "Mask :",
            record["mask"].name
        )




        print(
            "Spine:",
            record["spine"].name
        )




        print()




    return valid_records








# ================================================================
# DATASET
# ================================================================




class FilamentSpineDataset(Dataset):




    def __init__(
        self,
        records,
        image_size=(512, 512),
        augment=False,
    ):




        self.records = list(records)




        self.image_size = image_size




        self.augment = augment




    def __len__(self):




        return len(self.records)




    def _load_grayscale(self, path):




        image = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE
        )




        if image is None:




            raise RuntimeError(
                f"Could not read file:\n{path}"
            )




        return image




    def _augment(
        self,
        image,
        mask,
        spine,
    ):




        # Horizontal flip
        if random.random() < 0.5:




            image = np.fliplr(image).copy()




            mask = np.fliplr(mask).copy()




            spine = np.fliplr(spine).copy()




        # Vertical flip
        if random.random() < 0.5:




            image = np.flipud(image).copy()




            mask = np.flipud(mask).copy()




            spine = np.flipud(spine).copy()




        # 90-degree rotation
        if random.random() < 0.5:




            k = random.randint(1, 3)




            image = np.rot90(
                image,
                k
            ).copy()




            mask = np.rot90(
                mask,
                k
            ).copy()




            spine = np.rot90(
                spine,
                k
            ).copy()




        return image, mask, spine




    def __getitem__(self, index):




        record = self.records[index]




        image_path = record["image"]




        mask_path = record["mask"]




        spine_path = record["spine"]




        # --------------------------------------------------------
        # Load
        # --------------------------------------------------------




        image = self._load_grayscale(
            image_path
        )




        mask = self._load_grayscale(
            mask_path
        )




        spine = self._load_grayscale(
            spine_path
        )




        # --------------------------------------------------------
        # Resize
        # --------------------------------------------------------




        image = cv2.resize(
            image,
            self.image_size,
            interpolation=cv2.INTER_AREA
        )




        mask = cv2.resize(
            mask,
            self.image_size,
            interpolation=cv2.INTER_NEAREST
        )




        spine = cv2.resize(
            spine,
            self.image_size,
            interpolation=cv2.INTER_NEAREST
        )




        # --------------------------------------------------------
        # Normalize
        # --------------------------------------------------------




        image = (
            image.astype(np.float32)
            / 255.0
        )




        mask = (
            mask > 127
        ).astype(np.float32)




        spine = (
            spine > 127
        ).astype(np.float32)




        # --------------------------------------------------------
        # Augmentation
        # --------------------------------------------------------




        if self.augment:




            image, mask, spine = self._augment(
                image,
                mask,
                spine
            )




        # --------------------------------------------------------
        # Tensor conversion
        # --------------------------------------------------------




        image = torch.from_numpy(
            image
        ).unsqueeze(0)




        mask = torch.from_numpy(
            mask
        ).unsqueeze(0)




        spine = torch.from_numpy(
            spine
        ).unsqueeze(0)




        return {
            "image": image,
            "mask": mask,
            "spine": spine,
            "name": record["name"],
        }








# ================================================================
# TRAIN / VALIDATION SPLIT
# ================================================================




def split_records(records):
    """Use the fixed benchmark split to keep evaluation reproducible."""
    import json




    split_path = PROJECT_ROOT / "benchmark_512" / "benchmark_split.json"
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)




    train_names = [item["image"].lower() for item in split["train"]]
    val_names = [item["image"].lower() for item in split["validation"]]




    by_name = {}
    for record in records:
        image_value = record.get("image")
        if image_value is None:
            raise RuntimeError("Dataset record has no 'image' field.")
        name = Path(str(image_value)).name.lower()
        if name in by_name:
            raise RuntimeError(f"Duplicate discovered image filename: {name}")
        by_name[name] = record




    train_set, val_set = set(train_names), set(val_names)
    missing = (train_set | val_set) - set(by_name)
    extra = set(by_name) - (train_set | val_set)
    overlap = train_set & val_set
    if missing or extra or overlap:
        raise RuntimeError(
            f"Split mismatch: {len(missing)} missing, {len(extra)} extra, "
            f"{len(overlap)} overlapping filenames."
        )
    train_records = [by_name[name] for name in train_names]
    val_records = [by_name[name] for name in val_names]




    if len(train_records) != 565 or len(val_records) != 142:
        raise RuntimeError(
            f"Unexpected split sizes: train={len(train_records)}, "
            f"validation={len(val_records)}"
        )




    print("Loaded saved benchmark split:", split_path)
    print("Training images:", len(train_records))
    print("Validation images:", len(val_records))
    return train_records, val_records








# ================================================================
# SANITY CHECK
# ================================================================




def sanity_check(dataset):




    print()
    print("=" * 70)
    print("DATASET SANITY CHECK")
    print("=" * 70)




    sample = dataset[0]




    print(
        "Image:",
        sample["name"]
    )




    print(
        "Image shape:",
        tuple(sample["image"].shape)
    )




    print(
        "Mask shape :",
        tuple(sample["mask"].shape)
    )




    print(
        "Spine shape:",
        tuple(sample["spine"].shape)
    )




    print(
        "Image dtype:",
        sample["image"].dtype
    )




    print(
        "Mask dtype :",
        sample["mask"].dtype
    )




    print(
        "Spine dtype:",
        sample["spine"].dtype
    )




    print(
        "Image range:",
        f"{sample['image'].min().item():.4f}",
        "->",
        f"{sample['image'].max().item():.4f}",
    )




    print(
        "Mask values:",
        torch.unique(
            sample["mask"]
        ).tolist()
    )




    print(
        "Spine values:",
        torch.unique(
            sample["spine"]
        ).tolist()
    )




    print()
    print("SANITY CHECK PASSED")








# ================================================================
# MODEL OUTPUT HANDLING
# ================================================================




def extract_outputs(outputs):




    """
    Supports common output formats:




    tuple/list:
        (filament, spine)




    dict:
        {
            "mask": filament,
            "spine": spine
        }




    dict alternatives:
        "filament"
        "mask_logits"
        "spine_logits"
    """




    if isinstance(outputs, (tuple, list)):




        if len(outputs) < 2:




            raise RuntimeError(
                "Model returned fewer than two outputs."
            )




        filament = outputs[0]




        spine = outputs[1]




        return filament, spine




    if isinstance(outputs, dict):




        filament = None




        spine = None




        filament_keys = [
            "mask",
            "filament",
            "mask_logits",
            "filament_logits",
            "out",
            "output",
        ]




        spine_keys = [
            "spine",
            "spine_logits",
        ]




        for key in filament_keys:




            if key in outputs:




                filament = outputs[key]




                break




        for key in spine_keys:




            if key in outputs:




                spine = outputs[key]




                break




        if filament is None or spine is None:




            raise RuntimeError(
                "Could not identify filament/spine outputs.\n"
                f"Model output keys: {list(outputs.keys())}"
            )




        return filament, spine




    raise RuntimeError(
        "Unsupported model output type: "
        f"{type(outputs)}"
    )








# ================================================================
# BASIC DICE
# ================================================================




def soft_dice(
    logits,
    target,
    smooth=1.0,
):




    probability = torch.sigmoid(
        logits
    )




    probability = probability.reshape(
        probability.shape[0],
        -1
    )




    target = target.reshape(
        target.shape[0],
        -1
    )




    intersection = (
        probability * target
    ).sum(dim=1)




    denominator = (
        probability.sum(dim=1)
        +
        target.sum(dim=1)
    )




    dice = (
        2.0 * intersection
        + smooth
    ) / (
        denominator
        + smooth
    )




    return dice.mean()








# ================================================================
# HARD DICE
# ================================================================




def hard_dice_from_logits(
    logits,
    target,
    threshold=0.5,
    smooth=1.0,
):




    probability = torch.sigmoid(
        logits
    )




    prediction = (
        probability >= threshold
    ).float()




    prediction = prediction.reshape(
        prediction.shape[0],
        -1
    )




    target = target.reshape(
        target.shape[0],
        -1
    )




    intersection = (
        prediction * target
    ).sum(dim=1)
   
    denominator = (
        prediction.sum(dim=1)
        +
        target.sum(dim=1)
    )




    dice = (
        2.0 * intersection
        + smooth
    ) / (
        denominator
        + smooth
    )




    return dice.mean().item()








# ================================================================
# SOFT SKELETONIZATION
# ================================================================




def soft_erode(img):




    p1 = -F.max_pool2d(
        -img,
        kernel_size=(3, 1),
        stride=1,
        padding=(1, 0),
    )




    p2 = -F.max_pool2d(
        -img,
        kernel_size=(1, 3),
        stride=1,
        padding=(0, 1),
    )




    return torch.minimum(
        p1,
        p2
    )








def soft_dilate(img):




    return F.max_pool2d(
        img,
        kernel_size=3,
        stride=1,
        padding=1,
    )








def soft_open(img):




    return soft_dilate(
        soft_erode(img)
    )








def soft_skel(
    img,
    iterations=10,
):




    img = img.float()




    img = img.clamp(
        0.0,
        1.0
    )




    skeleton = F.relu(
        img - soft_open(img)
    )




    for _ in range(
        iterations - 1
    ):




        img = soft_erode(img)




        delta = F.relu(
            img - soft_open(img)
        )




        skeleton = skeleton + (
            F.relu(
                delta - skeleton * delta
            )
        )




    return skeleton








# ================================================================
# clDice
# ================================================================




def cldice_loss(
    logits,
    target,
    iterations=10,
    smooth=1.0,
):




    prediction = torch.sigmoid(
        logits
    ).clamp(
        0.0,
        1.0
    )




    target = target.float().clamp(
        0.0,
        1.0
    )




    prediction_skeleton = soft_skel(
        prediction,
        iterations=iterations,
    )




    target_skeleton = soft_skel(
        target,
        iterations=iterations,
    )




    # Skeleton precision
    pred_skel_intersection = (
        prediction_skeleton * target
    ).sum(
        dim=(1, 2, 3)
    )




    pred_skel_size = (
        prediction_skeleton.sum(
            dim=(1, 2, 3)
        )
        + smooth
    )




    topology_precision = (
        pred_skel_intersection
        + smooth
    ) / pred_skel_size




    # Skeleton sensitivity
    target_skel_intersection = (
        target_skeleton * prediction
    ).sum(
        dim=(1, 2, 3)
    )




    target_skel_size = (
        target_skeleton.sum(
            dim=(1, 2, 3)
        )
        + smooth
    )




    topology_sensitivity = (
        target_skel_intersection
        + smooth
    ) / target_skel_size




    cl_dice = (
        2.0
        * topology_precision
        * topology_sensitivity
        /
        (
            topology_precision
            +
            topology_sensitivity
            + smooth
        )
    )




    return 1.0 - cl_dice.mean()








# ================================================================
# COMBINED LOSS
# ================================================================




class ThinFilamentLoss(nn.Module):




    def __init__(self):




        super().__init__()




        self.bce = nn.BCEWithLogitsLoss()




    def forward(
        self,
        filament_logits,
        spine_logits,
        filament_target,
        spine_target,
    ):




        # --------------------------------------------------------
        # Filament BCE
        # --------------------------------------------------------




        filament_bce = self.bce(
            filament_logits,
            filament_target
        )




        # --------------------------------------------------------
        # Filament Dice
        # --------------------------------------------------------




        filament_dice = 1.0 - soft_dice(
            filament_logits,
            filament_target
        )




        # --------------------------------------------------------
        # clDice
        # --------------------------------------------------------




        topology = cldice_loss(
            filament_logits,
            filament_target,
            iterations=CLDICE_ITERATIONS,
            smooth=CLDICE_SMOOTH,
        )




        # --------------------------------------------------------
        # Spine BCE
        # --------------------------------------------------------




        spine_bce = self.bce(
            spine_logits,
            spine_target
        )




        # --------------------------------------------------------
        # Spine Dice
        # --------------------------------------------------------




        spine_dice = 1.0 - soft_dice(
            spine_logits,
            spine_target
        )




        # --------------------------------------------------------
        # Total
        # --------------------------------------------------------




        total = (




            FILAMENT_BCE_WEIGHT
            * filament_bce




            +




            FILAMENT_DICE_WEIGHT
            * filament_dice




            +




            CLDICE_WEIGHT
            * topology




            +




            SPINE_BCE_WEIGHT
            * spine_bce




            +




            SPINE_DICE_WEIGHT
            * spine_dice
        )




        components = {
            "filament_bce": filament_bce.item(),
            "filament_dice": filament_dice.item(),
            "cldice": topology.item(),
            "spine_bce": spine_bce.item(),
            "spine_dice": spine_dice.item(),
        }




        return total, components








# ================================================================
# CHECKPOINT LOADING
# ================================================================




def load_checkpoint(
    model,
    checkpoint_path,
    device,
):




    if checkpoint_path is None:




        print()
        print(
            "No checkpoint specified."
        )




        return 0




    checkpoint_path = Path(
        checkpoint_path
    )




    if not checkpoint_path.exists():




        print()
        print(
            "Checkpoint not found:"
        )




        print(
            checkpoint_path
        )




        print()
        print(
            "Training from scratch."
        )




        return 0




    print()
    print("=" * 70)
    print("LOADING PREVIOUS CHECKPOINT")
    print("=" * 70)




    print(
        checkpoint_path
    )




    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )




    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:




        state_dict = checkpoint[
            "model_state_dict"
        ]




        epoch = checkpoint.get(
            "epoch",
            0
        )




        saved_dice = checkpoint.get(
            "best_dice",
            checkpoint.get("best_val_dice", checkpoint.get("dice", None))
        )




    else:




        state_dict = checkpoint




        epoch = 0




        saved_dice = None




    # Fail safely instead of silently accepting incomplete model weights.
    missing, unexpected = model.load_state_dict(state_dict, strict=True)




    if missing:




        print()
        print(
            "WARNING: Missing model keys:"
        )




        for key in missing[:20]:




            print(
                " ",
                key
            )




    if unexpected:




        print()
        print(
            "WARNING: Unexpected checkpoint keys:"
        )




        for key in unexpected[:20]:




            print(
                " ",
                key
            )




    print()
    print(
        "Checkpoint epoch:",
        epoch
    )




    if saved_dice is not None:




        print(
            "Saved Dice:",
            f"{float(saved_dice):.6f}"
        )




    print(
        "Checkpoint loaded successfully."
    )




    return int(epoch)








# ================================================================
# TRAIN ONE EPOCH
# ================================================================




def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    epoch,
):




    model.train()




    running_loss = 0.0




    running_filament_bce = 0.0




    running_filament_dice = 0.0




    running_cldice = 0.0




    running_spine_bce = 0.0




    running_spine_dice = 0.0




    optimizer.zero_grad(
        set_to_none=True
    )




    total_batches = len(loader)




    start_time = time.time()




    for batch_index, batch in enumerate(
        loader,
        start=1
    ):




        images = batch[
            "image"
        ].to(
            device,
            non_blocking=True
        )




        masks = batch[
            "mask"
        ].to(
            device,
            non_blocking=True
        )




        spines = batch[
            "spine"
        ].to(
            device,
            non_blocking=True
        )




        with torch.amp.autocast(
            device_type="cuda",
            enabled=(
                USE_AMP
                and device.type == "cuda"
            ),
        ):




            outputs = model(
                images
            )




            filament_logits, spine_logits = (
                extract_outputs(outputs)
            )




            total_loss, components = criterion(
                filament_logits,
                spine_logits,
                masks,
                spines,
            )




            loss_for_backward = (
                total_loss
                /
                ACCUMULATION_STEPS
            )




        if scaler.is_enabled():




            scaler.scale(
                loss_for_backward
            ).backward()




        else:




            loss_for_backward.backward()




        should_step = (
            batch_index % ACCUMULATION_STEPS == 0
            or
            batch_index == total_batches
        )




        if should_step:




            if scaler.is_enabled():




                scaler.unscale_(
                    optimizer
                )




            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP
            )




            if scaler.is_enabled():




                scaler.step(
                    optimizer
                )




                scaler.update()




            else:




                optimizer.step()




            optimizer.zero_grad(
                set_to_none=True
            )




        running_loss += (
            total_loss.item()
        )




        running_filament_bce += (
            components["filament_bce"]
        )




        running_filament_dice += (
            components["filament_dice"]
        )




        running_cldice += (
            components["cldice"]
        )




        running_spine_bce += (
            components["spine_bce"]
        )




        running_spine_dice += (
            components["spine_dice"]
        )




        if (
            batch_index == 1
            or
            batch_index % 20 == 0
            or
            batch_index == total_batches
        ):




            print(
                f"\rBatch "
                f"{batch_index:4d}/"
                f"{total_batches:4d}"
                f" | Loss "
                f"{total_loss.item():.4f}",
                end="",
                flush=True,
            )




    print()




    elapsed = (
        time.time()
        - start_time
    )




    count = float(total_batches)




    return {
        "loss":
            running_loss / count,




        "filament_bce":
            running_filament_bce / count,




        "filament_dice_loss":
            running_filament_dice / count,




        "cldice_loss":
            running_cldice / count,




        "spine_bce":
            running_spine_bce / count,




        "spine_dice_loss":
            running_spine_dice / count,




        "time":
            elapsed,
    }








# ================================================================
# VALIDATION
# ================================================================




@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):




    model.eval()




    total_loss = 0.0




    total_filament_dice = 0.0
    total_filament_cldice = 0.0
    total_filament_iou = 0.0




    total_filament_precision = 0.0




    total_filament_recall = 0.0




    total_spine_dice = 0.0




    total_spine_iou = 0.0




    batches = 0
    image_count = 0
    start_time = time.time()




    for batch in loader:




        images = batch[
            "image"
        ].to(
            device,
            non_blocking=True
        )




        masks = batch[
            "mask"
        ].to(
            device,
            non_blocking=True
        )




        spines = batch[
            "spine"
        ].to(
            device,
            non_blocking=True
        )




        outputs = model(
            images
        )




        filament_logits, spine_logits = (
            extract_outputs(outputs)
        )




        loss, _ = criterion(
            filament_logits,
            spine_logits,
            masks,
            spines,
        )




        total_loss += loss.item()




        # --------------------------------------------------------
        # Filament metrics
        # --------------------------------------------------------




        filament_probability = torch.sigmoid(
            filament_logits
        )




        filament_prediction = (
            filament_probability >= 0.5
        ).float()




        tp = (
            filament_prediction
            *
            masks
        ).sum(
            dim=(1, 2, 3)
        )




        fp = (
            filament_prediction
            *
            (1.0 - masks)
        ).sum(
            dim=(1, 2, 3)
        )




        fn = (
            (1.0 - filament_prediction)
            *
            masks
        ).sum(
            dim=(1, 2, 3)
        )




        dice = (
            2.0 * tp + 1.0
        ) / (
            2.0 * tp
            + fp
            + fn
            + 1.0
        )




        iou = (
            tp + 1.0
        ) / (
            tp
            + fp
            + fn
            + 1.0
        )




        precision = (
            tp + 1.0
        ) / (
            tp
            + fp
            + 1.0
        )




        recall = (
            tp + 1.0
        ) / (
            tp
            + fn
            + 1.0
        )




        # --------------------------------------------------------
        # Hard clDice metric (threshold 0.5; mean per image)
        # --------------------------------------------------------
        pred_skeleton = soft_skel(
            filament_prediction, iterations=CLDICE_ITERATIONS
        )
        target_skeleton = soft_skel(
            masks, iterations=CLDICE_ITERATIONS
        )
        metric_smooth = 1e-6
        skel_precision = (
            (pred_skeleton * masks).sum(dim=(1, 2, 3)) + metric_smooth
        ) / (pred_skeleton.sum(dim=(1, 2, 3)) + metric_smooth)
        skel_sensitivity = (
            (target_skeleton * filament_prediction).sum(dim=(1, 2, 3)) + metric_smooth
        ) / (target_skeleton.sum(dim=(1, 2, 3)) + metric_smooth)
        hard_cldice = (
            2.0 * skel_precision * skel_sensitivity
            / (skel_precision + skel_sensitivity + metric_smooth)
        )


        # --------------------------------------------------------
        # Spine metrics
        # --------------------------------------------------------




        spine_probability = torch.sigmoid(
            spine_logits
        )




        spine_prediction = (
            spine_probability >= 0.5
        ).float()




        spine_tp = (
            spine_prediction
            *
            spines
        ).sum(
            dim=(1, 2, 3)
        )




        spine_fp = (
            spine_prediction
            *
            (1.0 - spines)
        ).sum(
            dim=(1, 2, 3)
        )




        spine_fn = (
            (1.0 - spine_prediction)
            *
            spines
        ).sum(
            dim=(1, 2, 3)
        )




        spine_dice = (
            2.0 * spine_tp + 1.0
        ) / (
            2.0 * spine_tp
            + spine_fp
            + spine_fn
            + 1.0
        )




        spine_iou = (
            spine_tp + 1.0
        ) / (
            spine_tp
            + spine_fp
            + spine_fn
            + 1.0
        )




        total_filament_dice += dice.sum().item()
        total_filament_cldice += hard_cldice.sum().item()




        total_filament_iou += (
            iou.mean().item()
        )




        total_filament_precision += (
            precision.mean().item()
        )




        total_filament_recall += (
            recall.mean().item()
        )




        total_spine_dice += (
            spine_dice.mean().item()
        )




        total_spine_iou += (
            spine_iou.mean().item()
        )




        batches += 1
        image_count += int(masks.shape[0])
    elapsed = (time.time() - start_time)




    return {
        "loss":
            total_loss / batches,




        "filament_dice": total_filament_dice / image_count,
        "filament_cldice": total_filament_cldice / image_count,
        "filament_iou": total_filament_iou / image_count,




        "precision":
            total_filament_precision / batches,




        "recall":
            total_filament_recall / batches,




        "spine_dice":
            total_spine_dice / batches,




        "spine_iou":
            total_spine_iou / batches,




        "time":
            elapsed,
    }








# ================================================================
# SAVE CHECKPOINT
# ================================================================




def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_dice,
    validation_metrics,
):




    checkpoint = {




        "epoch":
            epoch,




        "model_state_dict":
            model.state_dict(),




        "optimizer_state_dict":
            optimizer.state_dict(),




        "scheduler_state_dict":
            scheduler.state_dict(),




        "scaler_state_dict":
            scaler.state_dict(),




        "best_dice":
            best_dice,




        "validation_metrics":
            validation_metrics,




        "image_size":
            IMAGE_SIZE,




        "seed":
            SEED,




        "batch_size":
            BATCH_SIZE,




        "learning_rate":
            LEARNING_RATE,




        "cldice_weight":
            CLDICE_WEIGHT,




        "cldice_iterations":
            CLDICE_ITERATIONS,
    }




    torch.save(
        checkpoint,
        path
    )








# ================================================================
# MODEL SANITY CHECK
# ================================================================




def model_sanity_check(
    model,
    loader,
    device,
):




    print()
    print("=" * 70)
    print("MODEL SANITY CHECK")
    print("=" * 70)




    batch = next(
        iter(loader)
    )




    images = batch[
        "image"
    ].to(device)




    masks = batch[
        "mask"
    ].to(device)




    spines = batch[
        "spine"
    ].to(device)




    with torch.no_grad():




        outputs = model(
            images
        )




    filament_logits, spine_logits = (
        extract_outputs(outputs)
    )




    print(
        "Input:",
        tuple(images.shape)
    )




    print(
        "Target mask:",
        tuple(masks.shape)
    )




    print(
        "Target spine:",
        tuple(spines.shape)
    )




    print(
        "Filament output:",
        tuple(filament_logits.shape)
    )




    print(
        "Spine output:",
        tuple(spine_logits.shape)
    )




    if filament_logits.shape != masks.shape:




        raise RuntimeError(
            "Filament output shape does not match target."
        )




    if spine_logits.shape != spines.shape:




        raise RuntimeError(
            "Spine output shape does not match target."
        )




    print()
    print("MODEL SANITY CHECK PASSED")








# ================================================================
# CSV
# ================================================================




def initialize_csv():




    if CSV_LOG.exists():




        return




    with open(
        CSV_LOG,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:




        writer = csv.writer(f)




        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_filament_bce",
                "train_filament_dice_loss",
                "train_cldice_loss",
                "train_spine_bce",
                "train_spine_dice_loss",
                "val_loss",
                "filament_dice",
                "filament_cldice",
                "filament_iou",
                "precision",
                "recall",
                "spine_dice",
                "spine_iou",
                "learning_rate",
                "epoch_time",
                "validation_time",
            ]
        )








def append_csv(
    epoch,
    train_metrics,
    val_metrics,
    learning_rate,
):




    with open(
        CSV_LOG,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:




        writer = csv.writer(f)




        writer.writerow(
            [
                epoch,




                train_metrics["loss"],




                train_metrics[
                    "filament_bce"
                ],




                train_metrics[
                    "filament_dice_loss"
                ],




                train_metrics[
                    "cldice_loss"
                ],




                train_metrics[
                    "spine_bce"
                ],




                train_metrics[
                    "spine_dice_loss"
                ],




                val_metrics["loss"],




                val_metrics[
                    "filament_dice"
                ],
                val_metrics["filament_cldice"],
                val_metrics[
                    "filament_iou"
                ],




                val_metrics[
                    "precision"
                ],




                val_metrics[
                    "recall"
                ],




                val_metrics[
                    "spine_dice"
                ],




                val_metrics[
                    "spine_iou"
                ],




                learning_rate,




                train_metrics[
                    "time"
                ],




                val_metrics[
                    "time"
                ],
            ]
        )








# ================================================================
# MAIN
# ================================================================




def main():




    seed_everything(
        SEED
    )




    print()
    print("=" * 70)
    print("512x512 UNET++ + SPINE + clDice")
    print("IMPROVED THIN-FILAMENT TRAINING")
    print("=" * 70)




    print(
        "Project root:",
        PROJECT_ROOT
    )




    print(
        "Image directory:",
        IMAGE_DIR
    )




    print(
        "Mask directory:",
        MASK_DIR
    )




    print(
        "Spine directory:",
        SPINE_DIR
    )




    print("=" * 70)




    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------




    device = get_device()




    # ------------------------------------------------------------
    # Dataset discovery
    # ------------------------------------------------------------




    records = discover_dataset()




    print()
    print("=" * 70)
    print("DATASET SPLIT")
    print("=" * 70)




    train_records, val_records = split_records(
        records
    )




    print(
        "Total images:",
        len(records)
    )




    print(
        "Training images:",
        len(train_records)
    )




    print(
        "Validation images:",
        len(val_records)
    )




    print(
        "Image size:",
        f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"
    )




    print(
        "Batch size:",
        BATCH_SIZE
    )




    # ------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------




    train_dataset = FilamentSpineDataset(
        train_records,
        image_size=IMAGE_SIZE,
        augment=USE_AUGMENTATION,
    )




    val_dataset = FilamentSpineDataset(
        val_records,
        image_size=IMAGE_SIZE,
        augment=False,
    )




    sanity_check(
        train_dataset
    )




    # ------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------




    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )




    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )




    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------




    print()
    print("=" * 70)
    print("CREATING MODEL")
    print("=" * 70)




    model = UNetPPSpine(
        in_channels=1,
        base_channels=16,
    )




    model = model.to(
        device
    )




    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )




    print(
        "Trainable parameters:",
        f"{parameter_count:,}"
    )




    # ------------------------------------------------------------
    # Model sanity check
    # ------------------------------------------------------------




    model_sanity_check(
        model,
        train_loader,
        device,
    )




    # ------------------------------------------------------------
    # Load previous model
    # ------------------------------------------------------------




    loaded_epoch = 0




    if RESUME_CHECKPOINT is not None:




        loaded_epoch = load_checkpoint(
            model,
            RESUME_CHECKPOINT,
            device,
        )




    # ------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------




    criterion = ThinFilamentLoss()




    # ------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------




    actual_learning_rate = LEARNING_RATE




    if (
        RESUME_CHECKPOINT is not None
        and
        Path(
            RESUME_CHECKPOINT
        ).exists()
    ):




        actual_learning_rate = (
            RESUME_LEARNING_RATE
        )




    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=actual_learning_rate,
        weight_decay=WEIGHT_DECAY,
    )




    # ------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------




    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6,
    )




    # ------------------------------------------------------------
    # AMP
    # ------------------------------------------------------------




    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            USE_AMP
            and device.type == "cuda"
        ),
    )




    # ------------------------------------------------------------
    # Training information
    # ------------------------------------------------------------




    print()
    print("=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)




    print(
        "Image size:",
        f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"
    )




    print(
        "Batch size:",
        BATCH_SIZE
    )




    print(
        "Gradient accumulation:",
        ACCUMULATION_STEPS
    )




    print(
        "Effective batch size:",
        BATCH_SIZE * ACCUMULATION_STEPS
    )




    print(
        "Epochs:",
        EPOCHS
    )




    print(
        "Learning rate:",
        actual_learning_rate
    )




    print(
        "Weight decay:",
        WEIGHT_DECAY
    )




    print(
        "Filament BCE weight:",
        FILAMENT_BCE_WEIGHT
    )




    print(
        "Filament Dice weight:",
        FILAMENT_DICE_WEIGHT
    )




    print(
        "clDice weight:",
        CLDICE_WEIGHT
    )




    print(
        "Spine BCE weight:",
        SPINE_BCE_WEIGHT
    )




    print(
        "Spine Dice weight:",
        SPINE_DICE_WEIGHT
    )




    print(
        "clDice iterations:",
        CLDICE_ITERATIONS
    )




    print(
        "Augmentation:",
        USE_AUGMENTATION
    )




    print(
        "AMP:",
        USE_AMP
    )




    print()
    print(
        "Resume checkpoint:",
        RESUME_CHECKPOINT
    )




    # ------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------




    initialize_csv()




    # ------------------------------------------------------------
    # Best metric
    # ------------------------------------------------------------




    best_dice = 0.0
    best_cldice = 0.0
    best_epoch = 0
    best_cldice_epoch = 0




    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------




    for epoch in range(
        1,
        EPOCHS + 1
    ):




        epoch_start = time.time()




        print()
        print("-" * 70)




        print(
            f"Epoch {epoch:03d}/{EPOCHS:02d}"
        )




        print("-" * 70)




        # --------------------------------------------------------
        # Train
        # --------------------------------------------------------




        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            epoch,
        )




        # --------------------------------------------------------
        # Validation
        # --------------------------------------------------------




        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
        )




        # --------------------------------------------------------
        # Scheduler
        # --------------------------------------------------------




        scheduler.step()




        current_lr = (
            optimizer.param_groups[0]["lr"]
        )




        # --------------------------------------------------------
        # Epoch time
        # --------------------------------------------------------




        epoch_time = (
            time.time()
            - epoch_start
        )




        print()
        print(
            "Train loss:",
            f"{train_metrics['loss']:.6f}"
        )




        print(
            "Train filament BCE:",
            f"{train_metrics['filament_bce']:.6f}"
        )




        print(
            "Train filament Dice loss:",
            f"{train_metrics['filament_dice_loss']:.6f}"
        )




        print(
            "Train clDice loss:",
            f"{train_metrics['cldice_loss']:.6f}"
        )




        print(
            "Train spine loss:",
            f"{train_metrics['spine_dice_loss']:.6f}"
        )




        print(
            "Validation loss:",
            f"{val_metrics['loss']:.6f}"
        )




        print(
            "Filament Dice:",
            f"{val_metrics['filament_dice']:.6f}"
        )
        print("Validation hard clDice:", f"{val_metrics['filament_cldice']:.6f}")




        print(
            "Filament IoU:",
            f"{val_metrics['filament_iou']:.6f}"
        )




        print(
            "Precision:",
            f"{val_metrics['precision']:.6f}"
        )




        print(
            "Recall:",
            f"{val_metrics['recall']:.6f}"
        )




        print(
            "Spine Dice:",
            f"{val_metrics['spine_dice']:.6f}"
        )




        print(
            "Spine IoU:",
            f"{val_metrics['spine_iou']:.6f}"
        )




        print(
            "LR:",
            f"{current_lr:.8f}"
        )




        print(
            "Epoch time:",
            f"{epoch_time:.1f}s"
        )




        print(
            "Validation time:",
            f"{val_metrics['time']:.2f}s"
        )




        # --------------------------------------------------------
        # CSV
        # --------------------------------------------------------




        append_csv(
            epoch,
            train_metrics,
            val_metrics,
            current_lr,
        )




        # --------------------------------------------------------
        # Last checkpoint
        # --------------------------------------------------------




        save_checkpoint(
            LAST_CHECKPOINT,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_dice,
            val_metrics,
        )




        # --------------------------------------------------------
        # Best checkpoint
        # --------------------------------------------------------




        current_dice = (
            val_metrics["filament_dice"]
        )




        if current_dice > best_dice:




            best_dice = current_dice




            best_epoch = epoch




            save_checkpoint(
                BEST_CHECKPOINT,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_dice,
                val_metrics,
            )




            print()
            print(
                "*** NEW BEST MODEL ***"
            )




            print(
                "Dice:",
                f"{best_dice:.6f}"
            )




            print(
                "IoU:",
                f"{val_metrics['filament_iou']:.6f}"
            )




            print(
                "Saved:",
                BEST_CHECKPOINT
            )




   
        current_cldice = val_metrics["filament_cldice"]
        if current_cldice > best_cldice:
            best_cldice = current_cldice
            best_cldice_epoch = epoch
            save_checkpoint(
                BEST_CLDICE_CHECKPOINT, model, optimizer, scheduler,
                scaler, epoch, best_dice, val_metrics,
            )
            print("*** NEW BEST clDice MODEL ***")
            print("clDice:", f"{best_cldice:.6f}")
            print("Saved:", BEST_CLDICE_CHECKPOINT)
# ------------------------------------------------------------
    # Complete
    # ------------------------------------------------------------




    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)




    print(
        "Best validation Dice:",
        f"{best_dice:.6f}"
    )




    print("Best Dice epoch:", best_epoch)
    print("Best validation clDice:", f"{best_cldice:.6f}")
    print("Best clDice epoch:", best_cldice_epoch)




    print()
    print(
        "Best checkpoint:"
    )




    print(BEST_CHECKPOINT)
    print("Best clDice checkpoint:")
    print(BEST_CLDICE_CHECKPOINT)




    print()
    print(
        "Last checkpoint:"
    )




    print(
        LAST_CHECKPOINT
    )




    print()
    print(
        "Training CSV:"
    )




    print(
        CSV_LOG
    )




    print("=" * 70)








# ================================================================
# ENTRY POINT
# ================================================================




if __name__ == "__main__":

    main()
