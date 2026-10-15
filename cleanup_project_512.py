from pathlib import Path
import json
import hashlib
from datetime import datetime


# ============================================================
# CLEANUP PROJECT 512
# SAFE AUDIT ONLY — NOTHING IS DELETED
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

REPORT_TXT = PROJECT_ROOT / "cleanup_report_512.txt"
REPORT_JSON = PROJECT_ROOT / "cleanup_report_512.json"


# ============================================================
# CLASSIFICATION
# ============================================================

KEEP_EXACT_FILES = {
    "benchmark_setup_512.py",
    "train_benchmark_unetpp_512.py",
    "unetpp_spine.py",
}

KEEP_EXACT_DIRS = {
    "benchmark_512",
    "benchmark_models_512",
    "data",
    "MAGFiLO_1.0_Kaggle_2026",
}


IMPORTANT_OLD_FILES = {
    "train_unetpp_spine_512_cldice_improved.py",
    "train_unetpp_spine_512_improved.py",
    "train_unetpp_spine_cldice.py",
    "train_unetpp_spine_topology.py",
    "train_unetpp_spine.py",
    "loss_unetpp_spine.py",
    "loss_unetpp_spine_cldice.py",
    "loss_unetpp_spine_topology.py",
    "evaluate_final_512.py",
    "evaluate_unetpp_spine.py",
    "compare_checkpoints.py",
    "tune_threshold.py",
    "check_full_dataset.py",
    "check_raw_dataset.py",
    "check_unetpp_spine_dataset.py",
    "inspect_annotations.py",
    "create_full_masks.py",
}


OLD_CHECKPOINT_DIRS = {
    "checkpoints",
    "checkpoints_512",
    "checkpoints_512_cldice_improved",
    "checkpoints_512_improved",
    "checkpoints_cldice",
    "checkpoints_topology",
    "checkpoint_comparison",
}


OLD_RESULT_DIRS = {
    "evaluation_results",
    "final_evaluation_512",
    "threshold_tuning",
    "threshold_tuning_512",
    "threshold_tuning_512_improved",
}


# ============================================================
# HELPERS
# ============================================================

def human_size(size):
    units = ["B", "KB", "MB", "GB", "TB"]

    value = float(size)

    for unit in units:
        if value < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{value:.2f} PB"


def file_hash(path, chunk_size=1024 * 1024):
    """
    SHA256 hash for duplicate detection.
    """
    sha = hashlib.sha256()

    try:
        with path.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)

                if not chunk:
                    break

                sha.update(chunk)

        return sha.hexdigest()

    except Exception:
        return None


def classify_path(path):
    """
    Classify a file/folder without modifying anything.
    """

    name = path.name

    if path.is_dir():

        if name in KEEP_EXACT_DIRS:
            return "KEEP - CURRENT BENCHMARK/DATA"

        if name in OLD_CHECKPOINT_DIRS:
            return "REVIEW - OLD CHECKPOINTS"

        if name in OLD_RESULT_DIRS:
            return "REVIEW - OLD RESULTS"

        if name.startswith("."):
            return "KEEP - ENVIRONMENT"

        return "REVIEW - UNKNOWN DIRECTORY"

    if path.is_file():

        if name in KEEP_EXACT_FILES:
            return "KEEP - CURRENT BENCHMARK"

        if name in IMPORTANT_OLD_FILES:
            return "KEEP FOR REPRODUCIBILITY"

        if name.startswith("train_benchmark_"):
            return "KEEP - BENCHMARK TRAINING SCRIPT"

        if name.startswith("evaluate_"):
            return "REVIEW - EVALUATION SCRIPT"

        if name.startswith("loss_"):
            return "KEEP FOR POSSIBLE MODEL"

        if name.endswith(".pth"):
            return "REVIEW - CHECKPOINT"

        if name.endswith(".json"):
            return "REVIEW - JSON/RESULT"

        if name.endswith(".csv"):
            return "REVIEW - CSV/RESULT"

        if name.endswith(".txt"):
            return "REVIEW - TEXT"

        if name.endswith(".py"):
            return "REVIEW - PYTHON SCRIPT"

        if name.endswith(".whl"):
            return "KEEP - PYTORCH PACKAGE"

        return "REVIEW - UNKNOWN FILE"

    return "UNKNOWN"


def collect_children(root):
    """
    Only scan the project root.

    We deliberately do NOT recursively inspect the dataset itself,
    because MAGFiLO and data may contain thousands of files.
    """

    items = []

    try:
        children = sorted(
            root.iterdir(),
            key=lambda p: (not p.is_dir(), p.name.lower())
        )
    except Exception as e:
        print(f"Could not read project directory: {e}")
        return items

    for path in children:

        try:

            if path.is_dir():

                try:
                    count = sum(1 for _ in path.iterdir())
                except Exception:
                    count = None

                items.append({
                    "name": path.name,
                    "type": "directory",
                    "path": str(path),
                    "size_bytes": None,
                    "size": "DIRECTORY",
                    "child_count": count,
                    "classification": classify_path(path),
                })

            elif path.is_file():

                size = path.stat().st_size

                items.append({
                    "name": path.name,
                    "type": "file",
                    "path": str(path),
                    "size_bytes": size,
                    "size": human_size(size),
                    "child_count": None,
                    "classification": classify_path(path),
                })

        except Exception as e:

            items.append({
                "name": path.name,
                "type": "unknown",
                "path": str(path),
                "size_bytes": None,
                "size": "ERROR",
                "child_count": None,
                "classification": f"ERROR: {e}",
            })

    return items


# ============================================================
# SPECIAL CHECKS
# ============================================================

def check_critical_items(items):

    names = {item["name"] for item in items}

    checks = {}

    checks["benchmark_setup_512.py"] = (
        "benchmark_setup_512.py" in names
    )

    checks["benchmark_512"] = (
        "benchmark_512" in names
    )

    checks["benchmark_models_512"] = (
        "benchmark_models_512" in names
    )

    checks["train_benchmark_unetpp_512.py"] = (
        "train_benchmark_unetpp_512.py" in names
    )

    checks["unetpp_spine.py"] = (
        "unetpp_spine.py" in names
    )

    checks["data"] = (
        "data" in names
    )

    checks["MAGFiLO_1.0_Kaggle_2026"] = (
        "MAGFiLO_1.0_Kaggle_2026" in names
    )

    checks["old_cldice_training"] = (
        "train_unetpp_spine_512_cldice_improved.py" in names
    )

    checks["final_evaluation"] = (
        "evaluate_final_512.py" in names
    )

    return checks


# ============================================================
# WRITE TEXT REPORT
# ============================================================

def write_text_report(items, checks):

    lines = []

    lines.append("=" * 78)
    lines.append("FILAMENT SEGMENTATION 512 PROJECT CLEANUP AUDIT")
    lines.append("=" * 78)
    lines.append("")
    lines.append("IMPORTANT:")
    lines.append("This script ONLY AUDITS the project.")
    lines.append("NO FILES OR DIRECTORIES WERE DELETED.")
    lines.append("")
    lines.append(f"Project root:")
    lines.append(f"  {PROJECT_ROOT}")
    lines.append("")
    lines.append(
        f"Audit time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    lines.append("")

    # --------------------------------------------------------
    # Critical files
    # --------------------------------------------------------

    lines.append("=" * 78)
    lines.append("CRITICAL PROJECT ITEMS")
    lines.append("=" * 78)

    for name, exists in checks.items():

        status = "FOUND" if exists else "NOT FOUND"

        lines.append(
            f"{status:12} {name}"
        )

    lines.append("")

    # --------------------------------------------------------
    # Keep
    # --------------------------------------------------------

    keep = [
        x for x in items
        if x["classification"].startswith("KEEP")
    ]

    lines.append("=" * 78)
    lines.append("KEEP")
    lines.append("=" * 78)

    if not keep:
        lines.append("None")

    for item in keep:

        lines.append(
            f"[KEEP] {item['name']} "
            f"({item['classification']})"
        )

    lines.append("")

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    reproducibility = [
        x for x in items
        if x["classification"] == "KEEP FOR REPRODUCIBILITY"
    ]

    lines.append("=" * 78)
    lines.append("KEEP FOR REPRODUCIBILITY")
    lines.append("=" * 78)

    if not reproducibility:
        lines.append("None")

    for item in reproducibility:

        lines.append(
            f"[REPRODUCE] {item['name']}"
        )

    lines.append("")

    # --------------------------------------------------------
    # Benchmark scripts
    # --------------------------------------------------------

    benchmark = [
        x for x in items
        if x["classification"] == "KEEP - BENCHMARK TRAINING SCRIPT"
    ]

    lines.append("=" * 78)
    lines.append("BENCHMARK TRAINING SCRIPTS")
    lines.append("=" * 78)

    if not benchmark:
        lines.append("None")

    for item in benchmark:

        lines.append(
            f"[BENCHMARK] {item['name']}"
        )

    lines.append("")

    # --------------------------------------------------------
    # Review
    # --------------------------------------------------------

    review = [
        x for x in items
        if x["classification"].startswith("REVIEW")
    ]

    lines.append("=" * 78)
    lines.append("REVIEW — DO NOT DELETE AUTOMATICALLY")
    lines.append("=" * 78)

    if not review:
        lines.append("None")

    for item in review:

        size = item.get("size", "")

        lines.append(
            f"[REVIEW] {item['name']:<55} {size}"
        )

    lines.append("")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    lines.append("=" * 78)
    lines.append("SUMMARY")
    lines.append("=" * 78)

    lines.append(
        f"Total root items scanned: {len(items)}"
    )

    lines.append(
        f"Keep items: {len(keep)}"
    )

    lines.append(
        f"Reproducibility items: {len(reproducibility)}"
    )

    lines.append(
        f"Benchmark scripts: {len(benchmark)}"
    )

    lines.append(
        f"Review items: {len(review)}"
    )

    lines.append("")

    lines.append("=" * 78)
    lines.append("RECOMMENDATION")
    lines.append("=" * 78)

    lines.append(
        "Do NOT delete anything based only on this report."
    )

    lines.append(
        "First verify that the 0.716452 UNet++ + Spine + clDice "
        "checkpoint and its training script are preserved."
    )

    lines.append(
        "Then old checkpoints/results can be archived or removed."
    )

    lines.append("")

    REPORT_TXT.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


# ============================================================
# WRITE JSON REPORT
# ============================================================

def write_json_report(items, checks):

    report = {
        "audit": {
            "project_root": str(PROJECT_ROOT),
            "timestamp": datetime.now().isoformat(),
            "deletion_performed": False,
        },

        "critical_items": checks,

        "items": items,

        "classification_counts": {},
    }

    counts = {}

    for item in items:

        category = item["classification"]

        counts[category] = counts.get(category, 0) + 1

    report["classification_counts"] = counts

    REPORT_JSON.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print("FILAMENT SEGMENTATION 512 PROJECT CLEANUP AUDIT")
    print("=" * 78)

    print()
    print("SAFE MODE")
    print("Nothing will be deleted.")
    print()

    print(f"Project root:")
    print(f"  {PROJECT_ROOT}")

    print()
    print("Scanning project root...")

    items = collect_children(PROJECT_ROOT)

    print(
        f"Found {len(items)} root files/directories."
    )

    checks = check_critical_items(items)

    write_text_report(items, checks)
    write_json_report(items, checks)

    print()
    print("=" * 78)
    print("AUDIT COMPLETE")
    print("=" * 78)

    print()
    print("Text report:")
    print(f"  {REPORT_TXT}")

    print()
    print("JSON report:")
    print(f"  {REPORT_JSON}")

    print()
    print("IMPORTANT:")
    print("NO FILES WERE DELETED.")
    print()
    print("Open cleanup_report_512.txt and review it before deleting anything.")
    print()


if __name__ == "__main__":
    main()