from pathlib import Path
import re

files = [
    Path("unified_evaluation_clean_512.py"),
    Path("run_post_training_analysis_stronger_512.py"),
    Path("train_benchmark_unetpp_spine_only_512.py"),
]

pattern = re.compile(
    r"def .*metric|def .*mask|def __getitem__|"
    r"cv2\.resize|INTER_NEAREST|INTER_AREA|"
    r"threshold|mask_path|MASK_DIR|"
    r"tp\s*=|dice|cldice_score|"
    r"binary_cross_entropy|load_split|"
    r"Image\.open|imread|mean\(",
    re.I,
)

for path in files:
    print("\n" + "=" * 72)
    print(f"SOURCE: {path}")
    print("=" * 72)

    if not path.is_file():
        print("FILE NOT FOUND")
        continue

    lines = path.read_text(
        encoding="utf-8-sig", errors="replace"
    ).splitlines()

    selected = set()
    for i, line in enumerate(lines):
        if pattern.search(line):
            selected.update(range(max(0, i - 2), min(len(lines), i + 3)))

    last = -2
    shown = 0
    for i in sorted(selected):
        if shown >= 100:
            print("... output truncated for efficiency ...")
            break
        if i > last + 1:
            print("...")
        print(f"{i+1:4}: {lines[i]}")
        last = i
        shown += 1
