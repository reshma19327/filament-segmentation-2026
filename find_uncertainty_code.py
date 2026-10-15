from pathlib import Path
import re

root = Path(".")
pattern = re.compile(
    r"mc.?dropout|predictive.?uncertainty|uncertainty.?map|"
    r"predictive.?entropy|test.?time.?augmentation|"
    r"\btta\b|epistemic|aleatoric",
    re.I
)

excluded = {".venv", ".git", "__pycache__", "MAGFiLO_1.0_Kaggle_2026"}

print("=== UNCERTAINTY IMPLEMENTATION SEARCH ===")
found = []

for path in root.rglob("*.py"):
    if any(part in excluded for part in path.parts):
        continue
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    except OSError:
        continue

    hits = [(i + 1, line.strip()) for i, line in enumerate(lines)
            if pattern.search(line)]

    if hits:
        found.append(path)
        print(f"\n{path}")
        for lineno, line in hits[:4]:
            print(f"  L{lineno}: {line[:150]}")

if not found:
    print("No clear existing uncertainty-analysis implementation found.")

print("\n=== RESULT ===")
print(f"Python files containing relevant terms: {len(found)}")
print("This is a source-code search, not proof that no implementation exists elsewhere.")
