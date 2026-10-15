"""Read-only MAGFiLO annotation grouping audit.
Run from project root: python audit_annotation_grouping.py
Optional: python audit_annotation_grouping.py --project-root "C:\\Users\\Kevin\\Downloads\\filament-segmentation-2026"
Writes audit_reports/annotation_grouping_audit.csv and .json. Does not modify dataset/masks/checkpoints.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".fits", ".fit"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    root = parser.parse_args().project_root.expanduser().resolve()
    json_path = root / "MAGFiLO_1.0_Kaggle_2026" / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    image_dir = root / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
    out_dir = root / "audit_reports"
    if not json_path.is_file() or not image_dir.is_dir():
        print("ERROR: expected annotation JSON and train_images directory were not found.")
        print("JSON:", json_path)
        print("Images:", image_dir)
        return 2


    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    images, annotations = data["images"], data["annotations"]


    records_by_id, records_by_name = {}, defaultdict(list)
    anns_by_id, cats_by_id = Counter(), defaultdict(Counter)
    category_counts, duplicate_ids, malformed = Counter(), [], []
    for rec in images:
        if not isinstance(rec, dict) or rec.get("id") is None or not rec.get("file_name"):
            malformed.append(rec)
            continue
        iid, name = str(rec["id"]), str(rec["file_name"])
        if iid in records_by_id:
            duplicate_ids.append(iid)
        records_by_id[iid] = rec
        records_by_name[name.casefold()].append(rec)


    unknown_image_ids = Counter()
    for ann in annotations:
        if not isinstance(ann, dict) or ann.get("image_id") is None:
            unknown_image_ids["<missing image_id>"] += 1
            continue
        iid = str(ann["image_id"])
        if iid not in records_by_id:
            unknown_image_ids[iid] += 1
            continue
        anns_by_id[iid] += 1
        cat = str(ann.get("category_id", "<missing>"))
        cats_by_id[iid][cat] += 1
        category_counts[cat] += 1


    disk = {p.name.casefold(): p.name for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in EXTS}
    json_names = set(records_by_name)
    rows, repeated = [], []
    for name_key, recs in sorted(records_by_name.items()):
        ids = [str(r["id"]) for r in recs]
        per_record = [anns_by_id[i] for i in ids]
        cats = Counter()
        for iid in ids:
            cats.update(cats_by_id[iid])
        name = str(recs[0]["file_name"])
        rows.append({
            "file_name": name,
            "physical_file_found": name_key in disk,
            "physical_file_name": disk.get(name_key, ""),
            "json_record_count_for_filename": len(recs),
            "image_ids": "|".join(ids),
            "annotation_count_total": sum(per_record),
            "annotation_count_per_image_id": "|".join(map(str, per_record)),
            "category_counts": json.dumps(dict(cats), sort_keys=True),
            "record_dimensions": "|".join(f'{r.get("width","?")}x{r.get("height","?")}' for r in recs),
        })
        if len(recs) > 1:
            repeated.append({
                "file_name": name, "record_count": len(recs), "image_ids": ids,
                "annotation_count_per_record": per_record, "annotation_count_total": sum(per_record),
                "category_counts": dict(cats),
            })


    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, report_path = out_dir / "annotation_grouping_audit.csv", out_dir / "annotation_grouping_audit.json"
    fields = list(rows[0].keys()) if rows else [
        "file_name", "physical_file_found", "physical_file_name", "json_record_count_for_filename",
        "image_ids", "annotation_count_total", "annotation_count_per_image_id", "category_counts", "record_dimensions"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


    summary = {
        "annotation_json": str(json_path), "image_directory": str(image_dir),
        "json_image_record_count": len(images), "valid_json_image_records": len(records_by_id),
        "annotation_count": len(annotations), "annotation_category_counts": dict(category_counts),
        "physical_image_file_count": len(disk), "unique_json_filename_count": len(records_by_name),
        "json_filenames_matching_disk": len(json_names & set(disk)),
        "json_filenames_missing_from_disk": sorted(json_names - set(disk)),
        "disk_filenames_missing_from_json": sorted(set(disk) - json_names),
        "multiple_record_filename_count": len(repeated),
        "multiple_record_filenames": repeated,
        "malformed_image_record_count": len(malformed), "duplicate_image_ids": duplicate_ids,
        "annotations_with_unknown_image_ids": dict(unknown_image_ids),
        "csv_report": str(csv_path), "json_report": str(report_path),
        "note": "Audit only. It does not merge annotations or rebuild masks. Review repeated filenames before changing training targets."
    }
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=" * 68)
    print("MAGFiLO ANNOTATION GROUPING AUDIT — READ ONLY")
    print("=" * 68)
    print("JSON image records:", len(images))
    print("Annotation records:", len(annotations))
    print("Physical image files:", len(disk))
    print("Unique JSON filenames:", len(records_by_name))
    print("JSON filenames matching disk:", len(json_names & set(disk)))
    print("Filenames with multiple JSON records:", len(repeated))
    print("Malformed image records:", len(malformed))
    print("Duplicate image IDs:", len(duplicate_ids))
    print("Annotations with unknown image IDs:", sum(unknown_image_ids.values()))
    print("Category counts:", dict(category_counts))
    print("-" * 68)
    print("CSV:", csv_path)
    print("JSON:", report_path)
    print("No images, masks, checkpoints, or training files were modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





