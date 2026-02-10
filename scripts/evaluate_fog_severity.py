"""
Evaluate YOLOv8 model on Clear / Light / Medium / Heavy test sets.
Produces mAP@50 comparison chart and prints per-severity results.

Usage:
    python scripts/evaluate_fog_severity.py
"""
import os
import json
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = "C:/Users/yashw/cityscapes_project"
WEIGHTS = os.path.join(BASE, "yolo_runs/train_clear/weights/best.pt")
DATASET_DIR = os.path.join(BASE, "yolo_dataset")
OUT_CHART = os.path.join(BASE, "debug/model_test_results.png")
OUT_JSON = os.path.join(BASE, "debug/model_test_results.json")

CONDITIONS = ["test_clear", "test_light", "test_medium", "test_heavy"]
LABELS = ["Clear", "Light Fog", "Medium Fog", "Heavy Fog"]
COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]

def main():
    from ultralytics import YOLO

    model = YOLO(WEIGHTS)
    results_all = {}

    for cond, label in zip(CONDITIONS, LABELS):
        # Create a temporary YAML for this test condition
        test_yaml = os.path.join(DATASET_DIR, f"{cond}_eval.yaml")
        cfg = {
            "path": DATASET_DIR.replace("\\", "/"),
            "train": "train/images",
            "val": f"{cond}/images",
            "names": {0: "person", 1: "car", 2: "bicycle", 3: "motorcycle"},
        }
        with open(test_yaml, "w") as f:
            yaml.dump(cfg, f)

        print(f"\n{'='*60}")
        print(f"Evaluating on: {label} ({cond})")
        print(f"{'='*60}")

        metrics = model.val(
            data=test_yaml,
            imgsz=640,
            batch=8,
            device="cpu",
            workers=0,
            verbose=False,
        )

        map50 = float(metrics.box.map50)
        map50_95 = float(metrics.box.map)
        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)

        # Per-class mAP50
        per_class = {}
        class_names = ["person", "car", "bicycle", "motorcycle"]
        if hasattr(metrics.box, 'ap50') and metrics.box.ap50 is not None:
            for i, name in enumerate(class_names):
                if i < len(metrics.box.ap50):
                    per_class[name] = float(metrics.box.ap50[i])

        results_all[label] = {
            "condition": cond,
            "mAP50": round(map50, 4),
            "mAP50-95": round(map50_95, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "per_class_mAP50": {k: round(v, 4) for k, v in per_class.items()},
        }

        print(f"  mAP@50     = {map50:.4f}")
        print(f"  mAP@50-95  = {map50_95:.4f}")
        print(f"  Precision  = {precision:.4f}")
        print(f"  Recall     = {recall:.4f}")
        for cls_name, ap in per_class.items():
            print(f"    {cls_name}: AP50 = {ap:.4f}")

        # Clean up temp yaml
        os.remove(test_yaml)

    # Save JSON results
    with open(OUT_JSON, "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\nResults saved to {OUT_JSON}")

    # --- Generate chart ---
    map50_vals = [results_all[lbl]["mAP50"] for lbl in LABELS]
    map50_95_vals = [results_all[lbl]["mAP50-95"] for lbl in LABELS]
    prec_vals = [results_all[lbl]["precision"] for lbl in LABELS]
    rec_vals = [results_all[lbl]["recall"] for lbl in LABELS]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Chart 1: mAP@50 bar chart
    ax1 = axes[0]
    bars = ax1.bar(LABELS, map50_vals, color=COLORS, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, map50_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.set_ylabel("mAP@50", fontsize=12)
    ax1.set_title("Object Detection Performance vs Fog Severity", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, max(map50_vals) * 1.25 if max(map50_vals) > 0 else 1.0)
    ax1.grid(axis="y", alpha=0.3)

    # Chart 2: Multi-metric comparison
    ax2 = axes[1]
    x = np.arange(len(LABELS))
    w = 0.2
    ax2.bar(x - 1.5*w, map50_vals, w, label="mAP@50", color="#3498db", edgecolor="black", linewidth=0.5)
    ax2.bar(x - 0.5*w, map50_95_vals, w, label="mAP@50-95", color="#9b59b6", edgecolor="black", linewidth=0.5)
    ax2.bar(x + 0.5*w, prec_vals, w, label="Precision", color="#2ecc71", edgecolor="black", linewidth=0.5)
    ax2.bar(x + 1.5*w, rec_vals, w, label="Recall", color="#e67e22", edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(LABELS)
    ax2.set_ylabel("Score", fontsize=12)
    ax2.set_title("All Metrics vs Fog Severity", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_CHART, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {OUT_CHART}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Condition':<15} {'mAP@50':>8} {'mAP@50-95':>10} {'Precision':>10} {'Recall':>8}")
    print(f"{'-'*70}")
    for lbl in LABELS:
        r = results_all[lbl]
        print(f"{lbl:<15} {r['mAP50']:>8.4f} {r['mAP50-95']:>10.4f} {r['precision']:>10.4f} {r['recall']:>8.4f}")
    print(f"{'='*70}")

    # Degradation summary
    if map50_vals[0] > 0:
        for i in range(1, 4):
            drop = (1 - map50_vals[i] / map50_vals[0]) * 100
            print(f"  {LABELS[i]} vs Clear: {drop:+.1f}% mAP@50 drop")


if __name__ == "__main__":
    main()
