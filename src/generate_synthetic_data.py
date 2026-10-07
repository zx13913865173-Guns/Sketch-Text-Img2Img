import os
import numpy as np
import pandas as pd

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def main():
    np.random.seed(42)
    ensure_dir("data")

    # Main experiment
    n = 20
    text = pd.DataFrame({
        "participant_id": [f"T{i:02d}" for i in range(1, n+1)],
        "group": "Text-Only",
        "early_ideation": np.random.normal(14.5, 3.2, n).clip(8, 22),
        "physical_operation": np.random.normal(0.5, 0.2, n).clip(0.1, 1.2),
        "exploration": np.random.normal(16.0, 4.1, n).clip(8, 26),
        "detail_refinement": np.random.normal(7.7, 2.8, n).clip(3, 15),
        "iterations": np.random.normal(4.2, 0.8, n).round().clip(1, 5),
        "quality": np.random.normal(5.8, 1.3, n).clip(3, 9),
        "nasa_tlx": np.random.normal(68.5, 14, n).clip(30, 95),
    })
    sketch = pd.DataFrame({
        "participant_id": [f"S{i:02d}" for i in range(1, n+1)],
        "group": "Sketch-Loop",
        "early_ideation": np.random.normal(8.6, 2.1, n).clip(4, 15),
        "physical_operation": np.random.normal(1.8, 0.5, n).clip(0.8, 3.2),
        "exploration": np.random.normal(16.7, 3.8, n).clip(8, 26),
        "detail_refinement": np.random.normal(10.3, 3.2, n).clip(5, 18),
        "iterations": np.random.normal(3.0, 0.9, n).round().clip(1, 5),
        "quality": np.random.normal(7.4, 1.1, n).clip(4, 10),
        "nasa_tlx": np.random.normal(54.2, 13, n).clip(20, 85),
    })
    main = pd.concat([text, sketch], ignore_index=True)
    main["total_time"] = (main["early_ideation"] + main["exploration"] +
                          main["detail_refinement"])
    main["net_ideation"] = main["early_ideation"] - main["physical_operation"]
    main.to_csv("data/main_experiment.csv", index=False)

    # Ablation
    abl = []
    for level, ideation, quality, tlx, iters in [
        ("L0", 8.6, 7.4, 54.2, 3.0),
        ("L1", 11.1, 6.4, 61.0, 3.9),
        ("L2", 10.5, 6.3, 62.1, 4.0),
        ("L3", 14.5, 5.8, 68.5, 4.2),
    ]:
        for i in range(10):
            abl.append({
                "level": level,
                "participant_id": f"{level}_{i:02d}",
                "ideation_time": np.random.normal(ideation, 1.5).clip(5, 20),
                "quality": np.random.normal(quality, 0.8).clip(3, 10),
                "nasa_tlx": np.random.normal(tlx, 8).clip(30, 90),
                "iterations": np.random.normal(iters, 0.7).round().clip(1, 5),
            })
    pd.DataFrame(abl).to_csv("data/ablation.csv", index=False)

    # Auto sketch
    auto = []
    for group, n_grp, early, phys in [
        ("Text-Only", 14, 14.6, 0.5),
        ("Manual Sketch-Loop", 13, 8.7, 1.8),
        ("Auto Sketch-Loop", 13, 5.9, 0.2),
    ]:
        for i in range(n_grp):
            auto.append({
                "group": group,
                "participant_id": f"{group[:3]}_{i:02d}",
                "early_ideation": np.random.normal(early, 2.0).clip(3, 22),
                "physical_operation": np.random.normal(phys, 0.3).clip(0.05, 2.5),
                "quality": np.random.normal(7.0 if group != "Text-Only" else 5.8, 1.0).clip(3, 10),
                "nasa_tlx": np.random.normal(55 if group != "Text-Only" else 68, 10).clip(30, 90),
            })
    auto_df = pd.DataFrame(auto)
    auto_df["net_ideation"] = auto_df["early_ideation"] - auto_df["physical_operation"]
    auto_df.to_csv("data/auto_sketch.csv", index=False)

    # Multi-task
    multi = []
    for task, text_m, auto_m in [
        ("poster", 14.6, 6.1),
        ("ui_layout", 16.9, 10.6),
        ("product_concept", 16.0, 9.8),
        ("architectural_massing", 18.3, 11.5),
    ]:
        for i in range(40):
            multi.append({
                "task": task,
                "group": "Text-Only",
                "ideation_time": np.random.normal(text_m, 2.5).clip(5, 25),
                "quality": np.random.normal(5.8, 1.2).clip(3, 10),
            })
            multi.append({
                "task": task,
                "group": "Auto Sketch-Loop",
                "ideation_time": np.random.normal(auto_m, 2.0).clip(3, 20),
                "quality": np.random.normal(7.5, 1.0).clip(4, 10),
            })
    pd.DataFrame(multi).to_csv("data/multi_task.csv", index=False)

    # CLIP validation
    clip = pd.DataFrame({
        "sketch_id": [f"SK{i:02d}" for i in range(1, 41)],
        "object_accuracy": np.random.normal(78.3, 20.8, 40).clip(20, 100),
        "attribute_accuracy": np.random.normal(64.2, 24.2, 40).clip(10, 100),
        "spatial_accuracy": np.random.normal(56.7, 25.0, 40).clip(10, 100),
        "usability": np.random.normal(3.8, 0.9, 40).clip(1, 5),
        "manual_correction": np.random.normal(32.1, 23.6, 40).clip(0, 100),
    })
    clip.to_csv("data/clip_validation.csv", index=False)

    # Professional ratings
    prof = pd.DataFrame({
        "designer_id": [f"D{i}" for i in range(1, 6)],
        "creativity_text": np.random.normal(6.1, 1.1, 5).clip(1, 10),
        "creativity_sketch": np.random.normal(7.2, 1.0, 5).clip(1, 10),
        "visual_text": np.random.normal(5.8, 1.2, 5).clip(1, 10),
        "visual_sketch": np.random.normal(7.8, 0.9, 5).clip(1, 10),
        "task_fit_text": np.random.normal(5.6, 1.2, 5).clip(1, 10),
        "task_fit_sketch": np.random.normal(6.4, 1.1, 5).clip(1, 10),
    })
    prof.to_csv("data/professional_ratings.csv", index=False)

    print("Synthetic data written to data/")

if __name__ == "__main__":
    main()
