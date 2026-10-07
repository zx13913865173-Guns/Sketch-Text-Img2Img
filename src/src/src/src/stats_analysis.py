import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_sd = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_sd

def bootstrap_ci(x, y, n_boot=10000, alpha=0.05):
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        diffs.append(np.mean(xb) - np.mean(yb))
    lower = np.percentile(diffs, 100*alpha/2)
    upper = np.percentile(diffs, 100*(1-alpha/2))
    return lower, upper

def analyze_main():
    df = pd.read_csv("data/main_experiment.csv")
    text = df[df["group"] == "Text-Only"]
    sketch = df[df["group"] == "Sketch-Loop"]
    results = []
    for col in ["early_ideation", "physical_operation", "net_ideation",
                "exploration", "detail_refinement", "total_time",
                "iterations", "quality", "nasa_tlx"]:
        x = text[col].dropna().values
        y = sketch[col].dropna().values
        t, p = stats.ttest_ind(x, y, equal_var=False)
        d = cohens_d(x, y)
        ci_low, ci_high = bootstrap_ci(x, y)
        results.append({
            "metric": col,
            "text_mean": np.mean(x),
            "sketch_mean": np.mean(y),
            "delta": np.mean(y) - np.mean(x),
            "p_value": p,
            "cohens_d": d,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
        })
    out = pd.DataFrame(results)
    out.to_csv("results/main_analysis.csv", index=False)
    print(out)
    return out

def analyze_nasa_tlx():
    df = pd.read_csv("data/main_experiment.csv")
    # Example subscale analysis with FDR
    subscales = ["mental", "effort", "frustration", "performance"]
    pvals = []
    for s in subscales:
        text = df[df["group"] == "Text-Only"][s]
        sketch = df[df["group"] == "Sketch-Loop"][s]
        t, p = stats.ttest_ind(text, sketch, equal_var=False)
        pvals.append(p)
    _, qvals = pg.multicomp(pvals, method="fdr_bh")
    return qvals

def main():
    analyze_main()
    print("Statistical analysis complete.")

if __name__ == "__main__":
    main()
