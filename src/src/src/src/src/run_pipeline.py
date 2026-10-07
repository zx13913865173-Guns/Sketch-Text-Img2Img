import subprocess
import sys

def run(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    run("python src/generate_synthetic_data.py")
    run("python src/stats_analysis.py")
    run("python figures/fig1_workflow.py")
    run("python figures/fig2_time.py")
    run("python figures/fig3_quality.py")
    run("python figures/fig4_ablation.py")
    run("python figures/fig5_clip.py")
    run("python figures/fig6_multi_task.py")
    print("Pipeline complete.")
