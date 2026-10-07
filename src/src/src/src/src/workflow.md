# Workflow

## 1. Main Experiment

1. Recruit 40 industrial design students, 20 per group.
2. Random assignment to Text-Only or Sketch-Loop.
3. Complete 5-minute sketch warm-up.
4. Text-Only: SDXL text-to-image, max 5 iterations.
5. Sketch-Loop: sketch → photograph → CLIP Interrogator → img2img with ControlNet Canny, max 5 iterations.
6. Record screen, time each phase, count iterations.
7. Collect NASA-TLX.
8. Three experts rate quality.

## 2. Ablation

- L0: Full Sketch-Loop.
- L1: Text-to-Image + Img2Img without ControlNet.
- L2: Sketch + Text-to-Image.
- L3: Text only.
- L4: No-AI manual control.

Each level: 10 independent participants. Measure ideation time, quality, NASA-TLX.

## 3. Automated Sketch Capture

- Text-Only: keyboard entry.
- Manual Sketch-Loop: paper sketch → photo → upload.
- Auto Sketch-Loop: tablet drawing with automatic import.

N = 40. Measure net ideation time.

## 4. Multi-Task Validation

- N = 80, 40 per group.
- Tasks: poster, UI layout, product concept, architectural massing.
- Latin square counterbalancing.
- Linear mixed-effects model.

## 5. CLIP Interrogator Validation

- 40 sketches.
- 3 raters.
- Metrics: object, attribute, spatial accuracy, usability, manual correction.

## 6. Professional Designer Rating

- 5 professional designers.
- Blinded, randomized order.
- Dimensions: creativity, visual impact, task fit, spatial composition, concept clarity.

## 7. Statistical Analysis

- Bootstrap 10,000 samples.
- Cohen's d.
- FDR for NASA-TLX subscales.
- Software: Pingouin, SciPy.

## 8. Figures

- Fig. 1–6 generated at 600 DPI, 16:9.
