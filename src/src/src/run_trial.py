import argparse
import yaml
from PIL import Image
from generation_pipeline import SketchLoopPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["text_only", "sketch_loop"], required=True)
    parser.add_argument("--sketch", type=str, help="Path to sketch image")
    parser.add_argument("--prompt", type=str, default="eco-friendly smartwatch concept poster")
    parser.add_argument("--output", type=str, default="results/trial.png")
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    if args.condition == "sketch_loop":
        if not args.sketch:
            raise ValueError("Sketch image required for sketch_loop condition")
        sketch = Image.open(args.sketch).convert("RGB")
        pipeline = SketchLoopPipeline(config)
        image = pipeline.generate(sketch, args.prompt, seed=config["generation"]["seed"])
        image.save(args.output)
        print(f"Saved {args.output}")
    else:
        print("Text-Only baseline: use your SDXL WebUI or text-to-image pipeline.")

if __name__ == "__main__":
    main()
