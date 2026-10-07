import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, ControlNetModel
from PIL import Image
import cv2
import numpy as np

class SketchLoopPipeline:
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.controlnet = ControlNetModel.from_pretrained(
            config["generation"]["controlnet"],
            torch_dtype=torch.float16
        )
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            config["generation"]["model"],
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(self.device)
        self.pipe.enable_xformers_memory_efficient_attention()

    def canny(self, image: Image.Image):
        img = np.array(image)
        low = self.config["generation"]["canny_low"]
        high = self.config["generation"]["canny_high"]
        edges = cv2.Canny(img, low, high)
        edges = np.stack([edges]*3, axis=-1)
        return Image.fromarray(edges)

    def generate(self, sketch_image: Image.Image, prompt: str, seed: int = 42):
        generator = torch.Generator(self.device).manual_seed(seed)
        canny_image = self.canny(sketch_image)
        image = self.pipe(
            prompt=prompt,
            image=sketch_image,
            control_image=canny_image,
            negative_prompt="low quality, blurry",
            num_inference_steps=self.config["generation"]["steps"],
            guidance_scale=self.config["generation"]["cfg_scale"],
            strength=self.config["generation"]["img2img_strength"],
            controlnet_conditioning_scale=self.config["generation"]["controlnet_conditioning_scale"],
            generator=generator,
        ).images[0]
        return image
