import os
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import torch

# Load Stable Diffusion model
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Input and output folders
input_folder = "./SIP_data_images"
output_folder = "./result_image"
os.makedirs(output_folder, exist_ok=True)

# Parameters
prompt = "A close-up photo of a saffron flower, detailed, vibrant colors"
strength = 0.6
guidance_scale = 7.5
num_variations = 3   # number of variations per input image

# Process each image in folder
for file in os.listdir(input_folder):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(input_folder, file)
        init_image = Image.open(img_path).convert("RGB").resize((512, 512))

        for i in range(num_variations):
            result = pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=50
            ).images[0]

            out_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_var{i}.png")
            result.save(out_path)

print(f"All variations saved in: {output_folder}")