import os
from PIL import Image
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline

# Load SDXL Base & Refiner
device = "cuda" if torch.cuda.is_available() else "cpu"

base = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to(device)

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to(device)

# Input / Output folders
input_folder = "./SIP_data_images"
output_folder = "./result_image"
os.makedirs(output_folder, exist_ok=True)

# Prompt Engineering
# prompt = (
#     "A high-resolution close-up photograph of a fully bloomed saffron flower, "
#     "captured from different angles, showing the red stigma and style clearly for plucking, "
#     "macro photography, sharp focus, vibrant natural colors, realistic"
# )
# negative_prompt = "blurry, cropped, distorted, missing petals, low quality, watermark, text, cartoon, painting"

# prompt = (
#     "Ultra-realistic high-resolution close-up photograph of a saffron flower "
#     "in a green agricultural field, showing the full bloom with clear red stigmas "
#     "and yellow styles visible for plucking, captured from multiple angles. "
#     "Natural lighting, macro botanical photography style, extremely detailed petals, "
#     "sharp focus, vibrant natural colors, photorealistic, DSLR quality"
# )

# negative_prompt = (
#     "blurry, low quality, cartoon, painting, distorted, cropped, out of frame, "
#     "missing petals, artificial, unrealistic, duplicate flowers, watermark, text"
# )

prompt = (
    "Ultra-realistic high-resolution close-up photograph of a saffron flower "
    "growing naturally in a saffron field with rows of green saffron plants. "
    "The soil is light brown, slightly dry, and finely textured, with natural furrows "
    "typical of saffron cultivation. The flower is fully bloomed with vibrant purple petals, "
    "clear red stigmas, and yellow styles visible for plucking. Captured from multiple angles, "
    "showing both the flower and the surrounding field environment. "
    "Soft natural sunlight, macro botanical photography, sharp focus, "
    "extremely detailed petals and realistic soil texture, DSLR quality, photorealistic"
)

negative_prompt = (
    "blurry, low quality, cartoon, painting, distorted, cropped, out of frame, "
    "missing petals, artificial, unrealistic, duplicate flowers, fake soil, "
    "dark background, plastic look, watermark, text"
)



# Parameters
strength = 0.45           # how much to modify input image
guidance_scale = 9        # prompt adherence
num_steps_base = 60       # steps for base generation
num_steps_refiner = 30    # steps for refinement
num_variations = 3        # variations per image
img_size = 1024           # output resolution

# Generate Variations for Each Image
print("Generating realistic saffron flower variations...")

for file in os.listdir(input_folder):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(input_folder, file)
        init_image = Image.open(img_path).convert("RGB").resize((img_size, img_size))

        for i in range(num_variations):
            # Stage 1: Base Generation
            base_output = base(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps_base
            ).images[0]

            # Stage 2: Refinement
            refined_output = refiner(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=base_output,
                strength=0.3,  # slight change during refinement
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps_refiner
            ).images[0]

            # Save the result
            out_path = os.path.join(
                output_folder, f"{os.path.splitext(file)[0]}_var{i}.png"
            )
            refined_output.save(out_path)
            print(f"Saved: {out_path}")

print(f"\n All variations saved in: {output_folder}")
