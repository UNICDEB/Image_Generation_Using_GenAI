import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import glob
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================
DATA_DIR = "SIP_data_images"
OUTPUT_DIR = "result_image"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 100
LR = 0.0002
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# CUSTOM DATASET
# =========================
class SaffronDataset(Dataset):
    def __init__(self, root, transform=None):
        self.files = sorted(glob.glob(os.path.join(root, "*.*")))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        if self.transform:
            img = self.transform(img)
        return img, img  # input, target (identity transform)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = SaffronDataset(DATA_DIR, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =========================
# GENERATOR (U-Net)
# =========================
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetGenerator, self).__init__()
        def down(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        def up(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        
        self.encoder = nn.Sequential(
            down(3, 64),
            down(64, 128),
            down(128, 256),
            down(256, 512)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            up(512, 512),
            up(512, 256),
            up(256, 128),
            up(128, 64),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        enc = self.encoder(x)
        mid = self.middle(enc)
        dec = self.decoder(mid)
        return dec

# =========================
# DISCRIMINATOR (PatchGAN)
# =========================
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6):
        super(PatchDiscriminator, self).__init__()
        def block(in_ch, out_ch, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(6, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            nn.Conv2d(256, 1, 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, input_img, target_img):
        x = torch.cat([input_img, target_img], dim=1)
        return self.model(x)

# =========================
# INITIALIZE MODELS
# =========================
generator = UNetGenerator().to(DEVICE)
discriminator = PatchDiscriminator().to(DEVICE)

criterion_GAN = nn.BCELoss()
criterion_L1 = nn.L1Loss()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

# =========================
# TRAINING LOOP
# =========================
print("Training Pix2Pix cGAN for Saffron Flower Augmentation...")

for epoch in range(EPOCHS):
    for i, (input_img, target_img) in enumerate(tqdm(dataloader)):
        input_img = input_img.to(DEVICE)
        target_img = target_img.to(DEVICE)

        # Generate fake image
        gen_img = generator(input_img)

        # Get discriminator output size dynamically
        pred_fake = discriminator(gen_img, input_img)
        real = torch.ones_like(pred_fake).to(DEVICE)
        fake = torch.zeros_like(pred_fake).to(DEVICE)


        # Train Generator
        optimizer_G.zero_grad()
        gen_img = generator(input_img)
        pred_fake = discriminator(gen_img, input_img)
        loss_GAN = criterion_GAN(pred_fake, real)
        loss_L1 = criterion_L1(gen_img, target_img)
        loss_G = loss_GAN + 100 * loss_L1
        loss_G.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        pred_real = discriminator(target_img, input_img)
        loss_real = criterion_GAN(pred_real, real)

        pred_fake = discriminator(gen_img.detach(), input_img)
        loss_fake = criterion_GAN(pred_fake, fake)

        loss_D = (loss_real + loss_fake) * 0.5
        loss_D.backward()
        optimizer_D.step()

    # Save example outputs
    if (epoch + 1) % 10 == 0:
        gen_img = generator(input_img)
        save_image(gen_img.data[:4], f"{OUTPUT_DIR}/epoch_{epoch+1}_gen.png", nrow=2, normalize=True)
        print(f"Saved generated images for epoch {epoch+1}")

print("Training completed!")

# =========================
# GENERATE VARIATIONS
# =========================
print("Generating 5 variations per image...")
generator.eval()

image_files = glob.glob(os.path.join(DATA_DIR, "*.*"))
for img_path in image_files:
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    image = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    for i in range(5):
        with torch.no_grad():
            generated = generator(input_tensor)
        save_path = os.path.join(OUTPUT_DIR, f"{img_name}_aug_{i+1}.png")
        save_image(generated, save_path, normalize=True)

print(f"Saved all variations in: {OUTPUT_DIR}")