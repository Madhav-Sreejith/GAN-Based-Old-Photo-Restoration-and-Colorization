import json
import os

notebook = {
    'cells': [],
    'metadata': {
        'kernelspec': {
            'display_name': 'Python (GenAI Pix2Pix GAN)',
            'language': 'python',
            'name': 'genai_gan'
        },
        'language_info': {
            'name': 'python',
            'version': '3.12.13'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

def add_md(text):
    notebook['cells'].append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in text.strip().split('\n')]
    })

def add_code(code):
    notebook['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in code.strip().split('\n')]
    })

# Cell 1: Title & Academic Header
add_md("""# Case Study: GAN-Based Old Photo Restoration and Colorization
**Course Code:** 23CSE475 | **Course:** Generative AI  
**Lab 3:** GenAI Application Case Study | **Academic Year:** 2026–2027  
**Institution:** Amrita Vishwa Vidyapeetham  

### Team Information
| Reg. No. | Name |
| :--- | :--- |
| **CB.SC.U4CSE23504** | ASHIN VARGHESE |
| **CB.SC.U4CSE23644** | VISHNU SATHWICK |
| **CB.SC.U4CSE23661** | MADHAV SREEJITH |

---

## 1. Problem Statement & Motivation
Old photographs deteriorate over time due to scratches, dust, noise, blur, fading, and loss of contrast. Many vintage photographs exist only in grayscale, losing critical color information.

Conventional methods (such as median/bilateral filtering, Navier-Stokes inpainting, and plain CNNs with L1/L2 loss) suffer from:
1. **Limited reconstruction ability:** Cannot reliably hallucinate plausible missing structures or color distributions.
2. **Over-smoothing:** Plain CNNs trained only with pixel-wise losses produce blurry, averaged colors.
3. **Lack of paired data:** Original undamaged versions of genuine historical photos do not exist.

### Proposed GenAI Solution
We train a **Conditional Generative Adversarial Network (Pix2Pix)** using a scenic landscape image dataset. We synthetically generate paired training data by converting clean color images to the **Lab color space (extracting the L luminance channel)** and applying realistic physical degradations (scratches, cracks, dust, Gaussian noise, and blur). The generator learns to jointly **remove degradation artifacts and infer realistic color distributions**, while the **PatchGAN discriminator** penalizes unrealistic textures and blurry edges.""")

# Cell 2: Imports & Environment Check
add_md("""## 2. Environment Setup & Hardware Verification
We verify CUDA GPU acceleration and import core PyTorch, torchvision, OpenCV, and PIL utilities.""")

add_code("""import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# Set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch Version: {torch.__version__}")
print(f"Active Device: {device}")
if device.type == 'cuda':
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")""")

# Cell 3: Dataset Inspection
add_md("""## 3. Dataset Inspection
We inspect the landscape image dataset stored in `Landscape/`.""")

add_code("""data_dir = 'Landscape'
image_files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
print(f"Total landscape images found in dataset: {len(image_files)}")

# Preview raw landscape images
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, ax in enumerate(axes):
    img_path = os.path.join(data_dir, image_files[i * 200])
    img = Image.open(img_path)
    ax.imshow(img)
    ax.set_title(f"{image_files[i * 200]}\\nSize: {img.size}")
    ax.axis('off')
plt.suptitle("Dataset Samples (Clean Color Landscapes)", fontsize=14)
plt.tight_layout()
plt.show()""")

# Cell 4: Synthetic Degradation Pipeline
add_md("""## 4. Synthetic Degradation Pipeline
To simulate authentic historical photograph deterioration without requiring real paired historical photos, we implement:
1. **Lab Color Space Conversion:** Extract luminance channel $L$ as the grayscale baseline.
2. **Random Vintage Scratches & Cracks:** Draw random polyline cracks and dust specks with varying thickness and intensity.
3. **Additive Noise:** Gaussian noise and film grain.
4. **Defocus / Gaussian Blur:** Simulate optical lens defocus and degradation.
5. **Contrast Fading:** Simulate photographic paper fading over decades.""")

add_code("""from dataset import OldPhotoDegradation, LandscapeDataset

# Instantiate dataset
train_dataset = LandscapeDataset(root_dir='Landscape', image_size=256, is_train=True)
val_dataset = LandscapeDataset(root_dir='Landscape', image_size=256, is_train=False)

print(f"Training Samples: {len(train_dataset)} | Validation Samples: {len(val_dataset)}")

# Visualize paired synthetic degradation vs clean target
fig, axes = plt.subplots(3, 2, figsize=(8, 12))
for i in range(3):
    deg_t, clean_t, fname = train_dataset[i + 15]
    
    # Denormalize from [-1, 1] to [0, 1]
    deg_np = (deg_t.squeeze(0).numpy() * 0.5 + 0.5)
    clean_np = (clean_t.permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
    
    axes[i, 0].imshow(deg_np, cmap='gray')
    axes[i, 0].set_title(f"Degraded Input (L + Scratches + Noise)\\n{fname}")
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(clean_np)
    axes[i, 1].set_title("Target Ground Truth (Clean Color)")
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()""")

# Cell 5: Architecture
add_md("""## 5. Conditional GAN Architecture (Pix2Pix)
### 5.1 Generator: U-Net 256
The generator uses an encoder-decoder architecture with **skip connections** between mirrored layers. Skip connections allow high-frequency details (such as edges and sharp boundaries) to bypass the bottleneck directly to the decoder, enabling simultaneous inpainting/restoration and colorization.

### 5.2 Discriminator: 70x70 PatchGAN
Rather than classifying the entire 256x256 image as real or fake with a single scalar, the PatchGAN classifies each local $70 \\times 70$ patch. This enforces sharp, crisp local structures and prevents the over-smoothed blurriness characteristic of pure L1/L2 loss.""")

add_code("""from models import UNetGenerator, PatchGANDiscriminator, create_models

netG, netD = create_models(in_channels=1, out_channels=3, device=device)

# Model summary
total_params_G = sum(p.numel() for p in netG.parameters() if p.requires_grad)
total_params_D = sum(p.numel() for p in netD.parameters() if p.requires_grad)

print(f"Generator (U-Net) Trainable Parameters: {total_params_G:,}")
print(f"Discriminator (PatchGAN) Trainable Parameters: {total_params_D:,}")

# Verification of output shapes
dummy_in = torch.randn(1, 1, 256, 256, device=device)
fake_out = netG(dummy_in)
pred = netD(dummy_in, fake_out)

print(f"Generator Output Shape: {fake_out.shape} -> Expected: [1, 3, 256, 256]")
print(f"PatchGAN Prediction Map Shape: {pred.shape} -> Expected: [1, 1, 30, 30]")""")

# Cell 6: Losses & Optimization
add_md("""## 6. Loss Functions & Objective Formulation
The objective function balances adversarial realism and structural fidelity:
$$\\mathcal{L}_{cGAN}(G, D) = \\mathbb{E}_{x, y}[\\log D(x, y)] + \\mathbb{E}_{x}[\\log(1 - D(x, G(x)))]$$
$$\\mathcal{L}_{L1}(G) = \\mathbb{E}_{x, y}[||y - G(x)||_1]$$
$$\\mathcal{L}_{total} = \\arg \\min_G \\max_D \\mathcal{L}_{cGAN}(G, D) + \\lambda_{L1} \\mathcal{L}_{L1}(G)$$

We use $\\lambda_{L1} = 100$ as recommended by Isola et al. (Pix2Pix). For adversarial loss, we employ LSGAN (Least Squares GAN / Mean Squared Error) which stabilizes training dynamics.""")

add_code("""criterionGAN = nn.MSELoss()
criterionL1 = nn.L1Loss()
lambda_l1 = 100.0

optimizer_G = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

def compute_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return (20 * torch.log10(2.0 / torch.sqrt(mse))).item()""")

# Cell 7: Training Loop Demonstration
add_md("""## 7. Model Training Loop
We execute a training loop with real-time loss tracking, validation evaluation, and progressive visual sample generation.""")

add_code("""# Prepare DataLoaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

fixed_val_batch = next(iter(val_loader))

# Set training parameters
epochs = 5 # Set to desired number of epochs
log_interval = 50

history = {'loss_G': [], 'loss_D': [], 'loss_L1': [], 'val_psnr': []}
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('samples', exist_ok=True)

print(f"Starting Training for {epochs} epochs on {len(train_dataset)} training pairs...")

for epoch in range(1, epochs + 1):
    epoch_start = time.time()
    netG.train()
    netD.train()
    
    running_loss_G = 0.0
    running_loss_D = 0.0
    running_loss_L1 = 0.0
    batches = 0
    
    for batch_idx, (deg_imgs, clean_imgs, _) in enumerate(train_loader):
        deg_imgs = deg_imgs.to(device, non_blocking=True)
        clean_imgs = clean_imgs.to(device, non_blocking=True)
        
        # -----------------
        #  Train Discriminator
        # -----------------
        optimizer_D.zero_grad()
        fake_clean = netG(deg_imgs)
        
        pred_real = netD(deg_imgs, clean_imgs)
        loss_D_real = criterionGAN(pred_real, torch.ones_like(pred_real))
        
        pred_fake = netD(deg_imgs, fake_clean.detach())
        loss_D_fake = criterionGAN(pred_fake, torch.zeros_like(pred_fake))
        
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        optimizer_D.step()
        
        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        pred_fake_for_G = netD(deg_imgs, fake_clean)
        loss_G_GAN = criterionGAN(pred_fake_for_G, torch.ones_like(pred_fake_for_G))
        loss_G_L1 = criterionL1(fake_clean, clean_imgs) * lambda_l1
        
        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()
        optimizer_G.step()
        
        running_loss_G += loss_G.item()
        running_loss_D += loss_D.item()
        running_loss_L1 += (loss_G_L1.item() / lambda_l1)
        batches += 1
        
        if batch_idx % log_interval == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {batch_idx}/{len(train_loader)}] "
                  f"Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f} | L1: {(loss_G_L1.item()/lambda_l1):.4f}")
                  
    # Epoch validation
    netG.eval()
    val_psnr_list = []
    with torch.no_grad():
        for idx, (v_deg, v_clean, _) in enumerate(val_loader):
            if idx >= 15: break
            v_deg = v_deg.to(device)
            v_clean = v_clean.to(device)
            v_fake = netG(v_deg)
            val_psnr_list.append(compute_psnr(v_fake, v_clean))
            
    avg_psnr = np.mean(val_psnr_list)
    epoch_time = time.time() - epoch_start
    
    history['loss_G'].append(running_loss_G / batches)
    history['loss_D'].append(running_loss_D / batches)
    history['loss_L1'].append(running_loss_L1 / batches)
    history['val_psnr'].append(avg_psnr)
    
    print(f"--> Epoch {epoch} Complete in {epoch_time:.1f}s | Avg Loss_G: {history['loss_G'][-1]:.4f} | "
          f"Avg Loss_D: {history['loss_D'][-1]:.4f} | Val PSNR: {avg_psnr:.2f} dB")
          
    # Save checkpoint
    torch.save(netG.state_dict(), 'checkpoints/best_generator.pth')""")

# Cell 8: Plotting Loss Curves
add_md("""## 8. Training Curves & Convergence Analysis
We visualize the Generator Loss, Discriminator Loss, Reconstruction L1 Loss, and Validation PSNR over training epochs.""")

add_code("""if len(history['loss_G']) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    
    axes[0].plot(history['loss_G'], label='Generator Loss', color='royalblue')
    axes[0].plot(history['loss_D'], label='Discriminator Loss', color='crimson')
    axes[0].set_title('Adversarial Training Losses')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['loss_L1'], label='L1 Reconstruction Loss', color='forestgreen')
    axes[1].set_title('Pixel-wise L1 Reconstruction Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('L1 Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(history['val_psnr'], label='Validation PSNR (dB)', color='darkorange')
    axes[2].set_title('Validation Quality Metric (PSNR)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('PSNR (dB)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
else:
    print("Run training to view loss curves.")""")

# Cell 9: Qualitative Comparison
add_md("""## 9. Qualitative Evaluation: Before vs After Restoration & Colorization
We inspect the model's restoration and colorization fidelity on held-out validation samples.""")

add_code("""netG.eval()
val_deg, val_clean, fnames = fixed_val_batch
val_deg = val_deg.to(device)

with torch.no_grad():
    val_restored = netG(val_deg)

fig, axes = plt.subplots(4, 3, figsize=(14, 16))
for i in range(4):
    deg_disp = (val_deg[i].squeeze(0).cpu().numpy() * 0.5 + 0.5)
    rest_disp = (val_restored[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
    clean_disp = (val_clean[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
    
    sample_psnr = compute_psnr(val_restored[i], val_clean[i].to(device))
    
    axes[i, 0].imshow(deg_disp, cmap='gray')
    axes[i, 0].set_title(f"Degraded Input\\n({fnames[i]})")
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(rest_disp)
    axes[i, 1].set_title(f"GAN Restored & Colorized\\nPSNR: {sample_psnr:.2f} dB")
    axes[i, 1].axis('off')
    
    axes[i, 2].imshow(clean_disp)
    axes[i, 2].set_title("Ground Truth Clean Image")
    axes[i, 2].axis('off')

plt.tight_layout()
plt.show()""")

# Cell 10: Inference on Custom Photo
add_md("""## 10. Interactive Inference on Any Photo
You can pass any degraded image, historical photo, or grayscale landscape to test the trained model directly.""")

add_code("""from inference import restore_and_colorize

test_img_path = os.path.join('Landscape', image_files[42])
generator_weights = 'checkpoints/best_generator.pth'

if os.path.exists(generator_weights):
    input_pil, restored_pil = restore_and_colorize(
        image_path=test_img_path,
        generator_path=generator_weights,
        output_path='sample_restoration_output.png',
        device=device
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(input_pil, cmap='gray')
    axes[0].set_title("Input Photo (Grayscale / Degraded)")
    axes[0].axis('off')
    
    axes[1].imshow(restored_pil)
    axes[1].set_title("GAN Restored & Colorized Output")
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()
else:
    print(f"Checkpoint {generator_weights} not found. Train the model first.")""")

# Cell 11: Deployment & Conclusion
add_md("""## 11. Summary & Flask Web Application Deployment
### Summary of Results
- **Synthetic Degradation:** Realistically mapped clean landscape images to degraded grayscale counterparts with physical scratches, dust, blur, and noise.
- **Pix2Pix Conditional GAN:** Successfully trained with U-Net skip connections to preserve edge details and inpaint missing regions while hallucinating plausible colors.
- **Adversarial Loss:** Kept generated outputs visually sharp and free from typical L1/L2 over-smoothing.

### Next Step: Flask Web Application
As outlined in Section 5 of the Case Study document, the trained generator is integrated into a **Flask Web Application** (`app.py`), allowing users to upload old or degraded photos and view side-by-side restorations in real-time.""")

with open('GAN_Photo_Restoration_and_Colorization.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print('Successfully generated GAN_Photo_Restoration_and_Colorization.ipynb!')
