import os
import random
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class OldPhotoDegradation:
    """
    Applies synthetic old-photo degradation pipeline:
    1. Grayscale / Luminance extraction (Lab L-channel or Grayscale)
    2. Random scratches, cracks, and dust specks
    3. Additive Gaussian noise & film grain
    4. Gaussian blur / defocus
    5. Fading / contrast attenuation
    """
    def __init__(self, scratch_prob=0.85, noise_prob=0.85, blur_prob=0.75):
        self.scratch_prob = scratch_prob
        self.noise_prob = noise_prob
        self.blur_prob = blur_prob

    def add_scratches(self, img_np):
        """
        Draws realistic random vintage scratches and cracks on a single-channel image.
        img_np: numpy array (H, W), dtype uint8 [0, 255]
        """
        h, w = img_np.shape[:2]
        scratch_layer = np.zeros((h, w), dtype=np.uint8)
        
        # Number of scratches: between 3 and 12
        num_scratches = random.randint(3, 10)
        
        for _ in range(num_scratches):
            # Scratch type: straight line or multi-segment curve/crack
            points = []
            num_pts = random.randint(2, 6)
            start_x = random.randint(0, w)
            start_y = random.randint(0, h)
            points.append((start_x, start_y))
            
            curr_x, curr_y = start_x, start_y
            step_len = random.randint(20, 80)
            
            for _ in range(1, num_pts):
                angle = random.uniform(-np.pi, np.pi)
                curr_x = int(np.clip(curr_x + step_len * np.cos(angle), 0, w - 1))
                curr_y = int(np.clip(curr_y + step_len * np.sin(angle), 0, h - 1))
                points.append((curr_x, curr_y))
                
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            thickness = random.choice([1, 1, 2])
            intensity = random.choice([40, 200, 230, 255]) # dark or light scratch
            cv2.polylines(scratch_layer, [pts], isClosed=False, color=intensity, thickness=thickness)

        # Add small dust / pepper specks
        num_specks = random.randint(10, 40)
        for _ in range(num_specks):
            cx = random.randint(0, w - 1)
            cy = random.randint(0, h - 1)
            radius = random.randint(1, 2)
            cv2.circle(scratch_layer, (cx, cy), radius, random.choice([20, 240]), -1)

        # Blend scratch layer with original image
        mask = scratch_layer > 0
        result = img_np.copy()
        result[mask] = scratch_layer[mask]
        return result

    def add_noise(self, img_np):
        """
        Adds Gaussian and film grain noise.
        img_np: numpy array (H, W), dtype uint8 [0, 255]
        """
        h, w = img_np.shape[:2]
        std = random.uniform(10, 28)
        gaussian_noise = np.random.normal(0, std, (h, w))
        noisy_img = img_np.astype(np.float32) + gaussian_noise
        return np.clip(noisy_img, 0, 255).astype(np.uint8)

    def add_blur(self, img_np):
        """
        Adds subtle blur or lens defocus.
        """
        kernel_size = random.choice([3, 5])
        sigma = random.uniform(0.6, 1.8)
        blurred = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), sigma)
        return blurred

    def add_fading(self, img_np):
        """
        Simulates aged paper contrast fading and slight gamma shift.
        """
        alpha = random.uniform(0.85, 1.05) # contrast
        beta = random.uniform(-10, 15)      # brightness
        faded = cv2.convertScaleAbs(img_np, alpha=alpha, beta=beta)
        return faded

    def degrade(self, pil_rgb_img):
        """
        Given a PIL RGB image, converts to Lab L-channel (or grayscale),
        applies synthetic degradation, and returns:
          degraded_L: PIL Image (Grayscale/L)
          clean_rgb: PIL Image (Original RGB)
        """
        # Convert to numpy RGB
        rgb_np = np.array(pil_rgb_img)
        
        # Convert to Lab to extract L channel (or standard Grayscale)
        # Note: In OpenCV, cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB) gives L in [0, 255]
        lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
        L = lab[:, :, 0] # Luminance channel
        
        degraded = L.copy()
        
        # 1. Scratches & Cracks
        if random.random() < self.scratch_prob:
            degraded = self.add_scratches(degraded)
            
        # 2. Add Noise & Film Grain
        if random.random() < self.noise_prob:
            degraded = self.add_noise(degraded)
            
        # 3. Add Blur
        if random.random() < self.blur_prob:
            degraded = self.add_blur(degraded)
            
        # 4. Fading
        degraded = self.add_fading(degraded)
        
        degraded_pil = Image.fromarray(degraded, mode='L')
        return degraded_pil, pil_rgb_img


class LandscapeDataset(Dataset):
    """
    Paired Landscape Dataset for Pix2Pix Old Photo Restoration and Colorization.
    Yields:
      - degraded: Tensor [1, H, W] in range [-1, 1]
      - clean: Tensor [3, H, W] in range [-1, 1]
      - filename: str
    """
    def __init__(self, root_dir, image_size=256, is_train=True, split_ratio=0.9, seed=42):
        super().__init__()
        self.root_dir = root_dir
        self.image_size = image_size
        self.is_train = is_train
        self.degrader = OldPhotoDegradation()
        
        all_files = sorted([
            f for f in os.listdir(root_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        # Seeded deterministic train/val split
        rng = random.Random(seed)
        shuffled = list(all_files)
        rng.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * split_ratio)
        if is_train:
            self.files = shuffled[:split_idx]
        else:
            self.files = shuffled[split_idx:]
            
        # Standard paired transforms
        self.resize = T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC)
        self.to_tensor = T.ToTensor() # maps [0, 255] to [0.0, 1.0]
        self.normalize_rgb = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # maps to [-1, 1]
        self.normalize_l = T.Normalize((0.5,), (0.5,)) # maps to [-1, 1]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        img_path = os.path.join(self.root_dir, filename)
        clean_img = Image.open(img_path).convert('RGB')
        
        # Resize clean image
        clean_img = self.resize(clean_img)
        
        # Random horizontal flip for data augmentation during training
        if self.is_train and random.random() > 0.5:
            clean_img = TF.hflip(clean_img)
            
        # Apply synthetic degradation
        degraded_img, clean_img = self.degrader.degrade(clean_img)
        
        # Convert to tensors
        clean_tensor = self.normalize_rgb(self.to_tensor(clean_img))
        degraded_tensor = self.normalize_l(self.to_tensor(degraded_img))
        
        return degraded_tensor, clean_tensor, filename
