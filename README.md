# GAN-Based Old Photo Restoration and Colorization

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black.svg)](https://flask.palletsprojects.com/)

A Deep Learning framework using **Pix2Pix Conditional Generative Adversarial Networks (cGAN)** to restore and colorize degraded vintage photographs. The pipeline synthetically simulates realistic photo decay (scratches, dust, noise, blur, and fading) and trains a skip-connected U-Net generator with a PatchGAN discriminator to jointly remove defects and reconstruct natural color.


## 🌟 Key Features

1. **Synthetic Degradation Engine (`dataset.py`)**:
   - Random vintage scratches, cracks, and hairline fractures.
   - Additive Gaussian film grain and sensor noise.
   - Defocus and Gaussian blur.
   - Contrast fading and luminance extraction ($L$-channel in CIE Lab space).
2. **Pix2Pix GAN Architecture (`models.py`)**:
   - **Generator**: 8-layer U-Net with symmetric encoder-decoder blocks and skip connections to preserve high-frequency edge textures.
   - **Discriminator**: 70×70 PatchGAN classifier providing localized adversarial feedback.
   - **Objective Function**: $\mathcal{L}_{GAN}(G, D) + \lambda \mathcal{L}_{L1}(G)$ with $\lambda = 100.0$.
3. **Interactive Web Application (`app.py`)**:
   - Flask-based web application with interactive before/after split slider.
   - Upload custom degraded photographs or explore pre-loaded landscape samples.
4. **Jupyter Notebook (`GAN_Photo_Restoration_and_Colorization.ipynb`)**:
   - Complete step-by-step case study report with visualizations, data analysis, training curves, and quantitative metrics (PSNR, SSIM).

---

## 📁 Repository Structure

```text
├── app.py                                   # Flask web server and inference API
├── dataset.py                               # Synthetic degradation pipeline & PyTorch Dataset
├── models.py                                # Pix2Pix U-Net Generator & PatchGAN Discriminator
├── train.py                                 # GAN training loop with validation & checkpointing
├── inference.py                             # Command-line single-image restoration tool
├── GAN_Photo_Restoration_and_Colorization.ipynb # Complete case study notebook
├── requirements.txt                         # Python package dependencies
├── .gitignore                               # Git ignore rules for datasets, weights & caches
├── templates/
│   └── index.html                           # Web application HTML interface
├── static/
│   ├── css/
│   │   └── style.css                        # Modern UI styling
│   ├── samples/                             # Sample images for live demo
│   └── uploads/                             # Upload destination (.gitkeep)
└── checkpoints/                             # Model weights folder (.gitkeep)
```

---

## 🚀 Getting Started

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/Madhav-Sreejith/GAN-Based-Old-Photo-Restoration-and-Colorization.git
cd GAN-Based-Old-Photo-Restoration-and-Colorization

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Checkpoints

Place your trained generator weights (`best_generator.pth` or `latest_generator.pth`) in the `checkpoints/` directory:
```text
checkpoints/
└── best_generator.pth
```
*(Note: Model `.pth` weight files exceed GitHub's 100MB file limit and are excluded via `.gitignore`.)*

---

## 💻 Running the Web Application

Start the Flask application locally:

```bash
python app.py
```

Then open your browser and navigate to:
```text
http://127.0.0.1:5000/
```

- Upload your own vintage/degraded photograph or click one of the preset landscape samples.
- Drag the comparison slider to view the restored and colorized image side-by-side.

---

## 🔬 Command-Line Inference

To restore a single image directly from terminal:

```bash
python inference.py \
  --image_path path/to/old_photo.jpg \
  --generator_path checkpoints/best_generator.pth \
  --output_path restored_result.png \
  --image_size 256
```

---

## 🏋️ Model Training

To train the Pix2Pix GAN from scratch on a landscape dataset:

1. Prepare your dataset images in a folder named `Landscape/` (or specify `--data_dir`).
2. Run the training script:

```bash
python train.py \
  --data_dir Landscape \
  --epochs 15 \
  --batch_size 8 \
  --image_size 256 \
  --lr 0.0002 \
  --lambda_l1 100.0 \
  --checkpoint_dir checkpoints \
  --sample_dir samples
```

---

## 📊 Evaluation & Metrics

The restored outputs are evaluated using:
- **Peak Signal-to-Noise Ratio (PSNR)**: Measures pixel-level fidelity.
- **Structural Similarity Index (SSIM)**: Evaluates structural, luminance, and contrast preservation.
- **Adversarial Realism**: Assessed via PatchGAN discriminator score.
