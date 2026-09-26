import os
import io
import time
import base64
from PIL import Image
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from flask import Flask, render_template, request, jsonify, send_from_directory
from models import UNetGenerator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Device & Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'best_generator.pth')
if not os.path.exists(generator_path):
    generator_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'latest_generator.pth')

generator = None
if os.path.exists(generator_path):
    print(f"Loading generator model from: {generator_path} on {device}")
    generator = UNetGenerator(in_channels=1, out_channels=3)
    checkpoint = torch.load(generator_path, map_location=device)
    if 'netG' in checkpoint:
        generator.load_state_dict(checkpoint['netG'])
    else:
        generator.load_state_dict(checkpoint)
    generator.to(device)
    generator.eval()
else:
    print("Warning: Generator checkpoint not found. Model will initialize upon training.")


def run_inference(pil_img, image_size=256):
    """
    Runs generator on input PIL image and returns restored PIL image.
    """
    global generator
    if generator is None:
        raise RuntimeError("Generator model is not loaded. Train the model first.")

    orig_w, orig_h = pil_img.size
    
    # Extract luminance / L-channel
    if pil_img.mode != 'L':
        rgb_np = np.array(pil_img.convert('RGB'))
        lab = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
        L = lab[:, :, 0]
        input_pil = Image.fromarray(L, mode='L')
    else:
        input_pil = pil_img

    transform = T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    
    input_tensor = transform(input_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        start = time.time()
        output_tensor = generator(input_tensor)
        inference_time_ms = (time.time() - start) * 1000.0

    # Denormalize output [-1, 1] to [0, 255] uint8 RGB
    out_np = (output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
    restored_pil = Image.fromarray((out_np * 255).astype(np.uint8))
    restored_resized = restored_pil.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    
    return input_pil, restored_resized, inference_time_ms


def pil_to_base64(pil_img, format="PNG"):
    buffered = io.BytesIO()
    pil_img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


@app.route('/')
def index():
    samples = [
        {'id': 15, 'thumb': '/static/samples/sample_15_degraded.png', 'name': 'Alpine Peak'},
        {'id': 42, 'thumb': '/static/samples/sample_42_degraded.png', 'name': 'Misty Forest'},
        {'id': 105, 'thumb': '/static/samples/sample_105_degraded.png', 'name': 'Valley River'},
        {'id': 230, 'thumb': '/static/samples/sample_230_degraded.png', 'name': 'Sunset Ridge'},
    ]
    return render_template('index.html', samples=samples, has_gpu=torch.cuda.is_available())


@app.route('/api/restore', methods=['POST'])
def restore():
    try:
        # Check if sample ID was passed
        sample_id = request.form.get('sample_id')
        if sample_id:
            sample_path = os.path.join(os.path.dirname(__file__), 'static', 'samples', f'sample_{sample_id}_degraded.png')
            if not os.path.exists(sample_path):
                return jsonify({'error': 'Sample image not found'}), 404
            img = Image.open(sample_path)
        elif 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            img = Image.open(file.stream)
        else:
            return jsonify({'error': 'No image provided'}), 400

        input_pil, restored_pil, inf_time = run_inference(img)

        # Convert images to base64 for instant display without disk clutter
        input_b64 = "data:image/png;base64," + pil_to_base64(input_pil)
        restored_b64 = "data:image/png;base64," + pil_to_base64(restored_pil)

        return jsonify({
            'success': True,
            'input_image': input_b64,
            'restored_image': restored_b64,
            'inference_time_ms': round(inf_time, 2),
            'dimensions': f"{restored_pil.width}x{restored_pil.height}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
