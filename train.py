import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import LandscapeDataset
from models import create_models

def calculate_psnr(img1, img2):
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) between two image tensors [-1, 1].
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    # Dynamic range in [-1, 1] is 2.0
    psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))
    return psnr.item()


def save_sample_grid(netG, val_batch, epoch, step, save_dir, device):
    """
    Generates and saves side-by-side comparison images:
    [Degraded Input (L+scratches+noise) | GAN Output (Restored+Colorized) | Clean Ground Truth]
    """
    netG.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    deg_tensors, clean_tensors, filenames = val_batch
    deg_tensors = deg_tensors.to(device)
    clean_tensors = clean_tensors.to(device)
    
    with torch.no_grad():
        fake_clean = netG(deg_tensors)
        
    num_samples = min(4, deg_tensors.size(0))
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)
        
    for i in range(num_samples):
        # Denormalize [-1, 1] to [0, 1]
        deg_disp = (deg_tensors[i].squeeze(0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        fake_disp = (fake_clean[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        clean_disp = (clean_tensors[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        
        # Calculate PSNR for this sample
        sample_psnr = calculate_psnr(fake_clean[i], clean_tensors[i])
        
        axes[i, 0].imshow(deg_disp, cmap='gray')
        axes[i, 0].set_title(f"Degraded Input\n({filenames[i]})")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(fake_disp)
        axes[i, 1].set_title(f"GAN Restored & Colorized\n(PSNR: {sample_psnr:.2f} dB)")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(clean_disp)
        axes[i, 2].set_title("Ground Truth Clean Color")
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"epoch_{epoch:03d}_step_{step:05d}.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    netG.train()
    return save_path


def train_pix2pix(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"=== Starting GAN Training on device: {device} ===")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    
    # 1. Dataset & DataLoaders
    train_dataset = LandscapeDataset(
        root_dir=args.data_dir,
        image_size=args.image_size,
        is_train=True,
        split_ratio=args.split_ratio
    )
    val_dataset = LandscapeDataset(
        root_dir=args.data_dir,
        image_size=args.image_size,
        is_train=False,
        split_ratio=args.split_ratio
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2
    )
    
    print(f"Dataset stats: Train samples={len(train_dataset)}, Val samples={len(val_dataset)}")
    
    # Fixed validation batch for visual comparison across epochs
    fixed_val_batch = next(iter(val_loader))
    
    # 2. Models
    netG, netD = create_models(in_channels=1, out_channels=3, device=device)
    
    # 3. Losses
    criterionGAN = nn.MSELoss() # LSGAN
    criterionL1 = nn.L1Loss()
    lambda_l1 = args.lambda_l1
    
    # 4. Optimizers & Schedulers
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    def lr_lambda(epoch):
        # Keep initial lr for first half, then linearly decay to 0
        if epoch < args.decay_epoch:
            return 1.0
        return 1.0 - (epoch - args.decay_epoch) / float(args.epochs - args.decay_epoch + 1)
        
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lr_lambda)
    
    # Initial sample before training
    print("Generating baseline sample before training...")
    save_sample_grid(netG, fixed_val_batch, 0, 0, args.sample_dir, device)
    
    best_val_psnr = -1.0
    history = {'loss_G': [], 'loss_D': [], 'loss_L1': [], 'val_psnr': []}
    
    total_steps = 0
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        running_loss_G = 0.0
        running_loss_D = 0.0
        running_loss_L1 = 0.0
        batches_in_epoch = 0
        
        for batch_idx, (deg_imgs, clean_imgs, _) in enumerate(train_loader):
            if args.max_batches and batch_idx >= args.max_batches:
                break
                
            deg_imgs = deg_imgs.to(device, non_blocking=True)
            clean_imgs = clean_imgs.to(device, non_blocking=True)
            
            # ----------------------------------------------------
            #  Train Discriminator: maximize log(D(x, y)) + log(1 - D(x, G(x)))
            # ----------------------------------------------------
            optimizer_D.zero_grad()
            
            # Generate fake clean images
            fake_clean = netG(deg_imgs)
            
            # Real pair
            pred_real = netD(deg_imgs, clean_imgs)
            target_real = torch.ones_like(pred_real, device=device)
            loss_D_real = criterionGAN(pred_real, target_real)
            
            # Fake pair (detach fake_clean so gradients don't flow to G here)
            pred_fake = netD(deg_imgs, fake_clean.detach())
            target_fake = torch.zeros_like(pred_fake, device=device)
            loss_D_fake = criterionGAN(pred_fake, target_fake)
            
            # Combined D loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            # ----------------------------------------------------
            #  Train Generator: maximize log(D(x, G(x))) + lambda * ||y - G(x)||_1
            # ----------------------------------------------------
            optimizer_G.zero_grad()
            
            # G wants D to think fake_clean is real
            pred_fake_for_G = netD(deg_imgs, fake_clean)
            loss_G_GAN = criterionGAN(pred_fake_for_G, target_real)
            
            # L1 reconstruction loss
            loss_G_L1 = criterionL1(fake_clean, clean_imgs) * lambda_l1
            
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizer_G.step()
            
            # Track losses
            running_loss_G += loss_G.item()
            running_loss_D += loss_D.item()
            running_loss_L1 += (loss_G_L1.item() / lambda_l1)
            batches_in_epoch += 1
            total_steps += 1
            
            if batch_idx % args.log_interval == 0:
                print(f"[Epoch {epoch:02d}/{args.epochs:02d}] [Batch {batch_idx:03d}/{len(train_loader):03d}] "
                      f"Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f} | "
                      f"GAN: {loss_G_GAN.item():.4f} | L1: {loss_G_L1.item() / lambda_l1:.4f}")
                      
            if total_steps % args.sample_interval == 0:
                sample_path = save_sample_grid(netG, fixed_val_batch, epoch, total_steps, args.sample_dir, device)
                print(f"--> Saved sample visual to {sample_path}")

        # Update learning rate
        scheduler_G.step()
        scheduler_D.step()
        
        # Epoch averages
        avg_loss_G = running_loss_G / max(batches_in_epoch, 1)
        avg_loss_D = running_loss_D / max(batches_in_epoch, 1)
        avg_loss_L1 = running_loss_L1 / max(batches_in_epoch, 1)
        history['loss_G'].append(avg_loss_G)
        history['loss_D'].append(avg_loss_D)
        history['loss_L1'].append(avg_loss_L1)
        
        # Validation Evaluation
        netG.eval()
        val_psnr_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for v_deg, v_clean, _ in val_loader:
                v_deg = v_deg.to(device)
                v_clean = v_clean.to(device)
                v_fake = netG(v_deg)
                val_psnr_sum += calculate_psnr(v_fake, v_clean)
                val_count += 1
                if val_count >= 20: # evaluate on 80 validation images for quick feedback
                    break
        netG.train()
        
        val_psnr = val_psnr_sum / max(val_count, 1)
        history['val_psnr'].append(val_psnr)
        epoch_time = time.time() - epoch_start
        
        print(f"=== Epoch {epoch:02d} Summary: Time={epoch_time:.1f}s | Avg_Loss_G={avg_loss_G:.4f} | "
              f"Avg_Loss_D={avg_loss_D:.4f} | Avg_L1={avg_loss_L1:.4f} | Val_PSNR={val_psnr:.2f} dB ===")
              
        # Save sample at end of each epoch
        save_sample_grid(netG, fixed_val_batch, epoch, total_steps, args.sample_dir, device)
        
        # Save checkpoints
        torch.save(netG.state_dict(), os.path.join(args.checkpoint_dir, 'latest_generator.pth'))
        torch.save({
            'epoch': epoch,
            'netG': netG.state_dict(),
            'netD': netD.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
            'history': history
        }, os.path.join(args.checkpoint_dir, 'latest_checkpoint.pth'))
        
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(netG.state_dict(), os.path.join(args.checkpoint_dir, 'best_generator.pth'))
            print(f"*** New best model saved with Val PSNR = {val_psnr:.2f} dB ***")
            
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes! Best Val PSNR = {best_val_psnr:.2f} dB")
    return netG, netD, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Pix2Pix Conditional GAN for Old Photo Restoration and Colorization")
    parser.add_argument('--data_dir', type=str, default='Landscape', help="Path to Landscape dataset")
    parser.add_argument('--epochs', type=int, default=15, help="Number of training epochs")
    parser.add_argument('--decay_epoch', type=int, default=8, help="Epoch to start linearly decaying learning rate")
    parser.add_argument('--batch_size', type=int, default=8, help="Mini-batch size")
    parser.add_argument('--image_size', type=int, default=256, help="Image resolution (256x256)")
    parser.add_argument('--lr', type=float, default=0.0002, help="Initial learning rate for Adam")
    parser.add_argument('--lambda_l1', type=float, default=100.0, help="Weight for L1 reconstruction loss")
    parser.add_argument('--split_ratio', type=float, default=0.9, help="Train/Validation split ratio")
    parser.add_argument('--num_workers', type=int, default=4, help="DataLoader worker processes")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Directory to save model checkpoints")
    parser.add_argument('--sample_dir', type=str, default='samples', help="Directory to save visual image samples")
    parser.add_argument('--log_interval', type=int, default=25, help="Batches between terminal progress logs")
    parser.add_argument('--sample_interval', type=int, default=100, help="Steps between visual sample generation")
    parser.add_argument('--max_batches', type=int, default=None, help="Max batches per epoch (useful for test runs)")
    parser.add_argument('--no_cuda', action='store_true', help="Disable CUDA acceleration")
    
    args = parser.parse_args()
    train_pix2pix(args)
