import torch
import torch.nn as nn

def init_weights(net, init_type='normal', gain=0.02):
    """
    Initialize network weights as specified in Pix2Pix paper:
    Normal distribution with mean=0.0, std=0.02.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    return net


class UNetBlock(nn.Module):
    """
    A recursive U-Net submodule with skip connection:
    down -> [submodule] -> up, and concatenates input with output.
    """
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(UNetBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            # Skip connection: concatenate feature maps along channel dimension
            return torch.cat([x, self.model(x)], 1)


class UNetGenerator(nn.Module):
    """
    U-Net 256 Generator for Image-to-Image translation.
    Takes degraded luminance (1 channel) or degraded RGB (3 channels)
    and reconstructs a restored, colorized 3-channel RGB image.
    """
    def __init__(self, in_channels=1, out_channels=3, num_downs=8, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=True):
        super(UNetGenerator, self).__init__()
        
        # Build U-Net from innermost layer outward
        # Innermost block (2x2 -> 1x1)
        unet_block = UNetBlock(ngf * 8, ngf * 8, submodule=None, innermost=True, norm_layer=norm_layer)
        
        # Intermediate blocks (8x8 -> 4x4 -> 2x2) with dropout
        for _ in range(num_downs - 5):
            unet_block = UNetBlock(ngf * 8, ngf * 8, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
            
        # Gradually upsample channels
        unet_block = UNetBlock(ngf * 4, ngf * 8, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UNetBlock(ngf * 2, ngf * 4, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UNetBlock(ngf, ngf * 2, submodule=unet_block, norm_layer=norm_layer)
        
        # Outermost block (takes in_channels -> outputs out_channels with Tanh)
        self.model = UNetBlock(out_channels, ngf, input_nc=in_channels, submodule=unet_block, outermost=True, norm_layer=norm_layer)

    def forward(self, x):
        return self.model(x)


class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN Discriminator.
    Evaluates whether local 70x70 patches of the generated/real image are authentic,
    conditioned on the degraded input image.
    Input: torch.cat([condition, target], dim=1) -> in_channels = 1 + 3 = 4
    """
    def __init__(self, in_channels=4, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super(PatchGANDiscriminator, self).__init__()
        
        # First layer (no normalization as per Pix2Pix specification)
        sequence = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        
        # Intermediate layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # Layer 4 (stride=1)
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # Output single-channel prediction map (stride=1)
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*sequence)

    def forward(self, condition, target):
        """
        Concatenates condition (degraded) and target (real/fake clean) along channel dimension.
        """
        x = torch.cat([condition, target], dim=1)
        return self.model(x)


def create_models(in_channels=1, out_channels=3, device='cuda'):
    """
    Convenience factory to instantiate and initialize Generator and Discriminator.
    """
    netG = UNetGenerator(in_channels=in_channels, out_channels=out_channels)
    netD = PatchGANDiscriminator(in_channels=in_channels + out_channels)
    
    init_weights(netG, 'normal', 0.02)
    init_weights(netD, 'normal', 0.02)
    
    netG = netG.to(device)
    netD = netD.to(device)
    return netG, netD
