import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import optim
from tqdm import tqdm
import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt


class ConvBlock(nn.Module):
    """Convolutional block with Conv2D + BatchNorm + ReLU + MaxPool"""
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1,
                 pool_size: int = 2, pool_stride: int = 2,
                 padding: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.99, eps=0.001)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.99, eps=0.001)
        
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x


class Encoder(nn.Module):
    """Encoder с 3 ConvBlocks - как в TensorFlow"""
    def __init__(self, latent_dim: int = 100, channel_num: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        
        # ← padding=0 (equivalent to 'valid' in TensorFlow)
        self.conv_block1 = ConvBlock(channel_num, 64, kernel_size=3, 
                                     stride=1, pool_size=2, pool_stride=2, padding=0)
        self.conv_block2 = ConvBlock(64, 128, kernel_size=3,
                                     stride=1, pool_size=2, pool_stride=2, padding=0)
        self.conv_block3 = ConvBlock(128, 256, kernel_size=3,
                                     stride=1, pool_size=2, pool_stride=2, padding=1)
        
        self.flatten = nn.Flatten()
        
        # С padding=0 для первых двух блоков, padding=1 для третьего
        # 32 -> (32-2)/2 = 15 -> (15-2)/2 = 6 -> (6-2)/2 = 2 (для TF 'valid')
        # Но с padding=1 в последнем: 6 -> (6-2)/2+1 = 3 -> (3-2)/2+1 = 2
        # Более точно: используем точные размеры как в TF
        self.feature_volume = 256 * 2 * 2  # ← 1024, как в TensorFlow
        
        self.output_layer = nn.Linear(self.feature_volume, latent_dim, bias=True)
        self.bn_out = nn.BatchNorm1d(latent_dim, momentum=0.99, eps=0.001)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)  # 32 -> 14
        x = self.conv_block2(x)  # 14 -> 5
        x = self.conv_block3(x)  # 5 -> 2
        x = self.flatten(x)
        x = F.relu(self.bn_out(self.output_layer(x)))
        return x


class Decoder(nn.Module):
    """Decoder with dense layers and ConvTranspose blocks"""
    def __init__(self, latent_dim: int = 100, out_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_volume = 256 * 4 * 4
        
        self.dense0 = nn.Linear(latent_dim, latent_dim, bias=True)
        self.bn0 = nn.BatchNorm1d(latent_dim, momentum=0.99, eps=0.001)
        
        self.dense1 = nn.Linear(latent_dim, 1024, bias=True)
        self.bn1 = nn.BatchNorm1d(1024, momentum=0.99, eps=0.001)
        
        self.dense2 = nn.Linear(1024, self.feature_volume, bias=True)
        self.bn2 = nn.BatchNorm1d(self.feature_volume, momentum=0.99, eps=0.001)
        
        self.conv_transpose1 = nn.ConvTranspose2d(256, 256, kernel_size=4,
                                                  stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256, momentum=0.99, eps=0.001)
        
        self.conv_transpose2 = nn.ConvTranspose2d(256, 256, kernel_size=3,
                                                  stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256, momentum=0.99, eps=0.001)
        
        self.conv_transpose3 = nn.ConvTranspose2d(256, 128, kernel_size=4,
                                                  stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128, momentum=0.99, eps=0.001)
        
        self.conv_transpose4 = nn.ConvTranspose2d(128, 128, kernel_size=3,
                                                  stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(128, momentum=0.99, eps=0.001)
        
        self.conv_transpose5 = nn.ConvTranspose2d(128, 64, kernel_size=4,
                                                  stride=2, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(64, momentum=0.99, eps=0.001)
        
        self.conv_transpose6 = nn.ConvTranspose2d(64, 64, kernel_size=3,
                                                  stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(64, momentum=0.99, eps=0.001)
        
        self.final_conv = nn.ConvTranspose2d(64, out_channels, kernel_size=3,
                                             stride=1, padding=1, bias=True)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn0(self.dense0(z)))
        x = F.relu(self.bn1(self.dense1(x)))
        x = F.relu(self.bn2(self.dense2(x)))
        x = x.view(-1, 256, 4, 4)
        
        x = F.relu(self.bn3(self.conv_transpose1(x)))
        x = F.relu(self.bn4(self.conv_transpose2(x)))
        x = F.relu(self.bn5(self.conv_transpose3(x)))
        x = F.relu(self.bn6(self.conv_transpose4(x)))
        x = F.relu(self.bn7(self.conv_transpose5(x)))
        x = F.relu(self.bn8(self.conv_transpose6(x)))
        x = torch.sigmoid(self.final_conv(x))
        
        return x


class VAE(nn.Module):
    """Variational Autoencoder"""
    def __init__(self, latent_dim: int = 100, channel_num: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.channel_num = channel_num
        
        self.encoder = Encoder(latent_dim=latent_dim, channel_num=channel_num)
        self.decoder = Decoder(latent_dim=latent_dim, out_channels=channel_num)
        
        # Latent space parameters
        self.mu_layer = nn.Linear(latent_dim, latent_dim)
        self.logvar_layer = nn.Linear(latent_dim, latent_dim)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode image to mean and log-variance"""
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: sample from N(mu, exp(logvar))"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to image"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate samples from standard normal distribution"""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
    

class VAEv2(nn.Module):
    """Variational Autoencoder из calculate_experiment_anton - гибкая архитектура"""
    
    def __init__(self, input_channels: int = 3, input_size: int = 32, 
                 filter_sizes: List[int] = None, latent_dim: int = 64, 
                 kernel_size: int = 3, stride: int = 2, padding: int = 1, 
                 is_variational: bool = True):
        super().__init__()
        
        if filter_sizes is None:
            filter_sizes = [32, 64, 128]
        
        self.latent_dim = latent_dim
        self.input_channels = input_channels
        self.input_size = input_size
        self.filter_sizes = filter_sizes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.is_variational = is_variational
        
        self._build_encoder()
        self._calculate_conv_output_shape()
        self._build_latent_layers()
        self._build_decoder()

    def _build_encoder(self) -> None:
        encoder_layers = []
        in_channels = self.input_channels
        
        for fs in self.filter_sizes:
            encoder_layers.extend([
                nn.Conv2d(in_channels, fs, kernel_size=self.kernel_size, 
                         stride=self.stride, padding=self.padding),
                nn.BatchNorm2d(fs),
                nn.ReLU()
            ])
            in_channels = fs
            
        self.encoder = nn.Sequential(*encoder_layers)

    def _calculate_conv_output_shape(self) -> None:
        """
        Вычисляет размер изображения после применения свёрточных слоёв.
        """
        current_size = self.input_size
        self.data_sizes = [current_size]
        for _ in self.filter_sizes:
            current_size = int(np.floor(
                (current_size + 2 * self.padding - self.kernel_size) / self.stride + 1
            ))
            self.data_sizes.append(current_size)

        self.flat_dim = self.filter_sizes[-1] * current_size * current_size
        self.conv_out_shape = torch.Size([int(self.filter_sizes[-1]), current_size, current_size])

    def _build_latent_layers(self) -> None:
        self.fc_mu = nn.Linear(self.flat_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, self.latent_dim)
        self.fc_decode = nn.Linear(self.latent_dim, self.flat_dim)

    def _build_decoder(self) -> None:
        decoder_layers = []
        reversed_filters = list(reversed(self.filter_sizes))
        in_channels = reversed_filters[0]
        
        for i, fs in enumerate(reversed_filters[1:] + [self.input_channels]):
            output_padding = self.data_sizes[::-1][i+1] - (
                (self.data_sizes[::-1][i] - 1) * self.stride - 2 * self.padding + self.kernel_size
            )
            
            decoder_layers.append(
                nn.ConvTranspose2d(
                    in_channels, 
                    fs, 
                    kernel_size=self.kernel_size, 
                    stride=self.stride, 
                    padding=self.padding,
                    output_padding=output_padding
                )
            )
            if i < len(reversed_filters) - 1:  # No BN/ReLU on last layer
                decoder_layers.append(nn.BatchNorm2d(fs))
                decoder_layers.append(nn.ReLU())
            in_channels = fs
            
        decoder_layers.append(nn.Sigmoid())  # Final activation
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        x_flat = x.view(x.size(0), -1)
        self.encoder_exit = x_flat
        return self.fc_mu(x_flat), self.fc_logvar(x_flat)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        if self.is_variational:
            return mu + eps * std
        else:
            return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc_decode(z)
        self.decoder_input = x
        x = x.view(z.size(0), *self.conv_out_shape)
        return self.decoder(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAETrainer:
    def __init__(self, model: VAE, device: torch.device, lr: float = 1e-4,
                 weight_decay: float = 1e-5, verbose = False):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr, 
                                    weight_decay=weight_decay
                                    )
        # self.scheduler = optim.lr_scheduler.StepLR(
        #     self.optimizer, step_size=30, gamma=0.1
        # )
        self.verbose = verbose
        self.training_history = {
            'total_loss': [],
            'recon_loss': [],
            'kl_loss': [],
            'val_total_loss': [],
            'val_recon_loss': [],
            'val_kl_loss': []
        }
    
    def reconstruction_loss(self, x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction loss: sum MSE over spatial dims, mean over batch"""
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        batch_size = x.size(0)
        return recon_loss / batch_size
    
    def kl_divergence_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL divergence loss: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))"""
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        batch_size = mu.size(0)
        return kl_loss / batch_size
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor], 
                   alpha: float = 1.0) -> Dict[str, float]:
        """Single training step"""
        x, _ = batch
        x = x.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass
        x_recon, mu, logvar = self.model(x)
        
        # Loss computation
        recon_loss = self.reconstruction_loss(x_recon, x)
        kl_loss = self.kl_divergence_loss(mu, logvar)
        total_loss = alpha * recon_loss + kl_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item()
        }
    
    @torch.no_grad()
    def eval_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                  alpha: float = 1.0) -> Dict[str, float]:
        """Single evaluation step"""
        x, _ = batch
        x = x.to(self.device)
        
        x_recon, mu, logvar = self.model(x)
        
        recon_loss = self.reconstruction_loss(x_recon, x)
        kl_loss = self.kl_divergence_loss(mu, logvar)
        total_loss = alpha * recon_loss + kl_loss
        
        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item()
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int = 100, alpha: float = 1.0) -> Dict:
        self.model.train()
        
        for epoch in range(1, epochs + 1):
            # Training phase
            train_losses = {
                'total_loss': 0.0,
                'recon_loss': 0.0,
                'kl_loss': 0.0
            }
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}')
            for batch in pbar:
                losses = self.train_step(batch, alpha=alpha)
                for key in train_losses:
                    train_losses[key] += losses[key]
                
                pbar.set_postfix({
                    'total': f"{losses['total_loss']:.4f}",
                    'recon': f"{losses['recon_loss']:.4f}",
                    'kl': f"{losses['kl_loss']:.4f}"
                })
            
            # Average losses over batches
            num_batches = len(train_loader)
            for key in train_losses:
                train_losses[key] /= num_batches
            
            # Validation phase
            self.model.eval()
            val_losses = {
                'total_loss': 0.0,
                'recon_loss': 0.0,
                'kl_loss': 0.0
            }
            
            with torch.no_grad():
                for batch in val_loader:
                    losses = self.eval_step(batch, alpha=alpha)
                    for key in val_losses:
                        val_losses[key] += losses[key]
            
            num_val_batches = len(val_loader)
            for key in val_losses:
                val_losses[key] /= num_val_batches
            
            self.model.train()
            # self.scheduler.step()
            
            # Log metrics
            if self.verbose:
                print(f"\nEpoch {epoch} | "
                    f"Train Loss: {train_losses['total_loss']:.4f} "
                    f"(recon: {train_losses['recon_loss']:.4f}, "
                    f"kl: {train_losses['kl_loss']:.4f}) | "
                    f"Val Loss: {val_losses['total_loss']:.4f} "
                    f"(recon: {val_losses['recon_loss']:.4f}, "
                    f"kl: {val_losses['kl_loss']:.4f})")
            
            for key in train_losses:
                self.training_history[key].append(train_losses[key])
            for key in val_losses:
                self.training_history[f'val_{key}'].append(val_losses[key])
        
        return self.training_history
    
    @torch.no_grad()
    def reconstruct_image(self, img: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        
        # Add batch dimension if needed
        if img.dim() == 3:
            img = img.unsqueeze(0)
        
        img = img.to(self.device)
        x_recon, _, _ = self.model(img)
        
        return x_recon.cpu()
    
    @torch.no_grad()
    def encode_image(self, img: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        
        # Add batch dimension if needed
        if img.dim() == 3:
            img = img.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        img = img.to(self.device)
        mu, _ = self.model.encode(img)
        
        if squeeze_output:
            mu = mu.squeeze(0)
        
        return mu.cpu()
    
    @torch.no_grad()
    def generate(self, num_samples: int = 1) -> torch.Tensor:
        self.model.eval()
        
        z = torch.randn(num_samples, self.model.latent_dim, 
                       device=self.device)
        generated = self.model.decode(z)
        
        return generated.cpu()
    
    def plot_training_history(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Total loss
        axes[0].plot(self.training_history['total_loss'], label='Train')
        axes[0].plot(self.training_history['val_total_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Total Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Reconstruction loss
        axes[1].plot(self.training_history['recon_loss'], label='Train')
        axes[1].plot(self.training_history['val_recon_loss'], label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Reconstruction Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        # KL loss
        axes[2].plot(self.training_history['kl_loss'], label='Train')
        axes[2].plot(self.training_history['val_kl_loss'], label='Val')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('KL Divergence Loss')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
