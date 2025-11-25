import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets, transforms
from torch import optim
import utils
from torch.autograd import Variable
import visual


class VAE(nn.Module):
    def __init__(self, label, image_size, channel_num, kernel_num, z_size):
        # configurations
        super().__init__()
        self.label = label
        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.z_size = z_size

        # encoder
        self.encoder = nn.Sequential(
            self._conv(channel_num, kernel_num // 4),
            self._conv(kernel_num // 4, kernel_num // 2),
            self._conv(kernel_num // 2, kernel_num),
        )

        # encoded feature's size and volume
        self.feature_size = image_size // 8
        self.feature_volume = kernel_num * (self.feature_size ** 2)

        # q
        self.q_mean = self._linear(self.feature_volume, z_size, relu=False)
        self.q_logvar = self._linear(self.feature_volume, z_size, relu=False)

        # projection
        self.project = self._linear(z_size, self.feature_volume, relu=False)

        # decoder
        self.decoder = nn.Sequential(
            self._deconv(kernel_num, kernel_num // 2),
            self._deconv(kernel_num // 2, kernel_num // 4),
            self._deconv(kernel_num // 4, channel_num),
            nn.Sigmoid()
        )

    def forward(self, x):
        # encode x
        encoded = self.encoder(x)

        # sample latent code z from q given x.
        mean, logvar = self.q(encoded)
        z = self.z(mean, logvar)
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )

        # reconstruct x from z
        x_reconstructed = self.decoder(z_projected)

        # return the parameters of distribution of q given x and the
        # reconstructed image.
        return (mean, logvar), x_reconstructed

    # ==============
    # VAE components
    # ==============

    def q(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_mean(unrolled), self.q_logvar(unrolled)

    def z(self, mean, logvar):
        # logvar = torch.clamp(logvar, min=-20, max=20)
        std = logvar.mul(0.5).exp_()
        eps = (
            Variable(torch.randn(std.size())).cuda() if self._is_on_cuda else
            Variable(torch.randn(std.size()))
        )
        return eps.mul(std).add_(mean)

    def reconstruction_loss(self, x_reconstructed, x):
        return nn.BCELoss(size_average=False)(x_reconstructed, x) / x.size(0)

    def kl_divergence_loss(self, mean, logvar):
        return ((mean**2 + logvar.exp() - 1 - logvar) / 2).mean()

    # =====
    # Utils
    # =====

    @property
    def name(self):
        return (
            'VAE'
            '-{kernel_num}k'
            '-{label}'
            '-{channel_num}x{image_size}x{image_size}'
        ).format(
            label=self.label,
            kernel_num=self.kernel_num,
            image_size=self.image_size,
            channel_num=self.channel_num,
        )

    def sample(self, size):
        z = Variable(
            torch.randn(size, self.z_size).cuda() if self._is_on_cuda() else
            torch.randn(size, self.z_size)
        )
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )
        return self.decoder(z_projected).data

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    # ======
    # Layers
    # ======

    def _conv(self, channel_size, kernel_num):
        return nn.Sequential(
            nn.Conv2d(
                channel_size, kernel_num,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _deconv(self, channel_num, kernel_num):
        return nn.Sequential(
            nn.ConvTranspose2d(
                channel_num, kernel_num,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _linear(self, in_size, out_size, relu=True):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
        ) if relu else nn.Linear(in_size, out_size)

# ============================================================


class LatentSpace:
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = VAE(
            label="KUKU",
            image_size=32,
            channel_num=3,
            kernel_num=128,
            z_size=128,
        ).to(self.device)
        self.training_history = {
            'total_loss': [],
            'recon_loss': [],
            'kld_loss': []
        }

    def train_model(self, dataset, epochs=10,
                    batch_size=32, sample_size=32,
                    lr=3e-04, weight_decay=1e-5,
                    loss_log_interval=30,
                    image_log_interval=300,
                    checkpoint_dir='./checkpoints',
                    resume=False,
                    cuda=False):
        # prepare optimizer and model
        self.model.train()
        optimizer = optim.Adam(
            self.model.parameters(), lr=lr,
            weight_decay=weight_decay,
        )
    
        if resume:
            epoch_start = utils.load_checkpoint(self.model, checkpoint_dir)
        else:
            epoch_start = 1
    
        for epoch in range(epoch_start, epochs+1):
            data_loader = utils.get_data_loader(dataset, batch_size, cuda=cuda)
            data_stream = tqdm(enumerate(data_loader, 1))
    
            for batch_index, (x, _) in data_stream:
                # where are we?
                iteration = (epoch-1)*(len(dataset)//batch_size) + batch_index
    
                # prepare data on gpu if needed
                x = Variable(x).cuda() if cuda else Variable(x)
    
                # flush gradients and run the model forward
                optimizer.zero_grad()
                (mean, logvar), x_reconstructed = self.model(x)
                reconstruction_loss = self.model.reconstruction_loss(x_reconstructed, x)
                kl_divergence_loss = self.model.kl_divergence_loss(mean, logvar)
                total_loss = reconstruction_loss + kl_divergence_loss
    
                # backprop gradients from the loss
                total_loss.backward()
                optimizer.step()
    
                # update progress
                data_stream.set_description((
                    'epoch: {epoch} | '
                    'iteration: {iteration} | '
                    'progress: [{trained}/{total}] ({progress:.0f}%) | '
                    'loss => '
                    'total: {total_loss:.4f} / '
                    're: {reconstruction_loss:.3f} / '
                    'kl: {kl_divergence_loss:.3f}'
                ).format(
                    epoch=epoch,
                    iteration=iteration,
                    trained=batch_index * len(x),
                    total=len(data_loader.dataset),
                    progress=(100. * batch_index / len(data_loader)),
                    total_loss=total_loss.item(),
                    reconstruction_loss=reconstruction_loss.item(),
                    kl_divergence_loss=kl_divergence_loss.item(),
                ))
    
                if iteration % loss_log_interval == 0:
                    losses = [
                        reconstruction_loss.item(),
                        kl_divergence_loss.item(),
                        total_loss.item()
                    ]
                    names = ['reconstruction', 'kl divergence', 'total']
                    visual.visualize_scalars(
                        losses, names, 'loss',
                        iteration, env=self.model.name)
    
                if iteration % image_log_interval == 0:
                    images = self.model.sample(sample_size)
                    visual.visualize_images(
                        images, 'generated samples',
                        env=self.model.name
                    )

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
    
        self.model.load_state_dict(checkpoint['state'])
        epoch = checkpoint['epoch']
        return epoch


    def generate(self, num_samples=1):
        return self.model.sample(num_samples)

    def reconstruct_image(self, image_tensor):
        self.model.eval()
        
        # Добавляем batch dimension если нужно
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Перемещаем на device
        x = Variable(image_tensor).cuda() if self.device == 'cuda' else Variable(image_tensor)
        
        with torch.no_grad():
            (mean, logvar), x_reconstructed = self.model(x)
        
        return x_reconstructed.cpu()



