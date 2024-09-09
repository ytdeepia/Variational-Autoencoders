from torch import nn
from abc import abstractmethod
from typing import List, Any, TypeVar
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, utils
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import torch.nn.functional as F

Tensor = TypeVar("torch.tensor")


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class VanillaVAE(BaseVAE):

    def __init__(
        self, in_channels: int, latent_dim: int, hidden_dims: List = None, **kwargs
    ) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        loss = recons_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": -kld_loss.detach(),
        }

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:

        return self.forward(x)[0]


# Define paths
data_dir = "../../data/celeba/img_align_celeba"
weights_dir = "./weights/"
results_dir = "./results/"
reconstructions = results_dir + "reconstructions/"
samples = results_dir + "samples/"
log_dir = "./logs/"

# Create directories if not exist
os.makedirs(weights_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Hyperparameters
batch_size = 256
learning_rate = 0.005
weight_decay = 0.0
scheduler_gamma = 0.95
num_epochs = 100
latent_dim = 128
img_size = 64
kld_weight = 0.00025  # weight of KL divergence in the loss

torch.manual_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations
transform = transforms.Compose(
    [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]
)

# Load dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Limit dataset size
# train_dataset = Subset(dataset, range(100000))  # Use only 100 images for training
# val_dataset = Subset(dataset, range(100000, 101000))  # Use 10 images for validation

# Split dataset into train and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=10
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)

# Initialize model, optimizer, and loss function
model = VanillaVAE(in_channels=3, latent_dim=latent_dim).to(device)
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)
writer = SummaryWriter(log_dir)

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, _ = batch
        images = images.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss_dict = model.loss_function(*outputs, M_N=kld_weight)
        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()

    avg_train_loss = train_loss / len(train_loader)
    writer.add_scalar("Loss/Train", avg_train_loss, epoch)

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images, _ = batch
            images = images.to(device)

            outputs = model(images)
            loss_dict = model.loss_function(*outputs, M_N=kld_weight)
            val_loss += loss_dict["loss"].item()

    avg_val_loss = val_loss / len(val_loader)
    writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
    writer.add_scalar("Loss/Reconstruction", loss_dict["Reconstruction_Loss"], epoch)
    writer.add_scalar("Loss/KLD", loss_dict["KLD"], epoch)

    if epoch % 10 == 0:
        # Save model checkpoint
        torch.save(
            model.state_dict(), os.path.join(weights_dir, f"vae_epoch_{epoch+1}.pth")
        )

    # Generate and save examples
    with torch.no_grad():
        sample_images, _ = next(iter(val_loader))
        sample_images = sample_images.to(device)
        reconstructed_images, _, _, _ = model(sample_images)
        comparison = torch.cat([sample_images[:8], reconstructed_images[:8]])
        utils.save_image(
            comparison.cpu(),
            os.path.join(results_dir, f"reconstruction_epoch_{epoch+1}.png"),
            nrow=8,
            normalize=True,
        )

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
    )


# Close TensorBoard writer
writer.close()
