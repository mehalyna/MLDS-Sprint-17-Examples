"""
Task 1: Simple CNN-GAN on Fashion-MNIST
Objective: Build a small GAN that generates 28×28 Fashion-MNIST images
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
latent_size = 64
batch_size = 64
num_epochs = 20
learning_rate = 0.0002
betas = (0.5, 0.999)

# Data preprocessing - normalize to [-1, 1] for Tanh output
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Load Fashion-MNIST dataset
train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', 
    train=True, 
    transform=transform, 
    download=True
)

train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=batch_size, 
    shuffle=True
)

class Generator(nn.Module):
    """
    Generator network: latent vector (64) -> 28x28 image
    Uses ConvTranspose layers as suggested
    """
    def __init__(self, latent_size=64):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        
        # Start with a linear layer to get to the right size for conv layers
        self.linear = nn.Linear(latent_size, 7*7*128)
        
        # ConvTranspose stack
        self.conv_transpose = nn.Sequential(
            # 7x7x128 -> 14x14x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 14x14x64 -> 28x28x1
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), 128, 7, 7)  # Reshape for conv layers
        x = self.conv_transpose(x)
        return x

class Discriminator(nn.Module):
    """
    Discriminator network: 28x28 image -> real/fake probability
    Simple CNN with 2 conv layers as suggested
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # 28x28x1 -> 14x14x64
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 14x14x64 -> 7x7x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x

# Initialize networks
generator = Generator(latent_size).to(device)
discriminator = Discriminator().to(device)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=betas)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=betas)

def train_gan():
    """Training loop for the GAN"""
    generator.train()
    discriminator.train()
    
    # Lists to track losses
    D_losses = []
    G_losses = []
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        epoch_D_loss = 0.0
        epoch_G_loss = 0.0
        num_batches = 0
        
        for i, (real_images, _) in enumerate(train_loader):
            batch_size_current = real_images.size(0)
            real_images = real_images.to(device)
            
            # Create labels
            real_labels = torch.ones(batch_size_current, 1).to(device)
            fake_labels = torch.zeros(batch_size_current, 1).to(device)
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Real images
            outputs_real = discriminator(real_images)
            d_loss_real = criterion(outputs_real, real_labels)
            
            # Fake images
            noise = torch.randn(batch_size_current, latent_size).to(device)
            fake_images = generator(noise)
            outputs_fake = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs_fake, fake_labels)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()
            
            # ---------------------
            # Train Generator
            # ---------------------
            optimizer_G.zero_grad()
            
            # Generate fake images and try to fool discriminator
            outputs_fake = discriminator(fake_images)
            g_loss = criterion(outputs_fake, real_labels)  # Want discriminator to think these are real
            
            g_loss.backward()
            optimizer_G.step()
            
            # Track losses
            epoch_D_loss += d_loss.item()
            epoch_G_loss += g_loss.item()
            num_batches += 1
            
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')
        
        # Average losses for the epoch
        avg_D_loss = epoch_D_loss / num_batches
        avg_G_loss = epoch_G_loss / num_batches
        
        D_losses.append(avg_D_loss)
        G_losses.append(avg_G_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Avg D_loss: {avg_D_loss:.4f}, Avg G_loss: {avg_G_loss:.4f}')
    
    return D_losses, G_losses

def generate_and_save_images(generator, epoch=None, num_images=64):
    """Generate and save a grid of images"""
    generator.eval()
    
    with torch.no_grad():
        # Generate random noise
        noise = torch.randn(num_images, latent_size).to(device)
        fake_images = generator(noise)
        
        # Denormalize images from [-1, 1] to [0, 1]
        fake_images = (fake_images + 1) / 2.0
        
        # Create grid
        grid = torchvision.utils.make_grid(fake_images, nrow=8, normalize=True)
        
        # Convert to numpy and plot
        grid_np = grid.cpu().numpy().transpose(1, 2, 0)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_np, cmap='gray' if grid_np.shape[2] == 1 else None)
        plt.axis('off')
        plt.title('Generated Fashion-MNIST Images')
        
        # Save the image
        filename = 'fashion_generated.png'
        if epoch is not None:
            filename = f'fashion_generated_epoch_{epoch}.png'
        
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        print(f"Generated images saved as {filename}")
        plt.show()

def plot_losses(D_losses, G_losses):
    """Plot training losses"""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(D_losses) + 1), D_losses, label='Discriminator Loss', color='blue')
    plt.plot(range(1, len(G_losses) + 1), G_losses, label='Generator Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_losses.png', bbox_inches='tight', dpi=150)
    plt.show()

if __name__ == "__main__":
    # Train the GAN
    print("Training Fashion-MNIST GAN...")
    D_losses, G_losses = train_gan()
    
    # Generate and save final images
    print("\nGenerating final images...")
    generate_and_save_images(generator)
    
    # Plot losses
    print("\nPlotting training losses...")
    plot_losses(D_losses, G_losses)
    
    # Print final losses
    print(f"\nFinal Discriminator Loss: {D_losses[-1]:.4f}")
    print(f"Final Generator Loss: {G_losses[-1]:.4f}")
    
    # Save models
    torch.save(generator.state_dict(), 'generator_fashion_mnist.pth')
    torch.save(discriminator.state_dict(), 'discriminator_fashion_mnist.pth')
    print("\nModels saved as generator_fashion_mnist.pth and discriminator_fashion_mnist.pth")