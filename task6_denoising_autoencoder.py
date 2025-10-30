"""
Task 6: Denoising Autoencoder on Fashion-MNIST
Objective: Train a denoising autoencoder that removes additive Gaussian noise from Fashion-MNIST images
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameters
batch_size = 128
num_epochs = 15
learning_rate = 0.001
noise_std = 0.3  # Standard deviation for Gaussian noise

# Data preprocessing - normalize to [0, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    # Keep values in [0, 1] for easier noise handling
])

# Load Fashion-MNIST dataset
train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', 
    train=True, 
    transform=transform, 
    download=True
)

test_dataset = torchvision.datasets.FashionMNIST(
    root='./data', 
    train=False, 
    transform=transform, 
    download=True
)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Fashion-MNIST class names for visualization
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def add_noise(images, noise_std=0.3):
    """Add Gaussian noise to images and clip to valid range [0, 1]"""
    noise = torch.randn_like(images) * noise_std
    noisy_images = images + noise
    # Clip to keep pixel values in valid range [0, 1]
    noisy_images = torch.clamp(noisy_images, 0.0, 1.0)
    return noisy_images

class DenoisingAutoencoder(nn.Module):
    """
    Convolutional Denoising Autoencoder for Fashion-MNIST
    Encoder: conv→pool layers to compress
    Decoder: convtranspose layers to reconstruct
    """
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder: 28x28x1 -> compressed representation
        self.encoder = nn.Sequential(
            # 28x28x1 -> 14x14x32
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # 14x14x32 -> 7x7x64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # 7x7x64 -> 4x4x128 (3x3 -> 3x3 after padding, then pool to get smaller)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 3)),  # Force to 3x3 for symmetric reconstruction
        )
        
        # Decoder: compressed representation -> 28x28x1
        self.decoder = nn.Sequential(
            # 3x3x128 -> 7x7x64
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # 7x7x64 -> 14x14x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # 14x14x32 -> 28x28x1
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        # Decode
        decoded = self.decoder(encoded)
        return decoded

def train_denoising_autoencoder(model, train_loader, optimizer, criterion, num_epochs, noise_std):
    """Train the denoising autoencoder"""
    model.train()
    
    # Track losses
    train_losses = []
    
    print("Starting denoising autoencoder training...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (clean_images, _) in enumerate(train_loader):
            clean_images = clean_images.to(device)
            
            # Add noise to create noisy inputs
            noisy_images = add_noise(clean_images, noise_std)
            
            optimizer.zero_grad()
            
            # Forward pass: denoise the noisy images
            reconstructed = model(noisy_images)
            
            # Loss: MSE between reconstructed and clean images
            loss = criterion(reconstructed, clean_images)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.6f}')
        
        # Average loss for the epoch
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Average Loss: {avg_loss:.6f}')
    
    return train_losses

def evaluate_denoising(model, test_loader, noise_std, num_examples=10):
    """Evaluate denoising quality and return examples"""
    model.eval()
    
    examples = []
    total_loss = 0.0
    num_batches = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch_idx, (clean_images, labels) in enumerate(test_loader):
            clean_images = clean_images.to(device)
            
            # Add noise
            noisy_images = add_noise(clean_images, noise_std)
            
            # Reconstruct
            reconstructed = model(noisy_images)
            
            # Calculate loss
            loss = criterion(reconstructed, clean_images)
            total_loss += loss.item()
            num_batches += 1
            
            # Collect examples for visualization
            if len(examples) < num_examples:
                for i in range(min(num_examples - len(examples), clean_images.size(0))):
                    examples.append({
                        'clean': clean_images[i].cpu(),
                        'noisy': noisy_images[i].cpu(),
                        'reconstructed': reconstructed[i].cpu(),
                        'label': labels[i].item()
                    })
            
            if len(examples) >= num_examples:
                break
    
    avg_test_loss = total_loss / num_batches
    return examples, avg_test_loss

def visualize_denoising_results(examples, save_path='denoising_results.png'):
    """
    Visualize denoising results: noisy input → reconstructed output → original
    """
    num_examples = len(examples)
    fig, axes = plt.subplots(3, num_examples, figsize=(2*num_examples, 6))
    
    if num_examples == 1:
        axes = axes.reshape(-1, 1)
    
    for i, example in enumerate(examples):
        clean_img = example['clean'].squeeze().numpy()
        noisy_img = example['noisy'].squeeze().numpy()
        recon_img = example['reconstructed'].squeeze().numpy()
        label = example['label']
        
        # Noisy input (top row)
        axes[0, i].imshow(noisy_img, cmap='gray')
        axes[0, i].set_title(f'Noisy\n{class_names[label]}', fontsize=10)
        axes[0, i].axis('off')
        
        # Reconstructed (middle row)
        axes[1, i].imshow(recon_img, cmap='gray')
        axes[1, i].set_title('Reconstructed', fontsize=10)
        axes[1, i].axis('off')
        
        # Original clean (bottom row)
        axes[2, i].imshow(clean_img, cmap='gray')
        axes[2, i].set_title('Original', fontsize=10)
        axes[2, i].axis('off')
    
    # Add row labels
    fig.text(0.02, 0.83, 'Noisy Input', fontsize=12, fontweight='bold', rotation=90, va='center')
    fig.text(0.02, 0.5, 'Reconstructed', fontsize=12, fontweight='bold', rotation=90, va='center')
    fig.text(0.02, 0.17, 'Original Clean', fontsize=12, fontweight='bold', rotation=90, va='center')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.08)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Denoising results saved as {save_path}")
    plt.show()

def plot_training_loss(train_losses):
    """Plot training loss over epochs"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=2, marker='o')
    plt.title('Denoising Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('denoising_training_loss.png', dpi=150, bbox_inches='tight')
    plt.show()

def calculate_metrics(examples):
    """Calculate denoising quality metrics"""
    mse_noisy_vs_clean = 0.0
    mse_recon_vs_clean = 0.0
    
    for example in examples:
        clean = example['clean'].numpy()
        noisy = example['noisy'].numpy()
        recon = example['reconstructed'].numpy()
        
        mse_noisy_vs_clean += np.mean((noisy - clean) ** 2)
        mse_recon_vs_clean += np.mean((recon - clean) ** 2)
    
    mse_noisy_vs_clean /= len(examples)
    mse_recon_vs_clean /= len(examples)
    
    improvement = mse_noisy_vs_clean - mse_recon_vs_clean
    improvement_percent = (improvement / mse_noisy_vs_clean) * 100
    
    return mse_noisy_vs_clean, mse_recon_vs_clean, improvement_percent

def main():
    """Main function to run the denoising autoencoder experiment"""
    print("Starting Fashion-MNIST Denoising Autoencoder Experiment")
    print(f"Noise standard deviation: {noise_std}")
    print(f"Training epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    # Initialize model
    model = DenoisingAutoencoder().to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\nModel architecture:")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train the model
    train_losses = train_denoising_autoencoder(
        model, train_loader, optimizer, criterion, num_epochs, noise_std
    )
    
    # Plot training loss
    print("\nPlotting training loss...")
    plot_training_loss(train_losses)
    
    # Evaluate on test set
    print("\nEvaluating denoising quality...")
    examples, avg_test_loss = evaluate_denoising(model, test_loader, noise_std, num_examples=8)
    
    print(f"Average test reconstruction loss: {avg_test_loss:.6f}")
    
    # Calculate and display metrics
    mse_noisy, mse_recon, improvement_percent = calculate_metrics(examples)
    
    print(f"\nDenoising Quality Metrics:")
    print(f"MSE (Noisy vs Clean): {mse_noisy:.6f}")
    print(f"MSE (Reconstructed vs Clean): {mse_recon:.6f}")
    print(f"Improvement: {improvement_percent:.2f}%")
    
    # Visualize results
    print("\nGenerating denoising visualization...")
    visualize_denoising_results(examples)
    
    # Save the model
    torch.save(model.state_dict(), 'denoising_autoencoder_fashion_mnist.pth')
    print("\nModel saved as denoising_autoencoder_fashion_mnist.pth")
    
    # Denoising quality assessment
    print(f"\n=== Denoising Quality Assessment ===")
    if improvement_percent > 70:
        quality = "Excellent"
    elif improvement_percent > 50:
        quality = "Good"
    elif improvement_percent > 30:
        quality = "Fair"
    else:
        quality = "Poor"
    
    print(f"Overall denoising quality: {quality}")
    print(f"The autoencoder successfully reduced noise-related MSE by {improvement_percent:.1f}%")
    
    if improvement_percent > 50:
        print("The model demonstrates strong denoising capabilities, effectively removing")
        print("Gaussian noise while preserving important image features and details.")
    else:
        print("The model shows basic denoising ability but may benefit from:")
        print("- More training epochs")
        print("- Different architecture (deeper network)")
        print("- Adjusted noise levels during training")
    
    print(f"\nFinal training loss: {train_losses[-1]:.6f}")
    print(f"Final test loss: {avg_test_loss:.6f}")

if __name__ == "__main__":
    main()