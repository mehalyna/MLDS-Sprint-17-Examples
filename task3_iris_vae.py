"""
Task 3: Simple VAE on Iris Dataset
Objective: Train a VAE to compress Iris dataset into 2 latent dimensions and visualize latent space
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameters
input_size = 4      # Iris features
hidden_size = 16    # Hidden layer size
latent_size = 2     # Latent dimension
learning_rate = 0.001
num_epochs = 200
batch_size = 32

class VAE(nn.Module):
    """
    Variational Autoencoder for Iris dataset
    Encoder: 4 -> 16 -> mu/log_var (size 2)
    Decoder: 2 -> 16 -> 4
    """
    def __init__(self, input_size=4, hidden_size=16, latent_size=2):
        super(VAE, self).__init__()
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        
        # Mean and log variance layers
        self.mu_layer = nn.Linear(hidden_size, latent_size)
        self.logvar_layer = nn.Linear(hidden_size, latent_size)
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
        
    def encode(self, x):
        """Encode input to mean and log variance"""
        h = self.encoder(x)
        mu = self.mu_layer(h)
        log_var = self.logvar_layer(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick: z = mu + eps * exp(0.5*log_var)"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """Decode latent variable to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through VAE"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var, z

def vae_loss(reconstruction, x, mu, log_var):
    """
    VAE loss = Reconstruction loss (MSE) + KL divergence
    """
    # Reconstruction loss (MSE)
    recon_loss = nn.MSELoss()(reconstruction, x)
    
    # KL divergence loss
    # KL(q(z|x) || p(z)) where p(z) = N(0,I)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl_loss = kl_loss / x.size(0)  # Average over batch
    
    return recon_loss + kl_loss, recon_loss, kl_loss

def load_and_preprocess_data():
    """Load and preprocess Iris dataset"""
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler

def create_data_loader(X, batch_size):
    """Create data loader for training"""
    dataset = torch.utils.data.TensorDataset(X)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_vae(model, train_loader, optimizer, num_epochs):
    """Train the VAE"""
    model.train()
    
    # Track losses
    total_losses = []
    recon_losses = []
    kl_losses = []
    
    print("Starting VAE training...")
    
    for epoch in range(num_epochs):
        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data,) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass
            reconstruction, mu, log_var, z = model(data)
            
            # Calculate loss
            total_loss, recon_loss, kl_loss = vae_loss(reconstruction, data, mu, log_var)
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Track losses
            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            num_batches += 1
        
        # Average losses for the epoch
        avg_total_loss = epoch_total_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches
        
        total_losses.append(avg_total_loss)
        recon_losses.append(avg_recon_loss)
        kl_losses.append(avg_kl_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Total Loss: {avg_total_loss:.4f}, '
                  f'Recon Loss: {avg_recon_loss:.4f}, '
                  f'KL Loss: {avg_kl_loss:.4f}')
    
    return total_losses, recon_losses, kl_losses

def visualize_latent_space(model, X_data, y_data, title="Latent Space Visualization"):
    """Visualize the 2D latent space colored by species"""
    model.eval()
    
    with torch.no_grad():
        mu, log_var = model.encode(X_data)
        # Use mean of latent distribution for visualization
        latent_codes = mu.cpu().numpy()
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    
    # Define colors and labels for each species
    species_names = ['Setosa', 'Versicolor', 'Virginica']
    colors = ['red', 'blue', 'green']
    
    for i in range(3):
        mask = y_data.cpu().numpy() == i
        plt.scatter(latent_codes[mask, 0], latent_codes[mask, 1], 
                   c=colors[i], label=species_names[i], alpha=0.7, s=60)
    
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    filename = 'iris_vae_latent_space.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Latent space visualization saved as {filename}")
    plt.show()
    
    return latent_codes

def plot_training_losses(total_losses, recon_losses, kl_losses):
    """Plot training losses over epochs"""
    epochs = range(1, len(total_losses) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # Total loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, total_losses, 'b-', linewidth=2)
    plt.title('Total VAE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # Reconstruction loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, recon_losses, 'r-', linewidth=2)
    plt.title('Reconstruction Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # KL divergence loss
    plt.subplot(1, 3, 3)
    plt.plot(epochs, kl_losses, 'g-', linewidth=2)
    plt.title('KL Divergence Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('iris_vae_training_losses.png', dpi=150, bbox_inches='tight')
    plt.show()

def evaluate_reconstruction(model, X_data, scaler):
    """Evaluate reconstruction quality"""
    model.eval()
    
    with torch.no_grad():
        reconstruction, _, _, _ = model(X_data)
        
        # Calculate MSE in original scale
        X_original = scaler.inverse_transform(X_data.cpu().numpy())
        X_recon = scaler.inverse_transform(reconstruction.cpu().numpy())
        
        mse = np.mean((X_original - X_recon) ** 2)
        
    return mse, X_original, X_recon

def main():
    """Main function to run the VAE experiment"""
    print("Loading and preprocessing Iris dataset...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    
    # Combine train and test for full dataset visualization
    X_full = torch.cat([X_train, X_test], dim=0)
    y_full = torch.cat([y_train, y_test], dim=0)
    
    print(f"Dataset shape: {X_full.shape}")
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create data loader
    train_loader = create_data_loader(X_train, batch_size)
    
    # Initialize model
    model = VAE(input_size, hidden_size, latent_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\nModel architecture:")
    print(f"Input size: {input_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"Latent size: {latent_size}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train the model
    total_losses, recon_losses, kl_losses = train_vae(model, train_loader, optimizer, num_epochs)
    
    # Plot training losses
    print("\nPlotting training losses...")
    plot_training_losses(total_losses, recon_losses, kl_losses)
    
    # Visualize latent space
    print("\nVisualizing latent space...")
    latent_codes = visualize_latent_space(model, X_full, y_full)
    
    # Evaluate reconstruction
    print("\nEvaluating reconstruction quality...")
    train_mse, X_train_orig, X_train_recon = evaluate_reconstruction(model, X_train, scaler)
    test_mse, X_test_orig, X_test_recon = evaluate_reconstruction(model, X_test, scaler)
    
    print(f"Training reconstruction MSE: {train_mse:.6f}")
    print(f"Test reconstruction MSE: {test_mse:.6f}")
    
    # Print some reconstruction examples
    print("\nReconstruction examples (first 5 test samples):")
    print("Original vs Reconstructed features:")
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    
    for i in range(min(5, len(X_test_orig))):
        print(f"\nSample {i+1}:")
        for j, feature in enumerate(feature_names):
            print(f"  {feature}: {X_test_orig[i,j]:.3f} -> {X_test_recon[i,j]:.3f}")
    
    # Save the model
    torch.save(model.state_dict(), 'iris_vae_model.pth')
    print(f"\nModel saved as iris_vae_model.pth")
    
    # Final summary
    print(f"\n=== VAE Training Summary ===")
    print(f"Dataset: Iris (150 samples, 4 features)")
    print(f"Latent dimensions: {latent_size}")
    print(f"Training epochs: {num_epochs}")
    print(f"Final total loss: {total_losses[-1]:.4f}")
    print(f"Final reconstruction loss: {recon_losses[-1]:.4f}")
    print(f"Final KL loss: {kl_losses[-1]:.4f}")
    print(f"Test reconstruction MSE: {test_mse:.6f}")

if __name__ == "__main__":
    main()