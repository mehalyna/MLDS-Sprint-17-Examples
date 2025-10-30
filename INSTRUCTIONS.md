# MLDS Sprint 17 Examples Setup and Run Instructions

This repository contains implementations for 6 machine learning tasks involving GANs, VAEs, and Autoencoders. Follow these instructions to set up your environment and run the solutions.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Package Installation](#package-installation)
- [Task Solutions](#task-solutions)
  - [Task 1: Fashion-MNIST GAN](#task-1-fashion-mnist-gan)
  - [Task 3: Iris VAE](#task-3-iris-vae)
  - [Task 6: Fashion-MNIST Denoising Autoencoder](#task-6-fashion-mnist-denoising-autoencoder)
- [Expected Outputs](#expected-outputs)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- **Python**: Version 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **RAM**: At least 4GB (8GB+ recommended for faster training)
- **Storage**: At least 2GB free space for datasets and outputs

## Environment Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd MLDS-Sprint-17-Examples
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
venv\Scripts\activate.bat

# On macOS/Linux:
source venv/bin/activate
```

### 3. Verify Environment
You should see `(venv)` at the beginning of your terminal prompt, indicating the virtual environment is active.

## Package Installation

Install all required packages:

```bash
pip install torch torchvision matplotlib numpy scikit-learn seaborn
```

### Package Versions (Tested)
- `torch`: 2.0+
- `torchvision`: 0.15+
- `matplotlib`: 3.5+
- `numpy`: 1.20+
- `scikit-learn`: 1.0+
- `seaborn`: 0.11+

## Task Solutions

### Task 1: Fashion-MNIST GAN

**File**: `task1_fashion_mnist_gan.py`

**Description**: Generates Fashion-MNIST images using a Generative Adversarial Network.

**Run Command**:
```bash
python task1_fashion_mnist_gan.py
```

**Training Details**:
- **Duration**: ~15-30 minutes (20 epochs)
- **Dataset**: Fashion-MNIST (auto-downloaded)
- **Architecture**: CNN-based Generator and Discriminator
- **Output Size**: 28×28 grayscale images

**Generated Files**:
- `fashion_generated.png` - Grid of generated fashion items
- `training_losses.png` - Loss curves
- `generator_fashion_mnist.pth` - Trained generator model
- `discriminator_fashion_mnist.pth` - Trained discriminator model

---

### Task 3: Iris VAE

**File**: `task3_iris_vae.py`

**Description**: Compresses Iris dataset into 2D latent space using Variational Autoencoder.

**Run Command**:
```bash
python task3_iris_vae.py
```

**Training Details**:
- **Duration**: ~2-5 minutes (200 epochs)
- **Dataset**: Iris (150 samples, 4 features)
- **Architecture**: MLP-based Encoder/Decoder
- **Latent Dimensions**: 2D for visualization

**Generated Files**:
- `iris_vae_latent_space.png` - 2D scatter plot colored by species
- `iris_vae_training_losses.png` - Training progress
- `iris_vae_model.pth` - Trained VAE model

---

### Task 6: Fashion-MNIST Denoising Autoencoder

**File**: `task6_denoising_autoencoder.py`

**Description**: Removes Gaussian noise from Fashion-MNIST images using a denoising autoencoder.

**Run Command**:
```bash
python task6_denoising_autoencoder.py
```

**Training Details**:
- **Duration**: ~10-20 minutes (15 epochs)
- **Dataset**: Fashion-MNIST with added Gaussian noise (σ=0.3)
- **Architecture**: Convolutional Autoencoder
- **Task**: Noise removal and image reconstruction

**Generated Files**:
- `denoising_results.png` - Visual comparison (noisy → reconstructed → original)
- `denoising_training_loss.png` - Training loss curve
- `denoising_autoencoder_fashion_mnist.pth` - Trained model

## Expected Outputs

### Task 1 Fashion-MNIST GAN
- **Generated Images**: Realistic clothing items (shirts, shoes, bags, etc.)
- **Final Generator Loss**: ~1.0-3.0
- **Final Discriminator Loss**: ~0.5-1.5
- **Training Progress**: Loss oscillations indicating adversarial training

### Task 3 Iris VAE
- **Latent Space**: Clear clustering of 3 Iris species in 2D space
- **Reconstruction MSE**: <0.1 (very low reconstruction error)
- **Species Separation**: Setosa clearly separated, some overlap between Versicolor/Virginica

### Task 6 Denoising Autoencoder
- **Noise Reduction**: 50-80% improvement in image quality
- **Visual Quality**: Clear removal of Gaussian noise while preserving details
- **Reconstruction Loss**: Decreasing trend over epochs

## Troubleshooting

### Common Issues and Solutions

#### 1. **Virtual Environment Not Activated**
**Error**: `ModuleNotFoundError` for installed packages
**Solution**: 
```bash
# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Check activation
python -c "import sys; print(sys.prefix)"
```

#### 2. **CUDA/GPU Issues**
**Error**: CUDA-related errors
**Solution**: The code automatically falls back to CPU. For GPU usage:
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

#### 3. **Memory Issues**
**Error**: `RuntimeError: out of memory`
**Solutions**:
- Reduce batch size in the code (e.g., change `batch_size = 64` to `batch_size = 32`)
- Close other applications
- Use CPU instead of GPU

#### 4. **Dataset Download Issues**
**Error**: Dataset download failures
**Solutions**:
- Check internet connection
- Delete partially downloaded data in `./data` folder and retry
- Use VPN if network restrictions exist

#### 5. **Import Errors**
**Error**: `ImportError` or `ModuleNotFoundError`
**Solutions**:
```bash
# Reinstall packages
pip uninstall torch torchvision matplotlib numpy scikit-learn seaborn
pip install torch torchvision matplotlib numpy scikit-learn seaborn

# Check installed packages
pip list
```

#### 6. **Dimension Mismatch Errors**
**Error**: Tensor size mismatch
**Solution**: This should be fixed in the provided code. If it persists:
- Ensure you're using the latest version of the files
- Check PyTorch version compatibility

### Performance Tips

#### 1. **Faster Training**
- Use GPU if available: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- Reduce number of epochs for testing
- Use smaller batch sizes if memory is limited

#### 2. **Better Results**
- Increase number of epochs for better convergence
- Adjust learning rates if training is unstable
- Monitor loss curves to ensure proper training

### File Permissions (Windows)
If you encounter permission errors:
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Project Structure

After running all solutions, your directory should contain:

```
MLDS-Sprint-17-Examples/
├── README.md
├── INSTRUCTIONS.md
├── venv/                          # Virtual environment
├── data/                          # Auto-created datasets
│   ├── FashionMNIST/
│   └── ...
├── task1_fashion_mnist_gan.py     # Task 1 solution
├── task3_iris_vae.py              # Task 3 solution
├── task6_denoising_autoencoder.py # Task 6 solution
├── test_architecture.py           # Helper script
├── fashion_generated.png          # Task 1 output
├── training_losses.png            # Task 1 output
├── iris_vae_latent_space.png      # Task 3 output
├── iris_vae_training_losses.png   # Task 3 output
├── denoising_results.png          # Task 6 output
├── denoising_training_loss.png    # Task 6 output
└── *.pth                          # Trained model files
```

## Learning Outcomes

After completing these tasks, you will have:

1. **GAN Understanding**: Practical experience with adversarial training
2. **VAE Knowledge**: Variational inference and latent space visualization
3. **Autoencoder Skills**: Noise removal and feature reconstruction
4. **PyTorch Proficiency**: Model building, training, and evaluation
5. **Deep Learning Workflow**: Complete ML pipeline from data to results

## Support

If you encounter issues not covered in this guide:

1. Check the console output for specific error messages
2. Verify all packages are installed correctly
3. Ensure virtual environment is activated
4. Review the troubleshooting section above
5. Check PyTorch and Python versions compatibility

## Quick Start Summary

```bash
# 1. Setup
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
pip install torch torchvision matplotlib numpy scikit-learn seaborn

# 2. Run tasks
python task1_fashion_mnist_gan.py      # ~20 minutes
python task3_iris_vae.py               # ~3 minutes  
python task6_denoising_autoencoder.py  # ~15 minutes

# 3. Check outputs
# Look for generated .png files and console output
```

---

**Happy Learning!** 