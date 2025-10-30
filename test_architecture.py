import torch
import torch.nn as nn

# Test different architectures to get exact 28x28 output
def test_architecture():
    print("Testing different ConvTranspose2d configurations for 28x28 output...")
    
    # Test with a simple input
    x = torch.randn(1, 128, 3, 3)
    
    # Option 1: 3x3 -> 7x7 -> 14x14 -> 28x28
    decoder1 = nn.Sequential(
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=0),  # 3->8, but we want 7
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 8->14
        nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),    # 14->28
    )
    
    # Option 2: More precise control
    decoder2 = nn.Sequential(
        nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=1),  # 3->7
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 7->14
        nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),    # 14->28
    )
    
    # Option 3: Using interpolation for exact control
    decoder3 = nn.Sequential(
        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),  # Keep 3x3
        nn.Upsample(size=(7, 7), mode='bilinear', align_corners=False),
        nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
        nn.Upsample(size=(14, 14), mode='bilinear', align_corners=False),
        nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
        nn.Upsample(size=(28, 28), mode='bilinear', align_corners=False),
    )
    
    print("Testing decoder options:")
    
    try:
        out1 = decoder1(x)
        print(f"Option 1 output shape: {out1.shape}")
    except Exception as e:
        print(f"Option 1 failed: {e}")
    
    try:
        out2 = decoder2(x)
        print(f"Option 2 output shape: {out2.shape}")
    except Exception as e:
        print(f"Option 2 failed: {e}")
    
    try:
        out3 = decoder3(x)
        print(f"Option 3 output shape: {out3.shape}")
    except Exception as e:
        print(f"Option 3 failed: {e}")

if __name__ == "__main__":
    test_architecture()