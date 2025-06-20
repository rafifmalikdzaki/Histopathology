import torch
import matplotlib.pyplot as plt
from histopathology.models.autoencoders.dae_kan_attention.model import DAE_KAN_Attention

def test_model():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create sample input (batch_size=2, channels=3, height=128, width=128)
    x = torch.rand(2, 3, 128, 128).to('cuda')
    print(f"Input shape: {x.shape}")
    print(f"Input value range: [{x.min():.3f}, {x.max():.3f}]")
    
    # Initialize model
    model = DAE_KAN_Attention(device='cuda')
    model = model.to('cuda')
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        encoded, decoded, z = model(x)
        
        # Print shapes and value ranges
        print("\nIntermediate tensor shapes and ranges:")
        print(f"Encoded shape: {encoded.shape}")
        print(f"Encoded value range: [{encoded.min():.3f}, {encoded.max():.3f}]")
        print(f"Bottleneck (z) shape: {z.shape}")
        print(f"Bottleneck value range: [{z.min():.3f}, {z.max():.3f}]")
        print(f"Decoded shape: {decoded.shape}")
        print(f"Decoded value range: [{decoded.min():.3f}, {decoded.max():.3f}]")
        
        # Verify output is in [0,1] range
        assert decoded.min() >= 0 and decoded.max() <= 1, \
            f"Output values outside [0,1] range: [{decoded.min():.3f}, {decoded.max():.3f}]"
        
        # Visualize first image from batch
        plt.figure(figsize=(10, 5))
        
        # Original image
        plt.subplot(121)
        plt.imshow(x[0].cpu().permute(1,2,0))
        plt.title('Original Image')
        plt.axis('off')
        
        # Reconstructed image
        plt.subplot(122)
        plt.imshow(decoded[0].cpu().permute(1,2,0))
        plt.title('Reconstructed Image')
        plt.axis('off')
        
        plt.savefig('reconstruction_test.png')
        plt.close()

if __name__ == '__main__':
    test_model()
