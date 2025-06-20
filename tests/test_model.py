import torch
import numpy as np
from histopathology.models.autoencoders.dae_kan_attention.model import DAE_KAN_Attention

def test_dae_kan_attention():
    """
    Test the DAE_KAN_Attention model implementation by verifying:
    1. Shape consistency through the network
    2. Proper output value ranges (should be [0,1] due to sigmoid)
    3. Correct bottleneck dimensionality
    4. Basic functionality of the full model
    """
    # Setup
    device = 'cpu'  # Use CPU for testing
    batch_size = 2
    
    # Create sample input
    x = torch.randn(batch_size, 3, 128, 128)
    
    # Initialize model
    model = DAE_KAN_Attention(device=device)
    model.eval()
    
    # Disable debug printing in decoder
    model.ae_decoder.debug = False
    
    # Forward pass
    with torch.no_grad():
        encoded, decoded, z = model(x)
        
        # Print shapes
        print("\nShape verification:")
        print(f"Input shape: {x.shape}")
        print(f"Encoded shape: {encoded.shape}")
        print(f"Bottleneck (z) shape: {z.shape}")
        print(f"Decoded shape: {decoded.shape}")
        
        # Verify value ranges
        print("\nValue ranges:")
        print(f"Input range: [{x.min().item():.3f}, {x.max().item():.3f}]")
        print(f"Decoded range: [{decoded.min().item():.3f}, {decoded.max().item():.3f}]")
        
        # Basic assertions
        assert decoded.shape == x.shape, f"Output shape {decoded.shape} doesn't match input shape {x.shape}"
        assert decoded.min() >= 0 and decoded.max() <= 1, f"Output values not in [0,1] range: [{decoded.min().item():.3f}, {decoded.max().item():.3f}]"
        assert z.shape == (batch_size, 16, 16, 16), f"Incorrect bottleneck shape: {z.shape}"
        assert encoded.shape == (batch_size, 384, 64, 64), f"Incorrect encoded shape: {encoded.shape}"
        
        # Check tensor dimensions flow
        # 1. Input: [B, 3, 128, 128]
        # 2. Encoder (after encoder1 + residual1): [B, 384, 64, 64] - This is what we test as "encoded"
        # 3. Encoder (after encoder2 + residual2): [B, 128, 32, 32] - Internal intermediate representation
        # 4. Bottleneck latent (z): [B, 16, 16, 16] - The most compressed representation
        # 5. Decoder (after decoder layers): [B, 384, 64, 64] - Before final upsampling 
        # 6. Decoder output: [B, 3, 128, 128] - Final reconstructed image
        
        # Test batch dimension handling by increasing batch size
        x_large_batch = torch.randn(batch_size * 2, 3, 128, 128)
        encoded_large, decoded_large, z_large = model(x_large_batch)
        
        assert decoded_large.shape == x_large_batch.shape, "Batch dimension handling failed"
        assert z_large.shape[0] == batch_size * 2, "Batch dimension not preserved in bottleneck"
        
        print("\nAll tests passed!")
        
        return True

if __name__ == "__main__":
    test_dae_kan_attention()
