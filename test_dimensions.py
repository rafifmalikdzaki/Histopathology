import torch
from histopathology.models.autoencoders.dae_kan_attention.model import DAE_KAN_Attention

def test_dimensions():
    # Create model
    model = DAE_KAN_Attention(device='cuda')
    model = model.to('cuda')
    model.eval()
    
    # Create test input
    batch_size = 2
    x = torch.randn(batch_size, 3, 128, 128).to('cuda')
    print("\nInput shape:", x.shape)
    
    # Test encoder path
    with torch.no_grad():
        # Get encoder outputs
        encoded, residual1, residual2 = model.ae_encoder(x)
        print("\nEncoder path:")
        print("Residual1 shape (from encoder1):", residual1.shape)
        print("Residual2 shape (from encoder2):", residual2.shape)
        print("Encoded shape (after encoder path):", encoded.shape)
        
        # Test bottleneck
        decoded_bottleneck, z = model.bottleneck(encoded)
        print("\nBottleneck:")
        print("Bottleneck encoding (z) shape:", z.shape)
        print("Decoded bottleneck shape:", decoded_bottleneck.shape)
        
        # Test decoder
        output = model.ae_decoder(decoded_bottleneck, residual1, residual2)
        print("\nDecoder path:")
        print("Final output shape:", output.shape)
        
        # Verify output range
        print("\nOutput statistics:")
        print("Min value:", output.min().item())
        print("Max value:", output.max().item())
        print("Mean value:", output.mean().item())
        
        # Test complete forward pass
        encoded_full, decoded_full, z_full = model(x)
        print("\nComplete forward pass:")
        print("Encoded shape:", encoded_full.shape)
        print("Decoded shape:", decoded_full.shape)
        print("Bottleneck shape:", z_full.shape)
        
        # Verify shapes match
        assert output.shape == x.shape, f"Output shape {output.shape} doesn't match input shape {x.shape}"
        assert decoded_full.shape == x.shape, f"Decoded shape {decoded_full.shape} doesn't match input shape {x.shape}"

if __name__ == '__main__':
    test_dimensions()
