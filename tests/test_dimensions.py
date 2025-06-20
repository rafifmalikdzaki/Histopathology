import torch
from histopathology.models.autoencoders.dae_kan_attention.model import DAE_KAN_Attention

def test_model_dimensions():
    """
    Test the tensor dimensions through each component of the DAE_KAN_Attention model.
    """
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DAE_KAN_Attention(device=device).to(device)
    model.eval()  # Set to eval mode
    
    # Test input (batch_size=2, channels=3, height=128, width=128)
    x = torch.randn(2, 3, 128, 128).to(device)
    
    print("Input shape:", x.shape)
    
    # Forward pass through encoder
    encoded, residual1, residual2 = model.ae_encoder(x)
    print("\nEncoder outputs:")
    print("- encoded shape:", encoded.shape)
    print("- residual1 shape:", residual1.shape)
    print("- residual2 shape:", residual2.shape)
    
    # Forward pass through bottleneck
    decoded_bottleneck, z = model.bottleneck(encoded)
    print("\nBottleneck outputs:")
    print("- decoded_bottleneck shape:", decoded_bottleneck.shape)
    print("- z shape:", z.shape)
    
    # Forward pass through decoder
    final_output = model.ae_decoder(decoded_bottleneck, residual1, residual2)
    print("\nDecoder output:")
    print("- final_output shape:", final_output.shape)
    
    # Full model forward pass
    encoded_full, decoded_full, z_full = model(x)
    print("\nFull model outputs:")
    print("- encoded_full shape:", encoded_full.shape)
    print("- decoded_full shape:", decoded_full.shape)
    print("- z_full shape:", z_full.shape)

if __name__ == '__main__':
    test_model_dimensions()
