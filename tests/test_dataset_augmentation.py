import pytest
import torch
import numpy as np
from histopathology.src.data_pipeline.histopathology_dataset import (
    ImageDataset,
    AutoencoderTransform
)

@pytest.fixture
def sample_images():
    """Create a batch of sample images for testing."""
    # Create 10 random test images (3 channels, 128x128)
    return torch.rand(10, 3, 128, 128)

@pytest.fixture
def sample_labels():
    """Create sample labels for testing."""
    return torch.randint(0, 2, (10,))  # Binary labels

class TestStandardMode:
    """Test the standard mode behavior (original functionality)."""
    
    def test_initialization(self, sample_images, sample_labels):
        """Test that the dataset can be initialized in standard mode."""
        dataset = ImageDataset(sample_images, sample_labels, mode="standard")
        assert len(dataset) == len(sample_images)
        assert dataset.mode == "standard"
    
    def test_getitem(self, sample_images, sample_labels):
        """Test that __getitem__ returns the expected output in standard mode."""
        dataset = ImageDataset(sample_images, sample_labels, mode="standard")
        item = dataset[0]
        
        # Standard mode should return just the image tensor
        assert isinstance(item, torch.Tensor)
        assert item.shape == (3, 128, 128)
        assert torch.allclose(item, sample_images[0])

class TestAutoencoderMode:
    """Test the autoencoder mode with dual augmentation."""
    
    def test_initialization(self, sample_images, sample_labels):
        """Test that the dataset can be initialized in autoencoder mode."""
        dataset = ImageDataset(
            sample_images, 
            sample_labels, 
            mode="autoencoder",
            noise_level=0.3,
            mask_ratio=0.4
        )
        assert len(dataset) == len(sample_images)
        assert dataset.mode == "autoencoder"
        assert hasattr(dataset, "transform")
        assert isinstance(dataset.transform, AutoencoderTransform)
    
    def test_getitem_format(self, sample_images, sample_labels):
        """Test that __getitem__ returns the expected dictionary format in autoencoder mode."""
        dataset = ImageDataset(
            sample_images, 
            sample_labels, 
            mode="autoencoder"
        )
        item = dataset[0]
        
        # Autoencoder mode should return a dictionary
        assert isinstance(item, dict)
        assert "input_image" in item
        assert "target_image" in item
        assert "label" in item
        
        # Check shapes
        assert item["input_image"].shape == (3, 128, 128)
        assert item["target_image"].shape == (3, 128, 128)
        assert item["label"].shape == torch.Size([])
        
        # Check values are in valid range
        assert (item["input_image"] >= 0).all() and (item["input_image"] <= 1).all()
        assert (item["target_image"] >= 0).all() and (item["target_image"] <= 1).all()
    
    def test_dual_augmentation(self, sample_images, sample_labels):
        """Test that dual augmentation produces different inputs and targets."""
        dataset = ImageDataset(
            sample_images, 
            sample_labels, 
            mode="autoencoder",
            noise_level=0.0,  # No noise to isolate augmentation effect
            mask_ratio=0.0    # No masking to isolate augmentation effect
        )
        
        # Get multiple samples
        items = [dataset[0] for _ in range(5)]
        
        # Verify inputs are different from targets
        for item in items:
            diff = (item["input_image"] - item["target_image"]).abs().mean().item()
            assert diff > 0.01, "Input and target images should be different due to dual augmentation"
        
        # Verify different calls produce different outputs
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                input_diff = (items[i]["input_image"] - items[j]["input_image"]).abs().mean().item()
                target_diff = (items[i]["target_image"] - items[j]["target_image"]).abs().mean().item()
                assert input_diff > 0.01, "Different calls should produce different augmentations"
                assert target_diff > 0.01, "Different calls should produce different augmentations"

class TestNoiseAndMasking:
    """Test noise addition and masking functionality."""
    
    def test_noise_effect(self, sample_images, sample_labels):
        """Test that noise_level parameter affects the output statistics."""
        # Create datasets with different noise levels
        dataset_no_noise = ImageDataset(
            sample_images, 
            sample_labels, 
            mode="autoencoder",
            noise_level=0.0,
            mask_ratio=0.0
        )
        dataset_low_noise = ImageDataset(
            sample_images, 
            sample_labels, 
            mode="autoencoder",
            noise_level=0.1,
            mask_ratio=0.0
        )
        dataset_high_noise = ImageDataset(
            sample_images, 
            sample_labels, 
            mode="autoencoder",
            noise_level=0.3,
            mask_ratio=0.0
        )
        
        # Get samples with same index to compare
        sample_no_noise = dataset_no_noise[0]
        sample_low_noise = dataset_low_noise[0]
        sample_high_noise = dataset_high_noise[0]
        
        # Measure standard deviation of each sample
        std_no_noise = sample_no_noise["input_image"].std().item()
        std_low_noise = sample_low_noise["input_image"].std().item()
        std_high_noise = sample_high_noise["input_image"].std().item()
        
        # Higher noise levels should result in higher standard deviation
        assert std_low_noise > std_no_noise
        assert std_high_noise > std_low_noise
    
    def test_masking_effect(self, sample_images, sample_labels):
        """Test that mask_ratio parameter affects the output statistics."""
        # Create datasets with different mask ratios
        dataset_no_mask = ImageDataset(
            sample_images, 
            sample_labels, 
            mode="autoencoder",
            noise_level=0.0,
            mask_ratio=0.0
        )
        dataset_light_mask = ImageDataset(
            sample_images, 
            sample_labels, 
            mode="autoencoder",
            noise_level=0.0,
            mask_ratio=0.2
        )
        dataset_heavy_mask = ImageDataset(
            sample_images, 
            sample_labels, 
            mode="autoencoder",
            noise_level=0.0,
            mask_ratio=0.5
        )
        
        # Get samples
        sample_no_mask = dataset_no_mask[0]
        sample_light_mask = dataset_light_mask[0]
        sample_heavy_mask = dataset_heavy_mask[0]
        
        # Count zero pixels in each sample
        zeros_no_mask = (sample_no_mask["input_image"] == 0).float().mean().item()
        zeros_light_mask = (sample_light_mask["input_image"] == 0).float().mean().item()
        zeros_heavy_mask = (sample_heavy_mask["input_image"] == 0).float().mean().item()
        
        # Higher mask ratios should result in more zero pixels
        assert zeros_light_mask > zeros_no_mask
        assert zeros_heavy_mask > zeros_light_mask
    
    def test_noise_and_mask_together(self, sample_images, sample_labels):
        """Test that noise and masking can be applied together."""
        dataset = ImageDataset(
            sample_images,
            sample_labels,
            mode="autoencoder",
            noise_level=0.3,
            mask_ratio=0.4
        )
        
        sample = dataset[0]
        
        # Original image should be different from input due to both noise and masking
        diff = (sample_images[0] - sample["input_image"]).abs().mean().item()
        assert diff > 0.1, "Combined noise and masking should significantly alter the image"
        
        # Target should not have noise or masking
        target_diff = (sample_images[0] - sample["target_image"]).abs().mean().item()
        assert target_diff < diff, "Target should be less different than input vs original"

class TestInputValidation:
    """Test input validation and error handling."""
    
    def test_invalid_images_tensor(self):
        """Test validation of images tensor."""
        # Wrong type
        with pytest.raises(TypeError):
            ImageDataset(["not", "a", "tensor"])
            
        # Wrong dimensions
        with pytest.raises(ValueError):
            ImageDataset(torch.rand(10, 128, 128))  # Missing channel dim
            
        # Wrong number of channels
        with pytest.raises(ValueError):
            ImageDataset(torch.rand(10, 1, 128, 128))  # Grayscale not allowed
    
    def test_invalid_labels_tensor(self, sample_images):
        """Test validation of labels tensor."""
        # Wrong type
        with pytest.raises(TypeError):
            ImageDataset(sample_images, labels=["not", "a", "tensor"])
            
        # Wrong dimensions
        with pytest.raises(ValueError):
            ImageDataset(sample_images, labels=torch.rand(10, 2))  # Too many dims
            
        # Mismatched length
        with pytest.raises(ValueError):
            ImageDataset(sample_images, labels=torch.zeros(5))  # Wrong length
    
    def test_invalid_mode(self, sample_images):
        """Test validation of mode parameter."""
        with pytest.raises(ValueError):
            ImageDataset(sample_images, mode="invalid_mode")
    
    def test_normalization(self):
        """Test that images are properly normalized."""
        # Create images outside [0,1] range
        images = torch.rand(10, 3, 128, 128) * 2 - 0.5  # Range [-0.5, 1.5]
        
        # Should automatically normalize
        dataset = ImageDataset(images)
        
        # Check that dataset images are within [0,1]
        assert (dataset.images >= 0).all() and (dataset.images <= 1).all()

if __name__ == "__main__":
    # Allow running tests directly
    pytest.main(["-xvs", __file__])
