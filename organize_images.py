#!/usr/bin/env python3
"""
Script to organize histopathology images into train and test directories
based on CSV files containing the training/testing split.
"""

import os
import shutil
import pandas as pd
from tqdm import tqdm
import sys

# Define paths
BASE_DIR = "/home/dzakirm/Research/Histopathology/histopathology/data/processed"
SOURCE_DIR = os.path.join(BASE_DIR, "HeparUnifiedPNG")
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
TRAIN_DIR = os.path.join(SOURCE_DIR, "train")
TEST_DIR = os.path.join(SOURCE_DIR, "test")

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    return directory

def process_images(csv_path, target_dir, description):
    """Process images from CSV and move them to target directory."""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Ensure the 'Image' column exists
        if 'Image' not in df.columns:
            print(f"Error: 'Image' column not found in {csv_path}")
            # Try to find a column that might contain image filenames
            possible_image_cols = [col for col in df.columns if 'image' in col.lower() 
                                  or 'file' in col.lower() or '.png' in str(df[col].iloc[0])]
            if possible_image_cols:
                image_col = possible_image_cols[0]
                print(f"Using '{image_col}' as the image column instead")
            else:
                # Print the first few rows and columns to help diagnose the issue
                print("CSV columns:", df.columns.tolist())
                print("First few rows:")
                print(df.head())
                sys.exit(1)
        else:
            image_col = 'Image'
        
        # Process each image
        not_found_images = []
        moved_count = 0
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=description):
            image_name = row[image_col]
            
            # Add .png extension if not present
            if not image_name.endswith('.png'):
                image_name = f"{image_name}.png"
            
            source_path = os.path.join(SOURCE_DIR, image_name)
            target_path = os.path.join(target_dir, image_name)
            
            # Check if source file exists
            if os.path.exists(source_path):
                # Copy file to target directory (use shutil.move if you want to move instead of copy)
                shutil.copy2(source_path, target_path)
                moved_count += 1
            else:
                not_found_images.append(image_name)
        
        # Report results
        print(f"Successfully processed {moved_count} images to {target_dir}")
        
        if not_found_images:
            print(f"Warning: {len(not_found_images)} images were not found in the source directory")
            if len(not_found_images) < 10:
                print("Missing images:", not_found_images)
            else:
                print("First 10 missing images:", not_found_images[:10], "...")
        
        return moved_count
    
    except Exception as e:
        print(f"Error processing {csv_path}: {str(e)}")
        return 0

def main():
    """Main function to organize images."""
    print(f"Starting organization of images from {SOURCE_DIR}")
    
    # Ensure directories exist
    ensure_dir(SOURCE_DIR)
    ensure_dir(TRAIN_DIR)
    ensure_dir(TEST_DIR)
    
    # Process train images
    if os.path.exists(TRAIN_CSV):
        train_count = process_images(TRAIN_CSV, TRAIN_DIR, "Processing training images")
    else:
        print(f"Error: Training CSV file not found at {TRAIN_CSV}")
        train_count = 0
    
    # Process test images
    if os.path.exists(TEST_CSV):
        test_count = process_images(TEST_CSV, TEST_DIR, "Processing testing images")
    else:
        print(f"Error: Testing CSV file not found at {TEST_CSV}")
        test_count = 0
    
    # Final report
    print("\nSummary:")
    print(f"- Images moved to training directory: {train_count}")
    print(f"- Images moved to testing directory: {test_count}")
    print(f"- Total images processed: {train_count + test_count}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
