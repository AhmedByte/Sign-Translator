import torch
import torch.nn as nn
from sign_language_recognition import CNN_LSTM_Model, SignLanguageDataset
from torch.utils.data import DataLoader
import os
import shutil

def verify_architecture():
    print("Verifying Model Architecture...")
    # Mock parameters
    batch_size = 2
    frames = 60
    channels = 3
    height = 224
    width = 224
    num_classes = 5
    
    model = CNN_LSTM_Model(num_classes=num_classes, cnn_arch='resnet18')
    input_tensor = torch.randn(batch_size, frames, channels, height, width)
    
    try:
        output = model(input_tensor)
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output.shape}")
        
        expected_shape = (batch_size, num_classes)
        if output.shape == expected_shape:
            print("SUCCESS: Model forward pass successful with correct output shape.")
        else:
            print(f"FAILURE: Expected shape {expected_shape}, got {output.shape}")
            
    except Exception as e:
        print(f"FAILURE: Model forward pass threw exception: {e}")

def verify_dataset_logic():
    print("\nVerifying Dataset Sampling Logic...")
    # We can mock the dataset behavior or try to instantiate it if files exist.
    # Since we have the files, let's try to instantiate it with a small subset or check one item.
    
    root_dir = '/home/abdrabo/Desktop/graduation_project/simulation/sim_cropped_data'
    csv_file = '/home/abdrabo/Desktop/graduation_project/simulation/labels_data.csv'
    
    if os.path.exists(root_dir) and os.path.exists(csv_file):
        try:
            dataset = SignLanguageDataset(root_dir=root_dir, csv_file=csv_file, frames_per_clip=60, num_buckets=6)
            print(f"Dataset length: {len(dataset)}")
            
            # Try getting one item
            if len(dataset) > 0:
                frames, label = dataset[0]
                print(f"Item 0 frames shape: {frames.shape}")
                print(f"Item 0 label: {label}")
                
                if frames.shape == (60, 3, 224, 224): # If transforms resize, otherwise might be original size
                    print("SUCCESS: Dataset __getitem__ returns correct shape (assuming transform=None or mock resize).")
                else:
                     # Note: In the script I passed transform=None to dataset constructor in verify_dataset_logic, 
                     # but wait, the real script uses transforms.Resize. 
                     # If I don't pass transforms here, it will be original image size. 
                     # I should interpret the result accordingly.
                     print(f"Dataset __getitem__ frames shape (no resize): {frames.shape}")
                     # Check at least the frames count (60)
                     if frames.shape[0] == 60:
                         print("SUCCESS: Correct frame count (60).")
                     else:
                         print("FAILURE: Incorrect frame count.")
            else:
                print("Dataset is empty.")
                
        except Exception as e:
            print(f"FAILURE: Dataset instantiation or access failed: {e}")
    else:
        print("Real data not found, skipping dataset verification.")

if __name__ == "__main__":
    verify_architecture()
    verify_dataset_logic()
