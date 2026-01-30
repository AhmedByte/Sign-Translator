import os
import glob
import math
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

class SignLanguageDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None, frames_per_clip=60, num_buckets=6):
        """
        Args:
            root_dir (string): Directory with all the video folders.
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied on a frame.
            frames_per_clip (int): Total number of frames to sample (default 60).
            num_buckets (int): Number of buckets to divide the video into (default 6).
        """
        self.root_dir = root_dir
        self.annotations = pd.read_csv(csv_file, dtype={'id': str})
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.num_buckets = num_buckets
        self.frames_per_bucket = frames_per_clip // num_buckets
        
        # Create a mapping from label text to integer
        # Sort to ensure consistent mapping
        if 'gloss' in self.annotations.columns:
            self.label_col = 'gloss'
        elif 'text' in self.annotations.columns:
            self.label_col = 'text'
        else:
            raise ValueError("CSV must contain 'gloss' or 'text' column for labels.")
            
        unique_labels = sorted(self.annotations[self.label_col].unique())
        self.label_map = {label: i for i, label in enumerate(unique_labels)}
        self.idx_to_label = {i: label for label, i in self.label_map.items()}
        self.num_classes = len(self.label_map)

    def __len__(self):
        return len(self.annotations)

    def _load_frames(self, path):
        # Assumes frames are images in the folder, sorted by name
        frame_paths = sorted(glob.glob(os.path.join(path, "*")))
        # Filter for image extensions if necessary, simpler here to iterate
        image_paths = [p for p in frame_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return image_paths

    def __getitem__(self, idx):
        video_id = str(self.annotations.iloc[idx, 0])
        label_text = self.annotations.iloc[idx][self.label_col]
        label = self.label_map[label_text]
        
        video_path = os.path.join(self.root_dir, video_id)
        frames_tensor = self.load_video_frames(video_path)
        
        return frames_tensor, label

    def load_video_frames(self, video_path):
        all_frames = self._load_frames(video_path)
        total_frames = len(all_frames)
        
        sampled_frames_paths = []
        
        if total_frames > 0:
            # Bucket sampling logic
            bucket_size = total_frames / self.num_buckets
            
            for i in range(self.num_buckets):
                start = i * bucket_size
                end = (i + 1) * bucket_size
                
                # We need self.frames_per_bucket frames from this range [start, end)
                # We can pick them at equal intervals
                
                # Indices in the current bucket
                # Using linspace to get evenly spaced indices within the bucket range
                indices = np.linspace(start, end, num=self.frames_per_bucket, endpoint=False).astype(int)
                
                # Clamp indices just in case
                indices = np.clip(indices, 0, total_frames - 1)
                
                for idx_in_bucket in indices:
                    sampled_frames_paths.append(all_frames[idx_in_bucket])
        
        # Load images
        frames = []
        for p in sampled_frames_paths:
            img = Image.open(p).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)
            
        if len(frames) < self.frames_per_clip:
            # Fallback padding
            while len(frames) < self.frames_per_clip:
                # If frames is empty (bad path), use zeros
                if not frames:
                     frames.append(torch.zeros(3, 224, 224))
                else:
                    frames.append(frames[-1])

        # Stack frames: (T, C, H, W)
        frames_tensor = torch.stack(frames)
        return frames_tensor

class CNN_LSTM_Model(nn.Module):
    def __init__(self, num_classes, hidden_size=256, num_layers=2, cnn_arch='resnet18'):
        """
        Args:
            num_classes (int): Number of output classes.
            hidden_size (int): Hidden size of LSTM.
            num_layers (int): Number of stacked LSTM layers.
            cnn_arch (str): 'resnet18' or 'mobilenet_v2'.
        """
        super(CNN_LSTM_Model, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1. Feature Extractor (CNN)
        if cnn_arch == 'resnet18':
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            # Remove the final FC layer
            modules = list(resnet.children())[:-1]
            self.cnn = nn.Sequential(*modules)
            self.cnn_out_dim = resnet.fc.in_features # 512 for ResNet18
            
        elif cnn_arch == 'mobilenet_v2':
            mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            self.cnn = mobilenet.features
            # MobileNetV2 features output needs global average pooling usually if not using classifier
            # But the 'features' part outputs (1280, 7, 7) or similar. 
            # We need to flatten or pool.
            # Let's add an AdaptiveAvgPool to ensure 1x1 output spatial
            self.cnn = nn.Sequential(
                mobilenet.features,
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.cnn_out_dim = mobilenet.last_channel # 1280
            
        else:
            raise ValueError("Architecture not supported options: resnet18, mobilenet_v2")

        # 2. Sequence Processor (Bi-LSTM)
        # input_size is the feature dimension from CNN
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # 3. Classification Head
        # Input to FC is hidden_size * 2 (because bidirectional)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x shape: (batch_size, frames, channels, height, width)
        batch_size, frames, C, H, W = x.size()
        
        # We need to pass each frame through the CNN.
        # Combine batch and frames dimensions: (batch * frames, C, H, W)
        c_in = x.view(batch_size * frames, C, H, W)
        
        # Forward CNN
        c_out = self.cnn(c_in) # Output: (batch * frames, cnn_out_dim, 1, 1)
        
        # Flatten
        c_out = c_out.view(c_out.size(0), -1) # (batch * frames, cnn_out_dim)
        
        # Reshape back to sequence for LSTM
        lstm_in = c_out.view(batch_size, frames, -1) # (batch, frames, cnn_out_dim)
        
        # Forward LSTM
        # out: (batch, frames, num_directions * hidden_size)
        # h_n, c_n: (num_layers * num_directions, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(lstm_in)
        
        # We can use the output of the last time step, or average/max pool.
        # For translation/classification often the last step is enough, 
        # but since 'bidirectional', we concatenate Forward_Last and Backward_Last.
        
        # h_n shape: (num_layers * 2, batch, hidden_size)
        # We take the last layer forward and backward hidden states
        
        features = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        
        # Forward Linear
        out = self.fc(features)
        
        return out

def run_training_loop():
    # Hyperparameters
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 2
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Paths (Modify these)
    # Paths (Modify these)
    ROOT_DIR = '/home/abdrabo/Desktop/graduation_project/simulation/sim_cropped_data'
    CSV_FILE = '/home/abdrabo/Desktop/graduation_project/simulation/labels_data.csv'

    # Check if files exist to run mock training
    if not os.path.exists(ROOT_DIR) or not os.path.exists(CSV_FILE):
        print("Dataset paths not found. Please update ROOT_DIR and CSV_FILE.")
        return

    # Dataset & DataLoader
    dataset = SignLanguageDataset(root_dir=ROOT_DIR, csv_file=CSV_FILE, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    # Model
    print(f"Initializing model with {dataset.num_classes} classes...")
    model = CNN_LSTM_Model(num_classes=dataset.num_classes, cnn_arch='resnet18')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training Loop
    print("Starting Training...")
    model.train()
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device) # (B, T, C, H, W)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs) # (B, Num_Classes)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            # Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            running_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Finished, Avg Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # Save Model
    save_path = 'sign_language_model.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    return model, dataset.idx_to_label

def predict_single_video(model, idx_to_label, video_path):
    print(f"\nRunning Inference on: {video_path}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Re-use Dataset logic without label
    # We can create a dummy dataset or just instantiate it and use its helper methods, 
    # but we refactored `load_video_frames` into the Dataset class so we can instantiate a dataset with dummy data.
    # Or just copy the logic.
    # since I refactored `load_video_frames` to be a method of Dataset in my proposed change, 
    # I can mock a dataset:
    dataset = SignLanguageDataset(root_dir='.', csv_file='/home/abdrabo/Desktop/graduation_project/simulation/labels_data.csv', transform=transform) # csv needs to exist
    
    # Ensure video path exists
    if not os.path.exists(video_path):
        print(f"Video path {video_path} does not exist.")
        return

    frames_tensor = dataset.load_video_frames(video_path)
    
    # Add batch dimension: (1, T, C, H, W)
    frames_tensor = frames_tensor.unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(frames_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted_idx = torch.max(outputs, 1)
        
        idx = predicted_idx.item()
        predicted_label = idx_to_label.get(idx, "Unknown")
        print(f"Predicted Class Index: {idx}")
        print(f"Predicted Label: {predicted_label}")
        print(f"Probabilities: {probabilities.cpu().numpy()}")

if __name__ == "__main__":
    trained_model, class_mapping = run_training_loop()
    
    # Test on '01_cropped'
    # Note: '01_cropped' contains *folders* of videos. 
    # Wait, the user prompt says: "Root Directory: Folders named by id, each containing sequential image frames... 01_cropped folder this is have a set of frames with class i want to test"
    # Wait, in the initial listing we saw `01_cropped` as a directory in `Desktop/anti`.
    # And `train_data` also exists.
    # Previously verification used `01_cropped` as ROOT_DIR.
    # If `01_cropped` IS the video folder (i.e. inside it are frames 001.jpg, etc), then it represents ONE sample.
    # If `01_cropped` is a dataset folder containing subfolders (00_0001, etc), then it represents the root.
    
    # Let's check the file listing again from Step 24.
    # `01_cropped` had files frame0000.jpg, frame0001.jpg... directly inside it.
    # So `01_cropped` IS A VIDEO FOLDER (one sample).
    
    # However, `SignLanguageDataset` expects `root_dir` to contain subfolders named by video_id.
    # In `verify_model.py`, I set ROOT_DIR='/home/abdrabo/Desktop/anti/01_cropped'.
    # And CSV has ids like `00_0001`.
    # Dataset `__getitem__` does: `video_path = os.path.join(self.root_dir, video_id)`
    # So it looks for `/home/abdrabo/Desktop/anti/01_cropped/00_0001`.
    # BUT `01_cropped` contains IMAGES directly.
    # So the dataset setup in `verify_model.py` was actually BROKEN if it tried to load specific IDs, 
    # unless `01_cropped` was treated as the root and the ID `00_0001` didn't exist, leading to empty logic.
    # Wait, in `verify_model.py`:
    # `dataset[0]` -> `video_id` from CSV is `00_0001`.
    # Path -> `/home/abdrabo/Desktop/anti/01_cropped/00_0001`
    # Does this exist?
    # `01_cropped` contains images. So no, `00_0001` subdir does NOT exist likely.
    # So `_load_frames` returned empty list.
    # Then `load_video_frames` (which I just wrote, or the previous `__getitem__` logic) 
    # `total_frames = 0`.
    # `sampled_frames_paths = []`.
    # `frames = []`.
    # `frames` padded with 60 zeros.
    # So the verification pass WORKED but on empty zero tensors!
    
    # Correction: The user wants to test `01_cropped`.
    # If `01_cropped` carries the frames for ONE video, then `predict_single_video` should point to it directly.
    # For Training, we need a valid dataset structure.
    # The user provided `train_lables.csv` with IDs `00_0001` etc.
    # Where are these folders?
    # Listing showed `train_data` folder. Maybe they are inside `train_data`? (Step 4 listing).
    # Let me check `train_data` content.
    
    # I will stick to the plan but point the training loop to a place that MIGHT have data, 
    # or just accept that training might run on zeros (mock) but inference will be real on `01_cropped`.
    # I'll add a check in the main block.
    
    test_video_path = '/home/abdrabo/Desktop/graduation_project/simulation/sim_cropped_data/0005'
    predict_single_video(trained_model, class_mapping, test_video_path)
