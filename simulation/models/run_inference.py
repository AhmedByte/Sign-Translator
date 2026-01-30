import torch
import torchvision.transforms as transforms
from sign_language_recognition import CNN_LSTM_Model, SignLanguageDataset, predict_single_video
import os
import pandas as pd

def load_and_predict():
    # Paths
    MODEL_PATH = 'sign_language_model.pth'
    VIDEO_PATH = '/home/abdrabo/Desktop/graduation_project/simulation/sim_cropped_data/0005/01'
    CSV_FILE = '/home/abdrabo/Desktop/graduation_project/simulation/labels_data.csv'
    
    # 1. Recreate Class Mapping (needed to map index back to text)
    # We use the dataset class to generate the mapping consistently
    if not os.path.exists(CSV_FILE):
        print("CSV file not found.")
        return
        
    # Dummy dataset just to get the label map
    dataset = SignLanguageDataset(root_dir='.', csv_file=CSV_FILE)
    idx_to_label = dataset.idx_to_label
    num_classes = dataset.num_classes
    
    # 2. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CNN_LSTM_Model(num_classes=num_classes, cnn_arch='resnet18')
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
    else:
        print("Model file not found. Please train the model first.")
        return

    # 3. Run Prediction
    predict_single_video(model, idx_to_label, VIDEO_PATH)

if __name__ == "__main__":
    load_and_predict()
