import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os

class FER2013Preprocessor:
    def __init__(self, csv_path, output_dir):
        self.csv_path = csv_path
        self.output_dir = output_dir
        
        # Stress level mapping
        self.emotion_to_stress = {
            0: 2,  # Angry -> High stress
            1: 1,  # Disgust -> Medium stress  
            2: 2,  # Fear -> High stress
            3: 0,  # Happy -> Low stress
            4: 2,  # Sad -> High stress
            5: 1,  # Surprise -> Medium stress
            6: 0   # Neutral -> Low stress
        }
        
    def load_data(self):
        """Load FER-2013 CSV data"""
        print("Loading FER-2013 dataset...")
        df = pd.read_csv(self.csv_path)
        
        # Convert pixel strings to numpy arrays
        pixels = []
        emotions = []
        
        for idx, row in df.iterrows():
            pixel_string = row['pixels']
            emotion = row['emotion']
            
            # Convert pixel string to array
            pixel_array = np.array([int(pixel) for pixel in pixel_string.split()])
            pixel_array = pixel_array.reshape(48, 48)
            
            pixels.append(pixel_array)
            emotions.append(emotion)
            
            if idx % 5000 == 0:
                print(f"Processed {idx} images...")
        
        return np.array(pixels), np.array(emotions)
    
    def preprocess_images(self, pixels):
        """Normalize and preprocess images"""
        # Normalize to 0-1 range
        pixels = pixels.astype(np.float32) / 255.0
        
        # Add channel dimension for CNN
        pixels = np.expand_dims(pixels, axis=1)  # Shape: (N, 1, 48, 48)
        
        return pixels
    
    def map_emotions_to_stress(self, emotions):
        """Map emotion labels to stress levels"""
        stress_levels = np.array([self.emotion_to_stress[emotion] for emotion in emotions])
        return stress_levels
    
    def create_datasets(self):
        """Create train/validation datasets"""
        # Load raw data
        pixels, emotions = self.load_data()
        
        # Preprocess
        pixels = self.preprocess_images(pixels)
        stress_levels = self.map_emotions_to_stress(emotions)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            pixels, stress_levels, test_size=0.2, random_state=42, stratify=stress_levels
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Stress distribution - Train: {np.bincount(y_train)}")
        print(f"Stress distribution - Val: {np.bincount(y_val)}")
        
        # Save preprocessed data
        os.makedirs(self.output_dir, exist_ok=True)
        
        np.save(os.path.join(self.output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(self.output_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(self.output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(self.output_dir, 'y_val.npy'), y_val)
        
        print(f"Preprocessed data saved to {self.output_dir}")
        
        return X_train, X_val, y_train, y_val

class FER2013Dataset(Dataset):
    """PyTorch Dataset for FER-2013"""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            # Ensure image is in correct format for ToPILImage
            # Remove channel dimension if present, ToPILImage expects (H, W) for grayscale
            if len(image.shape) == 3 and image.shape[0] == 1:
                image = image.squeeze(0)  # Remove channel dimension: (1, 48, 48) -> (48, 48)
            
            # Convert float32 to uint8 for PIL Image (0-1 range to 0-255)
            if image.dtype == np.float32:
                image = (image * 255).astype(np.uint8)
            
            # Convert to PIL Image for transforms - specify mode='L' for grayscale
            image = transforms.ToPILImage(mode='L')(image)
            image = self.transform(image)
        else:
            image = torch.FloatTensor(image)
            
        return image, torch.LongTensor([label])

def create_data_loaders(data_dir, batch_size=32):
    """Create PyTorch data loaders with augmentation"""
    
    # Load preprocessed data
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # No augmentation for validation
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create datasets
    train_dataset = FER2013Dataset(X_train, y_train, transform=train_transform)
    val_dataset = FER2013Dataset(X_val, y_val, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Example usage
    csv_path = "../../datasets/fer2013/fer2013.csv"
    output_dir = "../../datasets/fer2013/preprocessed"
    
    preprocessor = FER2013Preprocessor(csv_path, output_dir)
    X_train, X_val, y_train, y_val = preprocessor.create_datasets()
    
    print("FER-2013 preprocessing complete!")