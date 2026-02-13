import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from pathlib import Path

class FER2013FolderPreprocessor:
    def __init__(self, dataset_dir, output_dir):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        
        # Emotion to stress mapping
        self.emotion_to_stress = {
            'angry': 2,     # High stress
            'disgust': 1,   # Medium stress  
            'fear': 2,      # High stress
            'happy': 0,     # Low stress
            'sad': 2,       # High stress
            'surprise': 1,  # Medium stress
            'neutral': 0    # Low stress
        }
        
    def load_images_from_folder(self, folder_path):
        """Load all images from a folder"""
        images = []
        labels = []
        
        for emotion_folder in folder_path.iterdir():
            if emotion_folder.is_dir():
                emotion_name = emotion_folder.name.lower()
                if emotion_name in self.emotion_to_stress:
                    stress_level = self.emotion_to_stress[emotion_name]
                    
                    # Load all images in this emotion folder
                    image_files = list(emotion_folder.glob('*.jpg')) + list(emotion_folder.glob('*.png'))
                    
                    for img_path in image_files:
                        try:
                            # Load image in grayscale
                            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                # Resize to 48x48 if needed
                                if img.shape != (48, 48):
                                    img = cv2.resize(img, (48, 48))
                                images.append(img)
                                labels.append(stress_level)
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")
                            continue
                    
                    print(f"Loaded {len(image_files)} images from {emotion_name} -> stress level {stress_level}")
        
        return np.array(images), np.array(labels)
    
    def preprocess_images(self, images):
        """Normalize and preprocess images"""
        # Normalize to 0-1 range
        images = images.astype(np.float32) / 255.0
        
        # Add channel dimension for CNN
        images = np.expand_dims(images, axis=1)  # Shape: (N, 1, 48, 48)
        
        return images
    
    def create_datasets(self):
        """Create train/validation datasets from folder structure"""
        print("Loading FER-2013 dataset from folders...")
        
        # Check if we have train/test folders or just emotion folders
        train_path = self.dataset_dir / 'train'
        test_path = self.dataset_dir / 'test'
        
        if train_path.exists() and test_path.exists():
            print("Found train/test split folders")
            # Load training data
            X_train, y_train = self.load_images_from_folder(train_path)
            # Load test data as validation
            X_val, y_val = self.load_images_from_folder(test_path)
        else:
            print("No train/test split found, creating split from all images")
            # Load all images and create our own split
            X_all, y_all = self.load_images_from_folder(self.dataset_dir)
            X_train, X_val, y_train, y_val = train_test_split(
                X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
            )
        
        # Preprocess images
        X_train = self.preprocess_images(X_train)
        X_val = self.preprocess_images(X_val)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Image shape: {X_train[0].shape}")
        print(f"Stress distribution - Train: {np.bincount(y_train)}")
        print(f"Stress distribution - Val: {np.bincount(y_val)}")
        
        # Save preprocessed data
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(self.output_dir / 'X_train.npy', X_train)
        np.save(self.output_dir / 'X_val.npy', X_val)
        np.save(self.output_dir / 'y_train.npy', y_train)
        np.save(self.output_dir / 'y_val.npy', y_val)
        
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
    dataset_dir = "../../datasets/fer2013"
    output_dir = "../../datasets/fer2013/preprocessed"
    
    preprocessor = FER2013FolderPreprocessor(dataset_dir, output_dir)
    X_train, X_val, y_train, y_val = preprocessor.create_datasets()
    
    print("FER-2013 folder preprocessing complete!")