import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
import matplotlib.pyplot as plt

class FaceStressCNN(nn.Module):
    """Lightweight CNN for facial stress detection"""
    
    def __init__(self, num_classes=3):  # 3 stress levels: Low, Medium, High
        super(FaceStressCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 48x48 -> 24x24
        
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 24x24 -> 12x12
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 12x12 -> 6x6
        
        # Conv block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 6x6 -> 3x3
        
        # Global average pooling
        x = self.adaptive_pool(x)  # 3x3 -> 1x1
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def extract_features(self, x):
        """Extract feature embeddings (before final classification)"""
        # Conv layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Global average pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Feature layers (before final classification)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return x  # 64-dimensional feature vector

class FaceModelTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.squeeze().to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.squeeze().to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return avg_loss, accuracy, f1, all_preds, all_labels
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001):
        """Full training loop"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_acc = 0
        best_model_state = None
        
        print("Starting training...")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc, val_f1, val_preds, val_labels = self.validate(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                print(f"New best validation accuracy: {best_val_acc:.4f}")
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
        
        # Final evaluation
        val_loss, val_acc, val_f1, val_preds, val_labels = self.validate(val_loader, criterion)
        print("\nFinal Validation Results:")
        print(f"Accuracy: {val_acc:.4f}")
        print(f"F1-Score: {val_f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(val_labels, val_preds, 
                                  target_names=['Low Stress', 'Medium Stress', 'High Stress']))
        
        return best_val_acc
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def save_model(self, path):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_architecture': 'FaceStressCNN',
            'num_classes': 3,
            'input_size': (1, 48, 48)
        }, path)
        print(f"Model saved to {path}")

def train_face_model(data_dir, model_save_path, epochs=50):
    """Main training function"""
    from preprocessing.fer2013_preprocessor import create_data_loaders
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(data_dir, batch_size=32)
    
    # Initialize model
    model = FaceStressCNN(num_classes=3)
    trainer = FaceModelTrainer(model)
    
    # Train model
    best_acc = trainer.train(train_loader, val_loader, epochs=epochs)
    
    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    trainer.save_model(model_save_path)
    
    # Plot training history
    plot_path = model_save_path.replace('.pth', '_training_history.png')
    trainer.plot_training_history(plot_path)
    
    return model, best_acc

if __name__ == "__main__":
    # Train face model
    data_dir = "../../datasets/fer2013/preprocessed"
    model_save_path = "../../models/trained/face_stress_model.pth"
    
    model, accuracy = train_face_model(data_dir, model_save_path, epochs=30)
    print(f"Face model training complete! Final accuracy: {accuracy:.4f}")