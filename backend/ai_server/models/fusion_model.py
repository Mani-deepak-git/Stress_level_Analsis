import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
import matplotlib.pyplot as plt
from models.face_model import FaceStressCNN
from models.voice_model import VoiceStressLSTM

class MultimodalFusionModel(nn.Module):
    """Fusion model combining face and voice features for stress/confidence prediction"""
    
    def __init__(self, face_feature_dim=64, voice_feature_dim=32, num_classes=3):
        super(MultimodalFusionModel, self).__init__()
        
        self.face_feature_dim = face_feature_dim
        self.voice_feature_dim = voice_feature_dim
        self.fusion_dim = face_feature_dim + voice_feature_dim
        
        # Fusion layers
        self.fusion_fc1 = nn.Linear(self.fusion_dim, 128)
        self.fusion_fc2 = nn.Linear(128, 64)
        
        # Stress prediction head
        self.stress_head = nn.Linear(64, num_classes)
        
        # Confidence prediction head (regression)
        self.confidence_head = nn.Linear(64, 1)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        
    def forward(self, face_features, voice_features):
        # Concatenate features
        fused_features = torch.cat([face_features, voice_features], dim=1)
        
        # Fusion layers
        x = F.relu(self.bn1(self.fusion_fc1(fused_features)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fusion_fc2(x)))
        x = self.dropout(x)
        
        # Predictions
        stress_logits = self.stress_head(x)
        confidence_score = torch.sigmoid(self.confidence_head(x))  # 0-1 range
        
        return stress_logits, confidence_score

class StressConfidenceDataset(Dataset):
    """Dataset for multimodal stress and confidence prediction"""
    
    def __init__(self, face_features, voice_features, stress_labels, confidence_scores=None):
        self.face_features = torch.FloatTensor(face_features)
        self.voice_features = torch.FloatTensor(voice_features)
        self.stress_labels = torch.LongTensor(stress_labels)
        
        # Generate confidence scores if not provided
        if confidence_scores is None:
            # Inverse relationship: High stress = Low confidence
            confidence_scores = self._generate_confidence_scores(stress_labels)
        
        self.confidence_scores = torch.FloatTensor(confidence_scores)
        
    def _generate_confidence_scores(self, stress_labels):
        """Generate confidence scores based on stress levels"""
        confidence_mapping = {
            0: np.random.normal(0.8, 0.1),  # Low stress -> High confidence
            1: np.random.normal(0.5, 0.1),  # Medium stress -> Medium confidence
            2: np.random.normal(0.2, 0.1)   # High stress -> Low confidence
        }
        
        confidence_scores = []
        for label in stress_labels:
            score = confidence_mapping[label]
            score = np.clip(score, 0.0, 1.0)  # Ensure 0-1 range
            confidence_scores.append(score)
        
        return np.array(confidence_scores)
    
    def __len__(self):
        return len(self.face_features)
    
    def __getitem__(self, idx):
        return (
            self.face_features[idx],
            self.voice_features[idx],
            self.stress_labels[idx],
            self.confidence_scores[idx]
        )

class FusionModelTrainer:
    def __init__(self, fusion_model, face_model, voice_model, device='cpu'):
        self.fusion_model = fusion_model.to(device)
        self.face_model = face_model.to(device)
        self.voice_model = voice_model.to(device)
        self.device = device
        
        # Set feature extraction models to eval mode
        self.face_model.eval()
        self.voice_model.eval()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def extract_features(self, face_data, voice_data):
        """Extract features from both modalities"""
        with torch.no_grad():
            face_features = self.face_model.extract_features(face_data)
            voice_features = self.voice_model.extract_features(voice_data)
        return face_features, voice_features
    
    def train_epoch(self, train_loader, optimizer, stress_criterion, confidence_criterion):
        """Train for one epoch"""
        self.fusion_model.train()
        total_loss = 0
        all_stress_preds = []
        all_stress_labels = []
        
        for batch_idx, (face_data, voice_data, stress_labels, confidence_labels) in enumerate(train_loader):
            face_data = face_data.to(self.device)
            voice_data = voice_data.to(self.device)
            stress_labels = stress_labels.to(self.device)
            confidence_labels = confidence_labels.to(self.device)
            
            # Extract features
            face_features, voice_features = self.extract_features(face_data, voice_data)
            
            optimizer.zero_grad()
            
            # Forward pass
            stress_logits, confidence_pred = self.fusion_model(face_features, voice_features)
            
            # Calculate losses
            stress_loss = stress_criterion(stress_logits, stress_labels)
            confidence_loss = confidence_criterion(confidence_pred.squeeze(), confidence_labels)
            
            # Combined loss (weighted)
            total_batch_loss = stress_loss + 0.5 * confidence_loss
            
            total_batch_loss.backward()
            optimizer.step()
            
            total_loss += total_batch_loss.item()
            
            # Collect predictions for accuracy
            stress_pred = stress_logits.argmax(dim=1)
            all_stress_preds.extend(stress_pred.cpu().numpy())
            all_stress_labels.extend(stress_labels.cpu().numpy())
            
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {total_batch_loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_stress_labels, all_stress_preds)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, stress_criterion, confidence_criterion):
        """Validate the model"""
        self.fusion_model.eval()
        total_loss = 0
        all_stress_preds = []
        all_stress_labels = []
        all_confidence_preds = []
        all_confidence_labels = []
        
        with torch.no_grad():
            for face_data, voice_data, stress_labels, confidence_labels in val_loader:
                face_data = face_data.to(self.device)
                voice_data = voice_data.to(self.device)
                stress_labels = stress_labels.to(self.device)
                confidence_labels = confidence_labels.to(self.device)
                
                # Extract features
                face_features, voice_features = self.extract_features(face_data, voice_data)
                
                # Forward pass
                stress_logits, confidence_pred = self.fusion_model(face_features, voice_features)
                
                # Calculate losses
                stress_loss = stress_criterion(stress_logits, stress_labels)
                confidence_loss = confidence_criterion(confidence_pred.squeeze(), confidence_labels)
                total_batch_loss = stress_loss + 0.5 * confidence_loss
                
                total_loss += total_batch_loss.item()
                
                # Collect predictions
                stress_pred = stress_logits.argmax(dim=1)
                all_stress_preds.extend(stress_pred.cpu().numpy())
                all_stress_labels.extend(stress_labels.cpu().numpy())
                all_confidence_preds.extend(confidence_pred.squeeze().cpu().numpy())
                all_confidence_labels.extend(confidence_labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        stress_accuracy = accuracy_score(all_stress_labels, all_stress_preds)
        stress_f1 = f1_score(all_stress_labels, all_stress_preds, average='weighted')
        
        # Confidence MAE
        confidence_mae = np.mean(np.abs(np.array(all_confidence_preds) - np.array(all_confidence_labels)))
        
        return avg_loss, stress_accuracy, stress_f1, confidence_mae, all_stress_preds, all_stress_labels
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001):
        """Full training loop"""
        stress_criterion = nn.CrossEntropyLoss()
        confidence_criterion = nn.MSELoss()
        optimizer = optim.Adam(self.fusion_model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_acc = 0
        best_model_state = None
        
        print("Starting fusion model training...")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, stress_criterion, confidence_criterion)
            
            # Validate
            val_loss, val_acc, val_f1, conf_mae, val_preds, val_labels = self.validate(
                val_loader, stress_criterion, confidence_criterion
            )
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            print(f"Confidence MAE: {conf_mae:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.fusion_model.state_dict().copy()
                print(f"New best validation accuracy: {best_val_acc:.4f}")
        
        # Load best model
        if best_model_state:
            self.fusion_model.load_state_dict(best_model_state)
        
        print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
        
        # Final evaluation
        val_loss, val_acc, val_f1, conf_mae, val_preds, val_labels = self.validate(
            val_loader, stress_criterion, confidence_criterion
        )
        print("\nFinal Validation Results:")
        print(f"Stress Accuracy: {val_acc:.4f}")
        print(f"Stress F1-Score: {val_f1:.4f}")
        print(f"Confidence MAE: {conf_mae:.4f}")
        print("\nStress Classification Report:")
        
        # Check if we have all classes in validation set
        unique_labels = np.unique(val_labels)
        if len(unique_labels) == 3:
            print(classification_report(val_labels, val_preds, 
                                      target_names=['Low Stress', 'Medium Stress', 'High Stress']))
        else:
            print(f"Warning: Validation set only contains {len(unique_labels)} unique class(es): {unique_labels}")
            print(classification_report(val_labels, val_preds, 
                                      labels=unique_labels,
                                      target_names=[f'Class {i}' for i in unique_labels]))
        
        return best_val_acc
    
    def save_model(self, path):
        """Save trained fusion model"""
        torch.save({
            'model_state_dict': self.fusion_model.state_dict(),
            'model_architecture': 'MultimodalFusionModel',
            'face_feature_dim': self.fusion_model.face_feature_dim,
            'voice_feature_dim': self.fusion_model.voice_feature_dim,
            'num_classes': 3
        }, path)
        print(f"Fusion model saved to {path}")

def create_fusion_dataset(face_data_dir, voice_data_dir):
    """Create fusion dataset from preprocessed face and voice data"""
    # Load face data
    face_X_train = np.load(os.path.join(face_data_dir, 'X_train.npy'))
    face_X_val = np.load(os.path.join(face_data_dir, 'X_val.npy'))
    face_y_train = np.load(os.path.join(face_data_dir, 'y_train.npy'))
    face_y_val = np.load(os.path.join(face_data_dir, 'y_val.npy'))
    
    # Load voice data
    voice_X_train = np.load(os.path.join(voice_data_dir, 'X_train.npy'))
    voice_X_val = np.load(os.path.join(voice_data_dir, 'X_val.npy'))
    voice_y_train = np.load(os.path.join(voice_data_dir, 'y_train.npy'))
    voice_y_val = np.load(os.path.join(voice_data_dir, 'y_val.npy'))
    
    # Align datasets (take minimum size)
    min_train_size = min(len(face_X_train), len(voice_X_train))
    min_val_size = min(len(face_X_val), len(voice_X_val))
    
    # Create aligned datasets
    train_dataset = StressConfidenceDataset(
        face_X_train[:min_train_size],
        voice_X_train[:min_train_size],
        face_y_train[:min_train_size]  # Use face labels as primary
    )
    
    val_dataset = StressConfidenceDataset(
        face_X_val[:min_val_size],
        voice_X_val[:min_val_size],
        face_y_val[:min_val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader

def train_fusion_model(face_model_path, voice_model_path, face_data_dir, voice_data_dir, save_path):
    """Main function to train fusion model"""
    
    # Load pre-trained models
    face_checkpoint = torch.load(face_model_path, map_location='cpu')
    voice_checkpoint = torch.load(voice_model_path, map_location='cpu')
    
    # Initialize models
    face_model = FaceStressCNN(num_classes=3)
    face_model.load_state_dict(face_checkpoint['model_state_dict'])
    
    voice_model = VoiceStressLSTM(
        input_size=voice_checkpoint['input_size'],
        hidden_size=voice_checkpoint['hidden_size'],
        num_layers=voice_checkpoint['num_layers'],
        num_classes=3
    )
    voice_model.load_state_dict(voice_checkpoint['model_state_dict'])
    
    # Initialize fusion model
    fusion_model = MultimodalFusionModel(
        face_feature_dim=64,
        voice_feature_dim=32,
        num_classes=3
    )
    
    # Create datasets
    train_loader, val_loader = create_fusion_dataset(face_data_dir, voice_data_dir)
    
    # Train fusion model
    trainer = FusionModelTrainer(fusion_model, face_model, voice_model)
    best_acc = trainer.train(train_loader, val_loader, epochs=30)
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    trainer.save_model(save_path)
    
    return fusion_model, best_acc

if __name__ == "__main__":
    # Train fusion model
    face_model_path = "../../models/trained/face_stress_model.pth"
    voice_model_path = "../../models/trained/voice_stress_model.pth"
    face_data_dir = "../../datasets/fer2013/preprocessed"
    voice_data_dir = "../../datasets/ravdess/preprocessed"
    save_path = "../../models/trained/fusion_model.pth"
    
    model, accuracy = train_fusion_model(
        face_model_path, voice_model_path, 
        face_data_dir, voice_data_dir, save_path
    )
    print(f"Fusion model training complete! Final accuracy: {accuracy:.4f}")