import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
import matplotlib.pyplot as plt

class VoiceStressLSTM(nn.Module):
    """LSTM model for voice stress detection"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=3):
        super(VoiceStressLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Reshape input for LSTM (batch_size, seq_len, input_size)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(attended_output))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def extract_features(self, x):
        """Extract feature embeddings (before final classification)"""
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        lstm_out, _ = self.lstm(x)
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Feature layers (before final classification)
        x = F.relu(self.fc1(attended_output))
        x = F.relu(self.fc2(x))
        
        return x  # 32-dimensional feature vector

class VoiceModelTrainer:
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
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            
            if batch_idx % 50 == 0:
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
                data, target = data.to(self.device), target.to(self.device)
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
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001):
        """Full training loop"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        best_val_acc = 0
        best_model_state = None
        patience_counter = 0
        early_stopping_patience = 20
        
        print("Starting voice model training...")
        
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
                patience_counter = 0
                print(f"New best validation accuracy: {best_val_acc:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
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
        ax1.set_title('Voice Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Voice Model Accuracy')
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
            'model_architecture': 'VoiceStressLSTM',
            'input_size': self.model.input_size,
            'hidden_size': self.model.hidden_size,
            'num_layers': self.model.num_layers,
            'num_classes': self.model.num_classes
        }, path)
        print(f"Voice model saved to {path}")

def train_voice_model(data_dir, model_save_path, epochs=100):
    """Main training function for voice model"""
    from preprocessing.ravdess_preprocessor import create_audio_data_loaders
    
    # Create data loaders
    train_loader, val_loader = create_audio_data_loaders(data_dir, batch_size=32)
    
    # Get input size from first batch
    sample_batch = next(iter(train_loader))
    input_size = sample_batch[0].shape[1]
    
    print(f"Voice feature input size: {input_size}")
    
    # Initialize model
    model = VoiceStressLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=3
    )
    
    trainer = VoiceModelTrainer(model)
    
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
    # Train voice model
    data_dir = "../../datasets/ravdess/preprocessed"
    model_save_path = "../../models/trained/voice_stress_model.pth"
    
    model, accuracy = train_voice_model(data_dir, model_save_path, epochs=80)
    print(f"Voice model training complete! Final accuracy: {accuracy:.4f}")