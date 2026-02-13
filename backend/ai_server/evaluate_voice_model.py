"""
Voice Stress Model Evaluation Script
Evaluates the trained voice_stress_model.pth on test data
"""

import torch
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.voice_model import VoiceStressLSTM
from preprocessing.ravdess_preprocessor import create_audio_data_loaders

class VoiceModelEvaluator:
    def __init__(self, model_path, data_dir, device='cpu'):
        self.device = device
        self.data_dir = data_dir
        
        # Load model
        print("Loading voice stress model...")
        checkpoint = torch.load(model_path, map_location=device)
        
        self.model = VoiceStressLSTM(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            num_classes=checkpoint['num_classes']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        print("Model loaded successfully!")
        
    def evaluate(self, data_loader):
        """Evaluate model on dataset"""
        all_preds = []
        all_labels = []
        all_probs = []
        
        print("\nEvaluating model...")
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                
                output = self.model(data)
                probs = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                if batch_idx % 20 == 0:
                    print(f"Processed {batch_idx}/{len(data_loader)} batches")
        
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        metrics['per_class'] = {
            'precision': precision_per_class,
            'recall': recall_per_class,
            'f1': f1_per_class
        }
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                   xticklabels=['Low Stress', 'Medium Stress', 'High Stress'],
                   yticklabels=['Low Stress', 'Medium Stress', 'High Stress'])
        plt.title('Voice Stress Model - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")
        plt.show()
    
    def plot_class_distribution(self, y_true, y_pred, save_path=None):
        """Plot class distribution comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        classes = ['Low Stress', 'Medium Stress', 'High Stress']
        
        # True distribution
        true_counts = np.bincount(y_true, minlength=3)
        ax1.bar(classes, true_counts, color='lightgreen')
        ax1.set_title('True Label Distribution')
        ax1.set_ylabel('Count')
        ax1.set_xlabel('Stress Level')
        
        # Predicted distribution
        pred_counts = np.bincount(y_pred, minlength=3)
        ax2.bar(classes, pred_counts, color='lightcoral')
        ax2.set_title('Predicted Label Distribution')
        ax2.set_ylabel('Count')
        ax2.set_xlabel('Stress Level')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Class distribution plot saved to {save_path}")
        plt.show()
    
    def plot_per_class_metrics(self, metrics, save_path=None):
        """Plot per-class metrics comparison"""
        classes = ['Low Stress', 'Medium Stress', 'High Stress']
        precision = metrics['per_class']['precision']
        recall = metrics['per_class']['recall']
        f1 = metrics['per_class']['f1']
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width, precision, width, label='Precision', color='skyblue')
        ax.bar(x, recall, width, label='Recall', color='lightgreen')
        ax.bar(x + width, f1, width, label='F1-Score', color='salmon')
        
        ax.set_xlabel('Stress Level')
        ax.set_ylabel('Score')
        ax.set_title('Voice Model - Per-Class Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Per-class metrics plot saved to {save_path}")
        plt.show()
    
    def print_metrics(self, metrics):
        """Print all metrics in formatted way"""
        print("\n" + "="*60)
        print("VOICE STRESS MODEL EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision (Macro):  {metrics['precision_macro']:.4f}")
        print(f"  Precision (Weighted): {metrics['precision_weighted']:.4f}")
        print(f"  Recall (Macro):     {metrics['recall_macro']:.4f}")
        print(f"  Recall (Weighted):  {metrics['recall_weighted']:.4f}")
        print(f"  F1-Score (Macro):   {metrics['f1_macro']:.4f}")
        print(f"  F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        classes = ['Low Stress', 'Medium Stress', 'High Stress']
        for i, class_name in enumerate(classes):
            print(f"\n  {class_name}:")
            print(f"    Precision: {metrics['per_class']['precision'][i]:.4f}")
            print(f"    Recall:    {metrics['per_class']['recall'][i]:.4f}")
            print(f"    F1-Score:  {metrics['per_class']['f1'][i]:.4f}")
        
        print("\n" + "="*60)
    
    def save_results(self, metrics, y_true, y_pred, save_path):
        """Save evaluation results to file"""
        with open(save_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("VOICE STRESS MODEL EVALUATION RESULTS\n")
            f.write("="*60 + "\n\n")
            
            f.write("Overall Metrics:\n")
            f.write(f"  Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
            f.write(f"  Precision (Macro):  {metrics['precision_macro']:.4f}\n")
            f.write(f"  Precision (Weighted): {metrics['precision_weighted']:.4f}\n")
            f.write(f"  Recall (Macro):     {metrics['recall_macro']:.4f}\n")
            f.write(f"  Recall (Weighted):  {metrics['recall_weighted']:.4f}\n")
            f.write(f"  F1-Score (Macro):   {metrics['f1_macro']:.4f}\n")
            f.write(f"  F1-Score (Weighted): {metrics['f1_weighted']:.4f}\n\n")
            
            f.write("Per-Class Metrics:\n")
            classes = ['Low Stress', 'Medium Stress', 'High Stress']
            for i, class_name in enumerate(classes):
                f.write(f"\n  {class_name}:\n")
                f.write(f"    Precision: {metrics['per_class']['precision'][i]:.4f}\n")
                f.write(f"    Recall:    {metrics['per_class']['recall'][i]:.4f}\n")
                f.write(f"    F1-Score:  {metrics['per_class']['f1'][i]:.4f}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("\nDetailed Classification Report:\n")
            f.write(classification_report(y_true, y_pred, 
                                        target_names=['Low Stress', 'Medium Stress', 'High Stress']))
        
        print(f"\nResults saved to {save_path}")

def main():
    # Paths
    model_path = "../../models/trained/voice_stress_model.pth"
    data_dir = "../../datasets/ravdess/preprocessed"
    output_dir = "../../models/trained/evaluation_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the model first using train_models.py")
        return
    
    # Check if data exists
    if not os.path.exists(data_dir):
        print(f"ERROR: Preprocessed data not found at {data_dir}")
        print("Please preprocess the dataset first")
        return
    
    # Load data
    print("Loading validation data...")
    _, val_loader = create_audio_data_loaders(data_dir, batch_size=32)
    
    # Initialize evaluator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    evaluator = VoiceModelEvaluator(model_path, data_dir, device)
    
    # Evaluate
    y_pred, y_true, y_probs = evaluator.evaluate(val_loader)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    
    # Print metrics
    evaluator.print_metrics(metrics)
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, 
                              target_names=['Low Stress', 'Medium Stress', 'High Stress']))
    
    # Plot confusion matrix
    cm_path = os.path.join(output_dir, 'voice_model_confusion_matrix.png')
    evaluator.plot_confusion_matrix(y_true, y_pred, cm_path)
    
    # Plot class distribution
    dist_path = os.path.join(output_dir, 'voice_model_class_distribution.png')
    evaluator.plot_class_distribution(y_true, y_pred, dist_path)
    
    # Plot per-class metrics
    metrics_path = os.path.join(output_dir, 'voice_model_per_class_metrics.png')
    evaluator.plot_per_class_metrics(metrics, metrics_path)
    
    # Save results
    results_path = os.path.join(output_dir, 'voice_model_evaluation_results.txt')
    evaluator.save_results(metrics, y_true, y_pred, results_path)
    
    print("\n✓ Evaluation complete!")
    print(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    main()
