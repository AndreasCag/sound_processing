import torch
import torch.nn as nn

class RawAudioCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(RawAudioCNN, self).__init__()
        
        # Larger network with more layers and filters
        # Conv1D expects shape: [batch, channels, time]
        self.features = nn.Sequential(
            # First block
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Second block
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Third block
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Fourth block
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Fifth block
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Global pooling
            nn.AdaptiveAvgPool1d(16)  # reduce to fixed size
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 16, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x shape: [batch_size, 1, max_length]
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device='cpu'):
        """Load model and training state from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create new model instance
        model = cls()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return {
            'model': model,
            'optimizer_state_dict': checkpoint['optimizer_state_dict'],
            'scheduler_state_dict': checkpoint['scheduler_state_dict'],
            'epoch': checkpoint['epoch'],
            'val_acc': checkpoint['val_acc'],
            'val_loss': checkpoint['val_loss']
        }

    def load_weights(self, checkpoint_path, device='cpu'):
        """Load just the model weights."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.to(device)
        return checkpoint['val_acc']
