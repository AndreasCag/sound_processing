import torch
import torch.nn as nn

class RawAudioCNN(nn.Module):
    def __init__(self):
        super(RawAudioCNN, self).__init__()
        
        # Using fixed pooling sizes instead of adaptive pooling
        self.features = nn.Sequential(
            # First block
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=160, stride=32, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),  # Increased pool size to reduce dimensions faster

            # Second block
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=9, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),  # Increased pool size

            # Third block
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),  # Increased pool size
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(960, 256),  # Make sure this matches your input size
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, 1, max_length]
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x.squeeze(-1)

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
