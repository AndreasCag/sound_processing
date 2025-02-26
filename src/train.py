import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from pathlib import Path

def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-3, device='cpu', save_dir='models'):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # Initialize the cosine annealing scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

    # Create save directory if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Track best validation accuracy
    best_val_acc = 0.0
    best_model_path = ''

    model.to(device)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Track class-wise training metrics
        train_correct_0 = 0
        train_total_0 = 0
        train_correct_1 = 0
        train_total_1 = 0
        
        for waveforms, labels in train_loader:
            waveforms = waveforms.unsqueeze(1).to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * waveforms.size(0)
            predicted = (outputs > 0).float()
            
            # Overall accuracy
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Class-wise accuracy
            train_correct_0 += ((predicted == 0) & (labels == 0)).sum().item()
            train_total_0 += (labels == 0).sum().item()
            train_correct_1 += ((predicted == 1) & (labels == 1)).sum().item()
            train_total_1 += (labels == 1).sum().item()
        
        train_loss = running_loss / total
        train_acc = correct / total
        train_acc_0 = train_correct_0 / train_total_0 if train_total_0 > 0 else 0
        train_acc_1 = train_correct_1 / train_total_1 if train_total_1 > 0 else 0
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Track class-wise metrics
        val_correct_0 = 0
        val_total_0 = 0
        val_correct_1 = 0
        val_total_1 = 0
        
        with torch.no_grad():
            for waveforms, labels in val_loader:
                waveforms = waveforms.unsqueeze(1).to(device)
                labels = labels.float().to(device)
                outputs = model(waveforms)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * waveforms.size(0)
                predicted = (outputs > 0).float()
                
                # Overall accuracy
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
                # Class-wise accuracy
                val_correct_0 += ((predicted == 0) & (labels == 0)).sum().item()
                val_total_0 += (labels == 0).sum().item()
                val_correct_1 += ((predicted == 1) & (labels == 1)).sum().item()
                val_total_1 += (labels == 1).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_acc_0 = val_correct_0 / val_total_0 if val_total_0 > 0 else 0
        val_acc_1 = val_correct_1 / val_total_1 if val_total_1 > 0 else 0
        
        # Save model if validation accuracy improves
        if val_acc > best_val_acc:
            best_model_path = save_dir / f'best_model_{epoch}_{val_acc}.pth'

            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, best_model_path)
            print(f"Saved new best model with validation accuracy: {val_acc:.4f}")
        
        # Step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.3f} Train Acc: {train_acc:.3f} | "
              f"(0): {train_acc_0:.3f}(1): {train_acc_1:.3f} | "
              f"Val Loss: {val_loss:.3f} Val Acc: {val_acc:.3f} | "
              f"(0): {val_acc_0:.3f}(1): {val_acc_1:.3f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.2f}s")

    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final training accuracies - Class 0: {train_acc_0:.4f}, Class 1: {train_acc_1:.4f}")
    print(f"Final validation accuracies - Class 0: {val_acc_0:.4f}, Class 1: {val_acc_1:.4f}")
    return best_model_path

# Example of how to load the best model:
def load_best_model(model, path, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['val_acc']
