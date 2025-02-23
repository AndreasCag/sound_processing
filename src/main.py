import torch
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import BreathingDataset
from model import RawAudioCNN
from train import train_model
from augmentations import train_augment

if __name__ == "__main__":
    # Example data list: each element is (audio_path, start_sec, end_sec, label_str)
    data_dir = Path("data/Breath-Data")
    noise_dir = Path("data/noise")
    noise_val_dir = Path("data/noise_val")

    sounds_list = [
        data_dir / "08_male_21_TLong.wav",
        data_dir / "15_female_21_PPhuong.wav",
        data_dir / "06_male_21_QViet.wav",
        data_dir / "16_male_21_TTung.wav",
        data_dir / "22_male_21_VHung.wav",
        data_dir / "14_male_21_Khanh.wav",
        data_dir / "20_male_21_Viet.wav",
        data_dir / "18_male_21_Hoa.wav",
        data_dir / "03_male_21_BDuong.wav",
        data_dir / "29_male_19_Cong.wav",
        data_dir / "23_male_21_CNDuong.wav",
        data_dir / "17_male_21_Trung.wav",
        data_dir / "10_male_21_Nam.wav",
        data_dir / "24_female_21_MPham.wav",
        data_dir / "04_female_21_LAnh.wav",
        data_dir / "19_male_21_Minh.wav",
        data_dir / "05_male_21_NLinh.wav",
        data_dir / "11_female_21_Tam.wav",
        data_dir / "28_male_19_VHoa_asthma.wav",
        data_dir / "21_male_21_Hai.wav",
        data_dir / "07_male_21_MQuang.wav",
        data_dir / "27_female_19_TThanh.wav",
        data_dir / "01_male_23_BQuyen.wav",
        data_dir / "09_male_21_Ngon.wav",
        data_dir / "12_male_21_Tam.wav",
        data_dir / "13_female_20_TNhi.wav",
        data_dir / "02_male_22_PTuan.wav",
        data_dir / "26_female_19_Linh.wav"
    ]
    
    # Split into train / validation
    train_list = sounds_list[:-7]
    val_list   = sounds_list[-7:]
    # Create datasets (e.g., 1 second max length => 16000 samples)
    train_dataset = BreathingDataset(train_list, random_audios_folder=noise_dir)
    val_dataset = BreathingDataset(val_list, random_audios_folder=noise_val_dir, seed=42)
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initialize model
    model = RawAudioCNN(num_classes=2)

    # Train model (use 'cuda' if you have a GPU)
    train_model(model, train_loader, val_loader, num_epochs=13, lr=1e-3, device='mps')
