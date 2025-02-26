import torch
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import BreathingDataset
from model import RawAudioCNN
from paper_model import m5, m18
from train import train_model

if __name__ == "__main__":
    # Example data list: each element is (audio_path, start_sec, end_sec, label_str)
    data_dir = Path("data/Breath-Data")
    noise_dir = Path("data/noise")
    noise_val_dir = Path("data/noise_val")

    sounds_list = [
        data_dir / "Voice_breath.mp3",
        data_dir / "Voice_breath4.mp3",
        data_dir / "Voice_breath5.mp3",
        data_dir / "Voice_breath6.mp3",
        data_dir / "Voice_breath7.mp3",
        data_dir / "Voice_breath8.mp3",
        data_dir / "Voice_breath9.mp3",
        # data_dir / "08_male_21_TLong_denoised.mp3",
        # data_dir / "15_female_21_PPhuong_denoised.mp3",
        # data_dir / "06_male_21_QViet_denoised.mp3",
        # data_dir / "16_male_21_TTung_denoised.mp3",
        # data_dir / "22_male_21_VHung_denoised.mp3",
        # data_dir / "14_male_21_Khanh_denoised.mp3",
        # data_dir / "20_male_21_Viet_denoised.mp3",
        # data_dir / "18_male_21_Hoa_denoised.mp3",
        # data_dir / "03_male_21_BDuong_denoised.mp3",
        # data_dir / "29_male_19_Cong_denoised.mp3",
        # data_dir / "23_male_21_CNDuong_denoised.mp3",
        # data_dir / "17_male_21_Trung_denoised.mp3",
        # data_dir / "10_male_21_Nam_denoised.mp3",
        # data_dir / "24_female_21_MPham_denoised.mp3",
        # data_dir / "04_female_21_LAnh_denoised.mp3",
        # data_dir / "19_male_21_Minh_denoised.mp3",
        # data_dir / "05_male_21_NLinh_denoised.mp3",
        # data_dir / "11_female_21_Tam_denoised.mp3",
        # data_dir / "28_male_19_VHoa_asthma_denoised.mp3",
        # data_dir / "21_male_21_Hai_denoised.mp3",
        # data_dir / "07_male_21_MQuang_denoised.mp3",
        # data_dir / "27_female_19_TThanh_denoised.mp3",
        # data_dir / "01_male_23_BQuyen_denoised.mp3",
        # data_dir / "09_male_21_Ngon_denoised.mp3",
        # data_dir / "12_male_21_Tam_denoised.mp3",
        # data_dir / "13_female_20_TNhi_denoised.mp3",
        # data_dir / "02_male_22_PTuan_denoised.mp3",
        # data_dir / "26_female_19_Linh_denoised.mp3",
        data_dir / "Voice_breath2.mp3",
        data_dir / "Voice_breath10.mp3",
        data_dir / "Voice_breath11.mp3",
    ]
    
    # Split into train / validation
    train_list = sounds_list[:-3]
    val_list   = sounds_list[-3:]
    # Create datasets (e.g., 1 second max length => 16000 samples)
    train_dataset = BreathingDataset(train_list, samples_amount=1000, random_audios_folder=noise_dir)
    val_dataset = BreathingDataset(val_list, samples_amount=1000, random_audios_folder=noise_val_dir, seed=42)
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initialize model
    # model = RawAudioCNN()
    model = m18

    # Train model (use 'cuda' if you have a GPU)
    train_model(model, train_loader, val_loader, num_epochs=40, lr=1e-3, device='mps')
