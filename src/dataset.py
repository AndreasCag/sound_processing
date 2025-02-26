import random
import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path

label_map = {
    "deep": 1,
    "heavy": 1,
    "normal": 1,
    "nothing": 0,
}


class BreathingDataset(Dataset):
    def __init__(
        self,
        sounds_list,
        random_audios_folder,
        transform=None,
        sample_rate=16000,
        sample_length_sec=4,
        samples_amount=1000,
        seed=None
    ):
        self.sounds_list = sounds_list
        self.transform = transform

        self.sample_rate = sample_rate
        self.sample_length_sec = sample_length_sec
        self.samples_amount = samples_amount
        self.seed = seed

        self.load_data(sounds_list)
        self.load_random_audio(random_audios_folder)

    def load_random_audio(self, random_audios_folder):
        random_audio_files = list(random_audios_folder.glob("*.*"))

        random_audios = []

        for audio_file in random_audio_files:
            wave, sample_rate = torchaudio.load(audio_file, normalize = True)

            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                wave = resampler(wave)
            
            random_audios.append({
                "stem": audio_file.stem,
                "soundwave": wave,
                "duration": wave.shape[1] / self.sample_rate
            })
        
        self.random_audios = random_audios

        print(f"Loaded random audios: {len(random_audios)}")


    def load_data(self, sounds_list):
        sounds = []

        for sound_path_str in sounds_list:
            sound_path = Path(sound_path_str)
            wave, sample_rate = torchaudio.load(sound_path, normalize = True)
            print(f"Loaded {sound_path.stem} - {wave.shape}")

            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                wave = resampler(wave)

            labels = []
            labels_path = sound_path.parent / "label" / (sound_path.stem.replace('_denoised', '') + ".txt")

            with labels_path.open("r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    start_time, end_time, label = line.strip().split("\t")
                    labels.append((float(start_time), float(end_time), label))

            sounds.append(
                {
                    "stem": sound_path.stem,
                    "soundwave": wave,
                    "rate": sample_rate,
                    "labels": labels,
                    "duration": wave.shape[1] / sample_rate,
                }
            )

        self.sounds = sounds


    def __len__(self):
        return self.samples_amount
    
    def get_random_sample(self):
        sound = random.choice(self.sounds)
        time_start = random.random() * (sound["duration"] - self.sample_length_sec)
        time_end = time_start + self.sample_length_sec

        # Find all labels that overlap with our time window and their durations
        relevant_labels = {}
        total_labels_duration = 0
        for start, end, label in sound["labels"]:
            if start < time_end and end > time_start:
                if label not in relevant_labels:
                    relevant_labels[label] = 0
                # Calculate overlap duration
                overlap_start = max(start, time_start)
                overlap_end = min(end, time_end)
                overlap_duration = overlap_end - overlap_start

                relevant_labels[label] += overlap_duration
                total_labels_duration += overlap_duration

        relevant_labels["nothing"] = self.sample_length_sec - total_labels_duration

        biggest_label = ""
        biggest_label_duration = 0
        for label in relevant_labels.keys():
            if relevant_labels[label] > biggest_label_duration:
                biggest_label_duration = relevant_labels[label]
                biggest_label = label
        # print(sound["stem"])
        # print(sound["rate"])
        # print(time_start, time_end)
        # print(relevant_labels)
        # print(f"{sound['stem']} - {time_start} - {time_end}")

        waveform = sound["soundwave"][
            :,
            int(time_start * self.sample_rate) : int(time_end * self.sample_rate),
        ]

        if self.transform:
            waveform = self.transform(waveform)

        # Just take the first channel:
        waveform = waveform[0, :]  # shape: [max_length]

        # Convert label to tensor
        # label_tensor = torch.tensor(label, dtype=torch.long)

        # return waveform, label_tensor
        return waveform, label_map[biggest_label]

    def get_random_noise(self):
        should_generate_synthetic = random.random()

        if (should_generate_synthetic > 0.9):
            return self.generate_synthetic_noise('white')
        if (should_generate_synthetic > 0.8):
            return self.generate_synthetic_noise('pink')
        if (should_generate_synthetic > 0.7):
            return self.generate_synthetic_noise('brown')

        sound = random.choice(self.random_audios)
        time_start = random.random() * (sound['duration'] - self.sample_length_sec)
        time_end = time_start + self.sample_length_sec
        # print(f"{sound['stem']} - {time_start} - {time_end}")

        waveform = sound["soundwave"][
            :,
            int(time_start * self.sample_rate) : int(time_end * self.sample_rate),
        ]

        if self.transform:
            waveform = self.transform(waveform)

        waveform = waveform[0, :]

        return waveform, 0

    def __getitem__(self, idx):
        # Set seed based on index if seed is provided
        if self.seed is not None:
            # Use a different seed for each index but make it reproducible
            random.seed(self.seed + idx)

        expected_label = 0 if random.random() > 0.35 else 1

        if expected_label == 0:
            rnd_noise = self.get_random_noise()

            if self.seed is not None:
                # Use a different seed for each index but make it reproducible
                random.seed(None)

            return rnd_noise

        while True:
            wave, label = self.get_random_sample()
            if label == expected_label:
                break

        if self.seed is not None:
            # Use a different seed for each index but make it reproducible
            random.seed(None)
        return wave, label

    def generate_synthetic_noise(self, noise_type="white"):
        num_samples = int(self.sample_rate * self.sample_length_sec)
        
        if noise_type == "white":
            # Generate white noise (uniform distribution between -1 and 1)
            noise = torch.rand(num_samples) * 2 - 1
        
        elif noise_type == "pink":
            # Generate pink noise (1/f spectrum)
            white = torch.rand(num_samples) * 2 - 1
            # Create pink noise by applying 1/f filter in frequency domain
            freqs = torch.fft.rfftfreq(num_samples)
            spectrum = torch.fft.rfft(white)
            pink_filter = 1 / torch.sqrt(freqs[1:])  # 1/f filter
            spectrum[1:] *= pink_filter
            noise = torch.fft.irfft(spectrum, n=num_samples)
            # Normalize
            noise = noise / noise.abs().max()
        
        elif noise_type == "brown":
            # Generate brown noise (random walk / integrated white noise)
            white = torch.rand(num_samples) * 2 - 1
            noise = torch.cumsum(white, dim=0)
            # Normalize
            noise = noise / noise.abs().max()
        
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # Ensure the noise is in the right format and scale
        noise = noise.float()
        
        return noise, 0

if __name__ == "__main__":
    # Initialize dataset with example parameters
    data_dir = Path("data/Breath-Data")
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
        data_dir / "26_female_19_Linh.wav",
    ]
    print(f"Found {len(sounds_list)} WAV files")

    dataset = BreathingDataset(
        sounds_list=sounds_list,
        sample_rate=16000,
        sample_length_sec=4,
        samples_amount=1000,
    )

    # Load the data
    dataset.load_data(dataset.sounds_list)

    # Collect statistics for 1000 samples
    label_counts = {}

    for i in range(1000):
        waveform, label = dataset[i]

        # Update counts
        label_counts[label] = label_counts.get(label, 0) + 1

    print("\nLabel Statistics across 1000 samples:")
    print("\nLabel Counts:")
    total_samples = sum(label_counts.values())
    for label, count in sorted(label_counts.items()):
        percentage = (count / total_samples) * 100
        print(f"{label}: {count} times ({percentage:.1f}%)")
