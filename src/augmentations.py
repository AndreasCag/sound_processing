import torch
import random

class RandomGain:
    """Randomly apply a gain (volume change) to the waveform."""
    def __init__(self, gain_range=(0.8, 1.2)):
        """
        Args:
            gain_range: (min_gain, max_gain) for volume scaling
        """
        self.gain_min, self.gain_max = gain_range

    def __call__(self, waveform):
        """
        waveform: Tensor of shape [channels, num_samples] 
                  or [num_samples] if single-channel
        """
        gain = random.uniform(self.gain_min, self.gain_max)
        return waveform * gain
    
class AdditiveNoise:
    """Add random Gaussian noise to the waveform."""
    def __init__(self, noise_std_range=(0.0, 0.02)):
        """
        Args:
            noise_std_range: (min_std, max_std) for noise amplitude
        """
        self.noise_std_range = noise_std_range
    
    def __call__(self, waveform):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # shape: [1, num_samples]
        
        noise_std = random.uniform(*self.noise_std_range)
        if noise_std > 0.0:
            noise = torch.randn_like(waveform) * noise_std
            waveform = waveform + noise
        return waveform

class TimeShift:
    """Randomly circular-shift the waveform in time by up to shift_max_frac * length."""
    def __init__(self, shift_max_frac=0.1):
        self.shift_max_frac = shift_max_frac

    def __call__(self, waveform):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, num_samples]
        
        _, num_samples = waveform.shape
        shift_amt = int(random.uniform(-self.shift_max_frac, self.shift_max_frac) * num_samples)
        if shift_amt != 0:
            # circular shift
            waveform = torch.roll(waveform, shifts=shift_amt, dims=-1)
        return waveform
    
class ComposeTransforms:
    """Apply a list of transforms in sequence."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, waveform):
        for t in self.transforms:
            waveform = t(waveform)
        return waveform

# Example usage:
train_augment = ComposeTransforms([
    RandomGain(gain_range=(0.8, 1.2)),
    TimeShift(shift_max_frac=0.1),
    # AdditiveNoise(noise_std_range=(0.0, 0.02))
])