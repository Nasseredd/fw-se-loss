# author: @Nasser

import sys 
import torch
import torchaudio
import torch.nn as nn
import soundfile as sf
import torch.nn.functional as F
torch.manual_seed(42)


# Utils
def validate_input_dimensions(est_targets, targets):
    """
    Validate the dimensions of est_targets and targets.

    Args:
        est_targets (torch.Tensor): Estimated targets with shape [batch_size, n_sources, n_samples].
        targets (torch.Tensor): Ground truth targets with shape [batch_size, n_sources, n_samples].

    Raises:
        AssertionError: If any of the dimension checks fail.
    """
    assert est_targets.dim() == 3, "est_targets must have 3 dimensions: [batch_size, n_sources, n_samples]"
    assert targets.dim() == 3, "targets must have 3 dimensions: [batch_size, n_sources, n_samples]"
    assert est_targets.size(1) >= 1, "est_targets must have at least one source in dim=1"
    assert targets.size(1) >= 1, "targets must have at least one source in dim=1"

def select_sources(est_targets, targets):
    """
    Efficiently select the first source (speech) and second source (noise) from the input tensors.

    Args:
        est_targets (torch.Tensor): Tensor of estimated targets with shape [batch_size, n_sources, n_samples].
        targets (torch.Tensor): Tensor of ground truth targets with shape [batch_size, n_sources, n_samples].

    Returns:
        tuple: 
            - est_speech (torch.Tensor): First source (speech) from estimated targets, shape [batch_size, n_samples].
            - target_speech (torch.Tensor): First source (speech) from ground truth targets, shape [batch_size, n_samples].
            - target_noise (torch.Tensor): Second source (noise) from ground truth targets, shape [batch_size, n_samples].
    """
    return est_targets[:, 0, :], targets[:, 0, :], targets[:, 1, :]

def compute_speech_and_noise_fft(target_speech, target_noise):
    """
    Compute the frequency-domain representations of target speech and target noise.

    Args:
        target_speech (torch.Tensor): Time-domain target speech signal, shape [batch_size, n_samples].
        target_noise (torch.Tensor): Time-domain target noise signal, shape [batch_size, n_samples].

    Returns:
        tuple:
            - target_speech_freq (torch.Tensor): Frequency-domain representation of target speech, shape [batch_size, n_freq_bins].
            - target_noise_freq (torch.Tensor): Frequency-domain representation of target noise, shape [batch_size, n_freq_bins].
    """
    # Stack the time-domain signals along a new dimension
    stacked = torch.stack([target_speech, target_noise], dim=0)
    
    # Perform real FFT along the last dimension
    freq_results = torch.fft.rfft(stacked, dim=-1)
    
    # Split the results into target speech and target noise frequency components
    target_speech_freq, target_noise_freq = freq_results[0], n_freq_results[1]
    
    return target_speech_freq, target_noise_freq

def compute_speech_and_noise_stft(target_speech, target_noise, n_fft=512, hop_length=None, win_length=None, window=None):
    """
    Compute the STFT (Short-Time Fourier Transform) representations of target speech and target noise.

    Args:
        target_speech (torch.Tensor): Time-domain target speech signal, shape [batch_size, n_samples].
        target_noise (torch.Tensor): Time-domain target noise signal, shape [batch_size, n_samples].
        n_fft (int): Number of FFT points. Default is 512.
        hop_length (int, optional): Number of samples between successive frames. Defaults to n_fft // 4 if not provided.
        win_length (int, optional): Window size. Defaults to n_fft if not provided.
        window (torch.Tensor, optional): Window function applied to each frame. Default is None (rectangular window).

    Returns:
        tuple:
            - target_speech_stft (torch.Tensor): STFT representation of target speech, shape [batch_size, n_freq_bins, time_frames].
            - target_noise_stft (torch.Tensor): STFT representation of target noise, shape [batch_size, n_freq_bins, time_frames].
    """
    if hop_length is None:
        hop_length = n_fft // 4
    if win_length is None:
        win_length = n_fft
    if window is None:
        window = torch.hann_window(win_length)

    # Compute the STFT for the speech and noise signals
    target_speech_stft = torch.stft(
        target_speech, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, return_complex=True
    )
    target_noise_stft = torch.stft(
        target_noise, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, return_complex=True
    )
    
    return target_speech_stft, target_noise_stft

def compute_snr_db(target_speech_freq, target_noise_freq, eps=1e-8, snr_min=-10, snr_max=35):
    """
    Compute the Signal-to-Noise Ratio (SNR) in decibels (dB) given the frequency-domain representations 
    of speech and noise.

    Args:
        target_speech_freq (torch.Tensor): Frequency-domain representation of the target speech signal.
        target_noise_freq (torch.Tensor): Frequency-domain representation of the target noise signal.
        eps (float, optional): A small constant to avoid division by zero. Default is 1e-8.
        snr_min (float, optional): Minimum SNR value to clamp to. Default is -50 dB.
        snr_max (float, optional): Maximum SNR value to clamp to. Default is 50 dB.

    Returns:
        torch.Tensor: The SNR in decibels (dB), clamped to the range [snr_min, snr_max].
    """
    # Compute the Power Spectral Density (PSD) for speech and noise
    target_speech_psd = torch.abs(target_speech_freq) ** 2 + eps
    target_noise_psd = torch.abs(target_noise_freq) ** 2 + eps

    # Compute SNR in decibels
    snr_db = 10 * (torch.log10(target_speech_psd) - torch.log10(target_noise_psd))

    # Clamp the SNR to the specified range
    snr_db = torch.clamp(snr_db, min=snr_min, max=snr_max)

    return snr_db, target_speech_psd, target_noise_psd

def compute_projection_and_distortion(est_speech, target_speech, eps=1e-8):
    """
    Compute the projection of the estimated speech onto the target speech
    and the residual noise.

    Args:
        est_speech (torch.Tensor): Estimated speech signal, shape [batch_size, n_samples].
        target_speech (torch.Tensor): Target speech signal, shape [batch_size, n_samples].
        eps (float, optional): A small constant to avoid division by zero. Default is 1e-8.

    Returns:
        tuple:
            - proj (torch.Tensor): Projection of the estimated speech onto the target speech, shape [batch_size, n_samples].
            - e_dist (torch.Tensor): Residual noise after removing the projection, shape [batch_size, n_samples].
    """
    # Compute the dot product <s_hat, s>
    dot = torch.sum(est_speech * target_speech, dim=-1, keepdim=True)
    
    # Compute the energy of the target speech <s, s>
    target_energy = torch.sum(target_speech**2, dim=-1, keepdim=True) + eps
    
    # Compute the projection (<s_hat, s> / <s, s>) * s
    proj = dot * target_speech / target_energy
    
    # Compute the residual noise e_dist = s_hat - proj
    e_dist = est_speech - proj
    
    return proj, e_dist

def compute_e_interf(est_speech, target_noise, eps=1e-8):
    # Compute the dot product <s_hat, n>
    dot = torch.sum(est_speech * target_noise, dim=-1, keepdim=True)

    # Compute the energy of the target_noise <n, n>
    target_noise_energy = torch.sum(target_noise**2, dim=-1, keepdim=True) + eps
    
    # Compute the projection (<s_hat, n> / <n, n>) * n
    e_interf = (dot / target_noise_energy) * target_noise

    return e_interf



# FFT 
def signal_to_critic_band_spec_fft(signal, sr, cent_freq, bandwidth, n_fft=1024, eps=1e-8):
    """
    Converts a signal into a critical band spectrogram using the FFT method.
    
    Args:
        signal (Tensor): Input time-domain signal (1D tensor).
        sr (int): Sampling rate.
        cent_freq (Tensor): Tensor of center frequencies of critical bands.
        bandwidth (Tensor): Tensor of bandwidths corresponding to each center frequency.
        n_fft (int): FFT size. Default is 1024.
        eps (float): Small value to avoid log(0). Default is 1e-8.
    
    Returns:
        Tensor: 1D tensor representing the energy in each critical band.
    """
    
    max_freq = sr / 2  # Nyquist frequency
    num_crit = 25  # Total number of critical bands to simulate
    n_fftby4 = int(n_fft / 2)  # We're only interested in the positive frequencies

    # Get the smallest bandwidth to normalize the others
    bw_min = bandwidth[0]
    
    # Define a minimum threshold for filter values
    min_factor = torch.exp(torch.tensor(-30.0 / (2.0 * 2.303), dtype=torch.float64))  # ≈ -30 dB threshold

    # Initialize the filter bank matrix: each row corresponds to one critical band
    crit_filter = torch.zeros((num_crit, n_fftby4), dtype=torch.float64, device=signal.device)
    
    # Frequency bins (0 to n_fft/2)
    j = torch.arange(0, n_fftby4, dtype=torch.float64)

    # Construct Gaussian-like filters for each critical band
    for i in range(num_crit):
        # Normalize center frequency and bandwidth to FFT bins
        f0 = (cent_freq[i] / max_freq) * n_fftby4
        bw = (bandwidth[i] / max_freq) * n_fftby4
        
        # Normalize filter height with respect to bandwidth
        norm_factor = torch.log(bw_min + eps) - torch.log(bandwidth[i] + eps)
        
        # Gaussian shape centered at f0, scaled by norm_factor
        crit_filter[i, :] = torch.exp(-11 * ((j - torch.floor(f0)) / bw) ** 2 + norm_factor)
        
        # Apply a minimum threshold (cut very low values)
        crit_filter[i, :] *= (crit_filter[i, :] > min_factor)

    # Compute FFT of the signal and keep only positive frequencies
    fft_spectrum = torch.fft.fft(signal, n=n_fft)[:n_fftby4].abs()

    # Project the spectrum onto the critical band filter bank
    critical_band_spectrum = crit_filter.matmul(fft_spectrum.to(torch.float64))

    return critical_band_spectrum


# STFT
def signal_to_critic_band_spec_stft(signal_batch, sr, cent_freq, bandwidth, n_fft=1024, winlength=1024, hop_length=256, frameLen=0.03, overlap=0.75, eps=1e-8):
    """
    Compute critical band spectrogram from a batch of time-domain signals using STFT.
    
    Args:
        signal_batch (Tensor): Shape [batch_size, n_samples]
        sr: Sampling rate
        cent_freq, bandwidth: Torch tensors defining critical band parameters
        ...
    
    Returns:
        Tensor: [batch_size, n_crit_bands, n_frames]
    """
    # winlength = winlength #round(frameLen * sr)
    skiprate = hop_length #int((1 - overlap) * frameLen * sr)
    max_freq = sr / 2
    num_crit = len(cent_freq)
    n_fftby4 = n_fft // 2
    bw_min = bandwidth[0]
    min_factor = torch.exp(torch.tensor(-30.0 / (2.0 * 2.303), dtype=torch.float64))

    crit_filter = torch.zeros((num_crit, n_fftby4), dtype=torch.float64, device=signal_batch.device)
    j = torch.arange(0, n_fftby4, dtype=torch.float64)

    for i in range(num_crit):
        f0 = (cent_freq[i] / max_freq) * (n_fftby4)
        bw = (bandwidth[i] / max_freq) * (n_fftby4)
        norm_factor = torch.log(bw_min) - torch.log(bandwidth[i])
        crit_filter[i, :] = torch.exp(-11 * (((j - torch.floor(f0)) / bw) ** 2) + norm_factor)
        crit_filter[i, :] = crit_filter[i, :] * (crit_filter[i, :] > min_factor)

    hannWin = torch.hann_window(winlength).to(signal_batch.device)

    all_specs = []
    for signal in signal_batch:  # shape: [n_samples]
        # Check signal length
        if signal.size(0) < winlength:
            pad = winlength - signal.size(0)
            signal = F.pad(signal, (0, pad))

        spec = torch.stft(
            signal, n_fft=n_fft, hop_length=skiprate, win_length=winlength,
            window=hannWin, return_complex=True
        ).abs()  # [n_freq, n_frames]

        spec = spec[:-1, :]  # remove last bin to match filter
        spec_cb = crit_filter.matmul(spec.to(torch.float64))  # [n_crit_bands, n_frames]
        all_specs.append(spec_cb)

    return torch.stack(all_specs, dim=0)  # [batch_size, n_crit_bands, n_frames]

def signal_to_mel_stft(x, sr=16000, n_fft=1024, hop_length=256, n_mels=25):
    """
    Convert signal x to Mel-scaled STFT magnitude spectrogram.
    
    Args:
        x (torch.Tensor): Input signal of shape (batch_size, samples)
        sr (int): Sample rate (default: 16000)
        n_fft (int): FFT size (default: 1024)
        hop_length (int): Hop size (default: 256)
        n_mels (int): Number of Mel bands (default: 25)
    
    Returns:
        torch.Tensor: Mel spectrogram of shape (batch_size, n_mels, time_frames)
    """
    # 1. Compute STFT (returns complex-valued spectrogram)
    window = torch.hann_window(n_fft).to(x.device)
    stft = torch.stft(x, 
                     n_fft=n_fft, 
                     hop_length=hop_length, 
                     win_length=n_fft,
                     window=window,
                     center=True,
                     pad_mode='reflect',
                     return_complex=True)  # [B, F, T]
    
    # 2. Convert to magnitude spectrogram
    mag_spec = torch.abs(stft)  # [B, F, T]
    
    # 3. Initialize Mel scale transform
    mel_scale = torchaudio.transforms.MelScale(
        n_mels=n_mels,
        sample_rate=sr,
        n_stft=n_fft // 2 + 1,  # 513 for n_fft=1024
        f_min=0.0,
        f_max=sr/2,
        norm=None,
        mel_scale='htk'
    ).to(x.device)
    
    # 4. Convert to Mel scale (requires [..., freq, time] input)
    mel_spec = mel_scale(mag_spec.T).T 
    
    return mel_spec

# Losses
class XP1Loss(nn.Module):
    """
    Negative SI-SDR on the waveform (baseline)
    """
    def __init__(self, zero_mean=True, take_log=True, EPS=1e-8):
        super(XP1Loss, self).__init__()
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = EPS
    
    def forward(self, est_targets: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the negative SI-SDR loss.

        Args:
            est_targets (torch.Tensor): Estimated signals, shape [batch_size, n_sources, n_samples].
            targets (torch.Tensor): Target signals, shape [batch_size, n_sources, n_samples].

        Returns:
            torch.Tensor: Negative SI-SDR for each batch, shape [batch_size].
        """
        # Ensure the inputs have the expected dimensions
        validate_input_dimensions(est_targets, targets) # [batch_size, n_sources, n_samples]

        # Select target speech, target noise and estimated speech 
        est_speech, target_speech, _ = select_sources(est_targets, targets) # [batch_size, n_samples]

        # Zero-mean norm
        if self.zero_mean:
            target_speech -= torch.mean(target_speech, dim=1, keepdim=True) # [batch_size, n_samples]
            est_speech -= torch.mean(est_speech, dim=1, keepdim=True) # [batch_size, n_samples]
        
        # Compute the projections: proj and e_dist
        proj, e_dist = compute_projection_and_distortion(est_speech, target_speech) # [batch_size, n_samples]
        
        # SI-SDR computation
        sdr = torch.sum(proj**2, dim=-1) / (torch.sum(e_dist**2, dim=-1) + self.EPS) # [batch_size, ]

        if self.take_log:
            sdr = 10 * torch.log10(sdr + self.EPS) # [batch_size, ]

        return -sdr.mean()

class NegSISDRLoss(nn.Module):
    """SI-SDR computed using STFT."""
    def __init__(self, zero_mean=True, take_log=True, EPS=1e-8, sr=16000,
                 scale='mel', n_fft=1024, hop_length=256, n_bins=18, weights_type=None, gamma=0.2):
        super(NegSISDRLoss, self).__init__()
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = EPS 
        self.scale = scale
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.sr = sr
        self.n_bins = int(n_bins)
        self.weights_type = weights_type
        self.gamma = gamma

        if self.n_bins == 25:
            self.cent_freq = torch.tensor([50.0, 120.0, 190.0, 260.0, 330.0, 400.0, 470.0, 540.0, 617.372, 703.378, 798.717, 904.128, 1020.38, 1148.3, 1288.72, 1442.54, 1610.7, 1794.16, 1993.93, 2211.08, 2446.71, 2701.97, 2978.04, 3276.17, 3597.63])
            self.bandwidth = torch.tensor([70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 77.3724, 86.0056, 95.3398, 105.411, 116.256, 127.914, 140.423, 153.823, 168.154, 183.457, 199.776, 217.153, 235.631, 255.255, 276.072, 298.126, 321.465, 346.136])
            if self.weights_type == 'perceptual':
                self.weights = torch.tensor([0.003, 0.003, 0.003, 0.007, 0.010, 0.016, 0.016, 0.017, 0.017, 0.022, 0.027, 0.028, 0.030, 0.032, 0.034, 0.035, 0.037, 0.036, 0.036, 0.033, 0.030, 0.029, 0.027, 0.026, 0.026])
        
        elif self.n_bins == 21:
            self.cent_freq = torch.tensor([150, 250, 350, 450, 570, 700, 840, 1000, 1170, 1370, 1600, 1850, 2150, 2500, 2900, 3400, 4000, 4800, 5800, 7000, 8500])
            self.bandwidth = torch.tensor([100, 100, 100, 110, 120, 140, 150, 160, 190, 210, 240, 280, 320, 380, 450, 550, 700, 900, 1100, 1200, 1900])
            if self.weights_type == 'perceptual':
                self.weights = torch.tensor([0.0103, 0.0261, 0.0419, 0.0577, 0.0577, 0.0577, 0.0577, 0.0577, 0.0577, 0.0577, 0.0577, 0.0577, 0.1576, 0.1576, 0.1576, 0.1576, 0.1576, 0.0646, 0.0343, 0.0226, 0.0110])
        
        elif self.n_bins == 18:
            self.cent_freq = torch.tensor([160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000])
            self.bandwidth = torch.tensor([37.05, 46.31, 57.90, 72.95, 92.62, 115.78, 145.88, 185.25, 231.56, 289.46, 370.50, 463.12, 578.91, 729.43, 926.26, 1157.82, 1458.85, 1852.51])
            if self.weights_type == 'perceptual':
                self.weights = torch.tensor([0.0083, 0.0095, 0.0150, 0.0289, 0.0440, 0.0578, 0.0653, 0.0711, 0.0818, 0.0844, 0.0882, 0.0898, 0.0868, 0.0844, 0.0771, 0.0527, 0.0364, 0.0185])
            elif self.weights_type == 'perceptual-2':
                self.corr = torch.tensor([15.65, 16.65, 17.65, 18.65, 19.65, 20.65, 21.65, 22.65, 23.65, 24.65, 25.65, 26.65, 27.65, 28.65, 29.65, 30.65, 31.65, 32.65])
                self.weights = 10**(self.corr/10)
            elif self.weights_type == 'perceptual-3':
                self.corr = torch.tensor([15.65, 16.65, 17.65, 18.65, 19.65, 20.65, 21.65, 22.65, 23.65, 24.65, 25.65, 26.65, 27.65, 28.65, 29.65, 30.65, 31.65, 32.65])
                self.corr = 10**(self.corr/10)
                self.weights = torch.tensor([0.0083, 0.0095, 0.0150, 0.0289, 0.0440, 0.0578, 0.0653, 0.0711, 0.0818, 0.0844, 0.0882, 0.0898, 0.0868, 0.0844, 0.0771, 0.0527, 0.0364, 0.0185])
                self.weights = self.corr * self.weights
        
        elif self.n_bins == -1:
            pass

        else: 
            print("\033[91m[ERROR]\033[0m n_bins not known!")
            sys.exit()

    def forward(self, est_targets: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        
        
        validate_input_dimensions(est_targets, targets) # Input validation
        est_speech, target_speech, target_noise = select_sources(est_targets, targets) # Select estimated speech, target speech and target noise tensors

        # 
        if self.zero_mean:
            target_speech -= target_speech.mean(dim=-1, keepdim=True)
            est_speech -= est_speech.mean(dim=-1, keepdim=True)
            target_noise -= target_noise.mean(dim=-1, keepdim=True)

        # ---------------------------------------------------------------------------------------------
        # --------------------------- Speech Target and Distortion Error ------------------------------
        # ---------------------------------------------------------------------------------------------

        proj, e_dist = compute_projection_and_distortion(est_speech, target_speech) # Compute projections

        if self.scale == 'linear': 
            proj_stft = torch.stft(proj, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, return_complex=True)
            e_dist_stft = torch.stft(e_dist, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, return_complex=True)

        elif self.scale == 'critical':
            proj_stft = signal_to_critic_band_spec_stft(proj, sr=self.sr, cent_freq=self.cent_freq, bandwidth=self.bandwidth, n_fft=self.n_fft, winlength=self.n_fft, hop_length=self.hop_length) # [batch_size, n_freq_bins, time_frames]
            e_dist_stft = signal_to_critic_band_spec_stft(e_dist, sr=self.sr, cent_freq=self.cent_freq, bandwidth=self.bandwidth, n_fft=self.n_fft, winlength=self.n_fft, hop_length=self.hop_length) # [batch_size, n_freq_bins, time_frames]

        elif self.scale == 'mel':
            proj_stft = signal_to_mel_stft(proj, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_bins)
            e_dist_stft = signal_to_mel_stft(e_dist, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_bins)
        
        else:
            print('\033[91m[ERROR]\033[0m The specified scale is not valid!')
            sys.exit()

        # ---------------------------------------------------------------------------------------------
        # -------------------------------------------- WEIGHTS ----------------------------------------
        # ---------------------------------------------------------------------------------------------
        
        if 'perceptual' in self.weights_type:
            device = proj_stft.device
            weights = self.weights.to(device).view(1, len(self.weights), 1).repeat(proj_stft.size(0), 1, proj_stft.size(2)) # Broadcast weights
        
        else:
            # ******************************** Compute the STFT of the target speech and the target noise (linear, critical or mel) ********************************
            if self.scale == 'linear':
                target_speech_stft = torch.stft(target_speech, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, return_complex=True)
                target_noise_stft = torch.stft(target_noise, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft, return_complex=True)

            elif self.scale == 'mel':
                target_speech_stft = signal_to_mel_stft(target_speech, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_bins) # (batch_size, freq_bins, time_frames)
                target_noise_stft = signal_to_mel_stft(target_noise, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_bins) # (batch_size, freq_bins, time_frames)
            
            elif self.scale == 'critical':
                target_speech_stft = signal_to_critic_band_spec_stft(target_speech, sr=self.sr, cent_freq=self.cent_freq, bandwidth=self.bandwidth, n_fft=self.n_fft, winlength=self.n_fft, hop_length=self.hop_length)
                target_noise_stft = signal_to_critic_band_spec_stft(target_noise, sr=self.sr, cent_freq=self.cent_freq, bandwidth=self.bandwidth, n_fft=self.n_fft, winlength=self.n_fft, hop_length=self.hop_length)
            
            else: 
                print('\033[91m[ERROR]\033[0m The specified scale is not valid!')
                sys.exit()

            # ******************************** Compute the weight ********************************

            if self.weights_type == 'speech_norm':
                weights = torch.abs(target_speech_stft) ** self.gamma
                weights = weights.to(proj_stft.device)
            
            elif self.weights_type == 'log_snr_speech_norm':
                # Compute per-band SNR
                power_speech = torch.mean(torch.abs(target_speech_stft) ** 2, dim=-1) # (batch_size, freq_bins)
                power_noise = torch.mean(torch.abs(target_noise_stft) ** 2, dim=-1) # (batch_size, freq_bins)
                snr = power_speech / (power_noise + self.EPS) # (batch_size, freq_bins)
                # log(SNR)
                log_snr = torch.log10(snr + self.EPS) # (batch_size, freq_bins)
                log_snr = log_snr.unsqueeze(-1).expand_as(target_speech_stft) # add a dim and expand to shape: (batch_size, freq_bins, time_frames)
                # |S|^\gamma 
                power_spectrum = torch.abs(target_speech_stft) ** self.gamma # (batch_size, freq_bins, time_frames)
                # Weights
                weights = - log_snr * power_spectrum # -log(SNR) x |S|^\gamma 
                weights = weights.to(proj_stft.device)
            
            elif self.weights_type == 'log_snr_softmax_speech_norm':
                # Compute per-band SNR
                power_speech = torch.mean(torch.abs(target_speech_stft) ** 2, dim=-1)
                power_noise = torch.mean(torch.abs(target_noise_stft) ** 2, dim=-1)
                snr = power_speech / (power_noise + self.EPS)
                log_snr = torch.log10(snr + self.EPS) # (batch_size, freq_bins)
                log_snr = log_snr.unsqueeze(-1).expand_as(target_speech_stft) # add a dim and expand to shape: (batch_size, freq_bins, time_frames)
                # softmax_speech_norm
                softmax = torch.softmax(torch.abs(target_speech_stft), dim=1)
                # Weights
                weights = - log_snr * softmax # -log(SNR) x softmax(|S|)
                weights = weights.to(proj_stft.device)
            
            elif self.weights_type == 'softmax_inverse_snr_speech_norm':
                # Compute per-band SNR
                power_speech = torch.mean(torch.abs(target_speech_stft) ** 2, dim=-1)
                power_noise = torch.mean(torch.abs(target_noise_stft) ** 2, dim=-1)
                snr = power_speech / (power_noise + self.EPS)
                # softmax(1/SNR)
                softmax_inverse = torch.softmax(1/snr, dim=1) # (batch_size, freq_bins)
                softmax_inverse = softmax_inverse.unsqueeze(-1).expand_as(target_speech_stft) # add a dim and expand to shape: (batch_size, freq_bins, time_frames)
                # |S|^\gamma 
                power_spectrum = torch.abs(target_speech_stft) ** self.gamma # (batch_size, freq_bins, time_frames)
                # Weights
                weights = softmax_inverse * power_spectrum # softmax(1/SNR) x |S|^\gamma 
                weights = weights.to(proj_stft.device)

            elif self.weights_type == 'inverse_spectral_diff':
                spectral_diff = torch.abs(torch.abs(target_speech_stft) - torch.abs(target_noise_stft))
                weights = 1.0 / (spectral_diff + self.EPS)
                weights = weights.to(proj_stft.device)

            elif self.weights_type == 'softmax':
                # Compute per-band SNR
                power_speech = torch.mean(torch.abs(target_speech_stft) ** 2, dim=-1)
                power_noise = torch.mean(torch.abs(target_noise_stft) ** 2, dim=-1)
                snr = power_speech / (power_noise + self.EPS)
                # Compute weights: softmax(-SNR)
                weights = torch.softmax(-snr, dim=1).unsqueeze(-1).expand_as(proj_stft)  
                weights = weights.to(proj_stft.device)
            
            elif self.weights_type == 'softmax_ps':
                # Compute per-band SNR
                power_speech = torch.mean(torch.abs(target_speech_stft) ** 2, dim=-1)
                power_noise = torch.mean(torch.abs(target_noise_stft) ** 2, dim=-1)
                snr = power_speech / (power_noise + self.EPS)
                # Compute weights: softmax(-SNR)
                weights = torch.softmax(-snr, dim=1) * power_speech
                weights = weights.unsqueeze(-1).expand_as(proj_stft) 
                weights = weights.to(proj_stft.device)

            elif self.weights_type == 'softmax_log':
                # Compute per-band SNR
                power_speech = torch.mean(torch.abs(target_speech_stft) ** 2, dim=-1)
                power_noise = torch.mean(torch.abs(target_noise_stft) ** 2, dim=-1)
                snr = power_speech / (power_noise + self.EPS)
                # Compute weights: softmax(-log10(SNR))
                log_snr = torch.log10(snr + self.EPS)
                weights = torch.softmax(-log_snr, dim=1).unsqueeze(-1).expand_as(proj_stft)  
                weights = weights.to(proj_stft.device)

            elif self.weights_type == 'sigmoid':
                # Compute per-band SNR
                power_speech = torch.mean(torch.abs(target_speech_stft) ** 2, dim=-1)
                power_noise = torch.mean(torch.abs(target_noise_stft) ** 2, dim=-1)
                snr = power_speech / (power_noise + self.EPS)
                # Compute weights: sigmoid(-SNR)
                weights = torch.sigmoid(-snr).unsqueeze(-1).expand_as(proj_stft)
                weights = weights.to(proj_stft.device)
            
            elif self.weights_type == 'sigmoid_log':
                # Compute per-band SNR
                power_speech = torch.mean(torch.abs(target_speech_stft) ** 2, dim=-1)
                power_noise = torch.mean(torch.abs(target_noise_stft) ** 2, dim=-1)
                snr = power_speech / (power_noise + self.EPS)
                # Compute weights: sigmoid(-log10(SNR))
                log_snr = torch.log10(snr + self.EPS)
                weights = torch.sigmoid(-log_snr).unsqueeze(-1).expand_as(proj_stft)
                weights = weights.to(proj_stft.device)

            elif self.weights_type == 'relu':
                # Compute per-band SNR
                power_speech = torch.mean(torch.abs(target_speech_stft) ** 2, dim=-1)
                power_noise = torch.mean(torch.abs(target_noise_stft) ** 2, dim=-1)
                snr = power_speech / (power_noise + self.EPS)
                # Compute weights: ReLU(SNR)
                weights = F.relu(snr).unsqueeze(-1).expand_as(proj_stft)
                weights = weights.to(proj_stft.device)

            elif self.weights_type == 'relu_log':
                # Compute per-band SNR
                power_speech = torch.mean(torch.abs(target_speech_stft) ** 2, dim=-1)
                power_noise = torch.mean(torch.abs(target_noise_stft) ** 2, dim=-1)
                snr = power_speech / (power_noise + self.EPS)
                # Compute weights: ReLU(log10(SNR))
                log_snr = torch.log10(snr + self.EPS)
                weights = F.relu(log_snr).unsqueeze(-1).expand_as(proj_stft)
                weights = weights.to(proj_stft.device)

            elif self.weights_type == 'wo':
                weights = 1.0  # no weighting applied
            
            elif self.weights_type == 'None': 
                weights = 1.0  # no weighting applied

            else:
                print('\033[91m[ERROR]\033[0m The specified weight is not valid!')
                sys.exit()

        # Compute weighted SDR
        weighted_proj = torch.sum(weights * torch.abs(proj_stft) ** 2, dim=(-1, -2))
        weighted_dist = torch.sum(weights * torch.abs(e_dist_stft) ** 2, dim=(-1, -2))
        sdr = weighted_proj / (weighted_dist + self.EPS)

        if self.take_log:
            sdr = 10 * torch.log10(sdr + self.EPS)
            sdr = torch.clamp(sdr, min=-10.0, max=35.0)

        return -sdr.mean()
    
