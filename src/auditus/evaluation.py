import torch
from pystoi import stoi
import numpy as np
import torch.nn.functional as F

class Auditus:
    def __init__(self, zero_mean=True, take_log=True, W=None, gamma=0.2, sr=16000, eps=1e-8):
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.sr = sr
        self.eps = eps
        self.gamma = gamma
        self.W = W
        
        # Define center frequencies and bandwidths of the critical bands
        self.cent_freq = torch.tensor([
            50.0000, 120.000, 190.000, 260.000, 330.000, 400.000, 470.000, 540.000, 617.372, 703.378, 798.717, 904.128, 
            1020.38, 1148.30, 1288.72, 1442.54, 1610.70, 1794.16, 1993.93, 2211.08, 2446.71, 2701.97, 2978.04, 3276.17, 3597.63
        ]) # critical bands 
        self.bandwidth = torch.tensor([
            70.0000, 70.0000, 70.0000, 70.0000, 70.0000, 70.0000, 70.0000, 77.3724, 86.0056, 95.3398, 105.411, 116.256, 
            127.914, 140.423, 153.823, 168.154, 183.457, 199.776, 217.153, 235.631, 255.255, 276.072, 298.126, 321.465, 346.136
        ]) 

    def signal_to_critic_band_stft(self, signal_batch, n_fft=512, winlength=512, hop_length=128, frameLen=0.03, overlap=0.75, eps=1e-8):
        """
        Compute critical band spectrogram from a batch of time-domain signals using STFT.
        
        Args:
            signal_batch (Tensor): Shape [batch_size, n_samples]
            ...
        
        Returns:
            Tensor: [batch_size, n_crit_bands, n_frames]
        """
        sr=self.sr
        cent_freq = self.cent_freq
        bandwidth = self.bandwidth

        skiprate = hop_length
        max_freq = sr / 2
        num_crit = len(cent_freq)
        n_fftby2 = n_fft // 2 # Number of frequency bins in STFT
        bw_min = bandwidth[0]
        min_factor = torch.exp(torch.tensor(-30.0 / (2.0 * 2.303), dtype=torch.float64))

        crit_filter = torch.zeros((num_crit, n_fftby2), dtype=torch.float64, device=signal_batch.device)
        j = torch.arange(0, n_fftby2, dtype=torch.float64)

        for i in range(num_crit):
            f0 = (cent_freq[i] / max_freq) * (n_fftby2)
            bw = (bandwidth[i] / max_freq) * (n_fftby2)
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

    def signal_to_critic_band_stft_old(self, signal_batch, n_fft=512, winlength=512, hop_length=128, frameLen=0.03, overlap=0.75, eps=1e-8):
        """
        Compute critical band spectrogram from a batch of time-domain signals using STFT,
        with improved and normalized critical band filters.
        
        Args:
            signal_batch (Tensor): Shape [batch_size, n_samples]
            
        Returns:
            Tensor: [batch_size, n_crit_bands, n_frames]
        """
        sr = self.sr
        cent_freq = self.cent_freq
        bandwidth = self.bandwidth

        skiprate = hop_length
        max_freq = sr / 2
        num_crit = len(cent_freq)
        n_bins = n_fft // 2  # Number of frequency bins up to Nyquist

        # Create frequency axis in Hz
        freqs = torch.linspace(0, max_freq, steps=n_bins, dtype=torch.float64, device=signal_batch.device)

        # Minimum bandwidth for normalization
        bw_min = bandwidth[0]
        min_factor = torch.exp(torch.tensor(-30.0 / (2.0 * 2.303), dtype=torch.float64))

        # Build critical band filter bank
        crit_filter = torch.zeros((num_crit, n_bins), dtype=torch.float64, device=signal_batch.device)

        for i in range(num_crit):
            f0 = cent_freq[i]  # Center frequency in Hz
            bw = bandwidth[i]  # Bandwidth in Hz

            # Gaussian filter centered at f0 with bandwidth bw
            norm_factor = torch.log(bw_min) - torch.log(bw)
            crit_filter[i, :] = torch.exp(-11 * (((freqs - f0) / bw) ** 2) + norm_factor)

            # Apply a threshold to avoid very small values
            crit_filter[i, :] = crit_filter[i, :] * (crit_filter[i, :] > min_factor)

            # Normalize each filter so its area sums to 1 (optional but improves perceptual consistency)
            if crit_filter[i, :].sum() > 0:
                crit_filter[i, :] /= crit_filter[i, :].sum()

        # Create Hann window
        hannWin = torch.hann_window(winlength, dtype=torch.float64, device=signal_batch.device)

        all_specs = []
        for signal in signal_batch:  # [n_samples]
            
            # Zero-padding if signal is shorter than window length
            if signal.size(0) < winlength:
                pad = winlength - signal.size(0)
                signal = F.pad(signal, (0, pad))
            
            # Compute STFT
            spec = torch.stft(
                signal, n_fft=n_fft, hop_length=skiprate, win_length=winlength,
                window=hannWin, return_complex=True
            ).abs()  # [n_freq, n_frames]
            
            spec = spec[:n_bins, :]  # Match number of frequency bins

            # Apply critical band filter bank
            spec_cb = crit_filter.matmul(spec.to(torch.float64))  # [n_crit_bands, n_frames]
            all_specs.append(spec_cb)

        return torch.stack(all_specs, dim=0)  # [batch_size, n_crit_bands, n_frames]

    def decompose(self, speech, noise, est_speech):
        """ Compute the three components """
        
        speech_target = self.compute_speech_target(speech, est_speech)
        e_interf = self.compute_e_interf(noise, est_speech)
        e_artif = self.compute_e_artif(est_speech, speech_target, e_interf)
        return speech_target, e_interf, e_artif

    def compute_speech_target(self, speech, est_speech):
        """ Projects estimated speech onto clean speech """
        dot = torch.sum(est_speech * speech, dim=-1, keepdim=True) # Compute the dot product <s_hat, s>
        speech_energy = (torch.sum(speech**2, dim=-1, keepdim=True) + self.eps).detach() # Compute the energy of the clean speech <s, s>
        speech_target = (dot / speech_energy) * speech # Compute the projection (<s_hat, s> / <s, s>) * s
        return speech_target

    def compute_e_interf(self, noise, est_speech):
        """ Projects estimated speech onto noise"""
        # Compute the dot product <s_hat, n>
        dot = torch.sum(est_speech * noise, dim=-1, keepdim=True)

        # Compute the energy of the noise <n, n>
        noise_energy = (torch.sum(noise**2, dim=-1, keepdim=True) + self.eps).detach()
        
        # Compute the projection (<s_hat, n> / <n, n>) * n
        e_interf = (dot / noise_energy) * noise
        
        return e_interf

    def compute_e_artif(self, est_speech, speech_target, e_interf):
        """ Computes the artifact error as the residual component """
        # Compute the artifact error
        e_artif = est_speech - speech_target - e_interf

        return e_artif

    def compute_si_sdr(self, speech_target, est_speech):
        
        # Compute the residual noise
        e_noise = est_speech - speech_target
        
        # Compute the SI-SDR ratio
        si_sdr = torch.sum(speech_target**2, dim=-1) / (torch.sum(e_noise**2, dim=-1) + self.eps)

        # Apply the log 
        if self.take_log:
            si_sdr = 10 * torch.log10(si_sdr + self.eps)
        
        si_sdr = torch.clamp(si_sdr, min=-10.0, max=35.0)
        
        return si_sdr.mean().item()

    def compute_si_sir(self, speech_target, e_interf):
        # Compute the SI-SIR ratio
        si_sir = torch.sum(speech_target**2, dim=-1) / (torch.sum(e_interf**2, dim=-1) + self.eps)

        # Apply the log 
        if self.take_log:
            si_sir = 10 * torch.log10(si_sir + self.eps)
        
        si_sir = torch.clamp(si_sir, min=-10.0, max=35.0)

        return si_sir.mean().item()

    def compute_si_sar(self, speech_target, e_interf, e_artif):
        
        # Compute SI-SAR ratio
        si_sar = torch.sum((speech_target + e_interf)**2, dim=-1) / (torch.sum(e_artif**2, dim=-1) + self.eps)

        # Apply the log
        if self.take_log:
            si_sar = 10 * torch.log10(si_sar + self.eps)
        
        si_sar = torch.clamp(si_sar, min=-10.0, max=35.0)

        return si_sar.mean().item()

    def compute_fw_si_sdr(self, speech_target, est_speech):
        
        # Convert to critical band scale
        speech_target_stft = self.signal_to_critic_band_stft(speech_target)
        est_speech_stft = self.signal_to_critic_band_stft(est_speech)
        e_dist_stft = est_speech_stft - speech_target_stft

        # Broadcast weights 
        device = speech_target_stft.device
        weights = self.W.to(device)

        # Weighting average
        clean_energy = torch.abs(speech_target_stft) ** 2
        error_energy = torch.abs(e_dist_stft) ** 2 + self.eps
        SDRlog = 10 * torch.log10( clean_energy / error_energy + self.eps)
        
        fwSDR = torch.sum(weights*SDRlog, dim=1) / torch.sum(weights, dim=1)
        fwSDR = torch.clamp(fwSDR, min=-10.0, max=35.0)
        
        return fwSDR.mean().item()
        
    def compute_fw_si_sir(self, speech_target, e_interf):
        # Convert to critical band scale
        speech_target_stft = self.signal_to_critic_band_stft(speech_target)
        e_interf_stft = self.signal_to_critic_band_stft(e_interf)

        # Broadcast weights 
        weights = self.W

        clean_energy = torch.abs(speech_target_stft) ** 2
        error_energy = torch.abs(e_interf_stft) ** 2 + self.eps
        SIRlog = 10 * torch.log10( clean_energy / error_energy + self.eps)
        fwSIR = torch.sum(weights*SIRlog, dim=1) / torch.sum(weights, dim=1)
        fwSIR = torch.clamp(fwSIR, min=-10.0, max=35.0)

        return fwSIR.mean().item()
    
    def compute_fw_si_sar(self, speech_target, e_interf, e_artif):

        speech_target_stft = self.signal_to_critic_band_stft(speech_target)
        e_interf_stft = self.signal_to_critic_band_stft(e_interf)
        e_artif_stft = self.signal_to_critic_band_stft(e_artif)

        # Broadcast weights 
        weights = self.W

        clean_energy = torch.abs(speech_target_stft + e_interf_stft) ** 2
        error_energy = torch.abs(e_artif_stft) ** 2 + self.eps
        SARlog = 10 * torch.log10(clean_energy / error_energy + self.eps)
        fwSAR = torch.sum(weights*SARlog, dim=1) / torch.sum(weights, dim=1)
        fwSAR = torch.clamp(fwSAR, min=-10.0, max=35.0)

        return fwSAR.mean().item()

    def compute_stoi(self, speech, est_speech):
        # Initialize STOI score accumulator
        stoi_scores = []

        # Loop through batch
        for i in range(speech.shape[0]):  # Iterate over batch size
            stoi_score = stoi(
                speech[i, :].cpu().numpy(),  # Convert single sample to numpy
                est_speech[i, :].cpu().numpy(), 
                self.sr
            )
            stoi_scores.append(stoi_score)
        
        # Compute the average STOI over the batch
        avg_stoi = np.mean(stoi_scores)

        return avg_stoi
    
    def se_eval(self, speech, noise, est_speech, metrics=None):
        """
        Evaluate speech enhancement performance using scale-invariant metrics.

        Args:
            speech (torch.Tensor): Ground truth clean speech [1, n_samples].
            noise (torch.Tensor): Ground truth noise [1, n_samples].
            est_speech (torch.Tensor): Estimated speech signal [1, n_samples].
            metrics (list, optional): List of metric names to compute. Default: ['si-sdr', 'si-sir', 'si-sar'].

        Returns:
            dict: A dictionary containing the computed metric values, rounded to two decimal places.
        """
        # Default metric selection
        if metrics == None:
            metrics = ['si-sdr', 'si-sir', 'si-sar', 'fw-si-sdr', 'fw-si-sir', 'fw-si-sar', 'stoi']

        # Define metrics
        results = {}

        # Detach tensors and disable gradient tracking for efficiency
        with torch.no_grad():
            speech = speech.detach()
            noise = noise.detach()
            est_speech = est_speech.detach()

            # Zero-mean normalization
            if self.zero_mean:
                speech -= speech.mean(dim=-1, keepdim=True)
                noise -= noise.mean(dim=-1, keepdim=True)
                est_speech -= est_speech.mean(dim=-1, keepdim=True)

            # Signal decomposition into target speech, interference, and artifact components
            speech_target, e_interf, e_artif = self.decompose(speech, noise, est_speech)

            # Compute weights
            self.W = self.signal_to_critic_band_stft(speech) ** self.gamma

            # Compute selected metrics
            if 'si-sdr' in metrics:
                results['si-sdr'] = round(self.compute_si_sdr(speech_target, est_speech), 4)
            
            if 'si-sir' in metrics:
                results['si-sir'] = round(self.compute_si_sir(speech_target, e_interf), 4)

            if 'si-sar' in metrics:
                results['si-sar'] = round(self.compute_si_sar(speech_target, e_interf, e_artif), 4)

            if 'fw-si-sdr' in metrics:
                results['fw-si-sdr'] = round(self.compute_fw_si_sdr(speech_target, est_speech), 4)
                
            if 'fw-si-sir' in metrics:
                results['fw-si-sir'] = round(self.compute_fw_si_sir(speech_target, e_interf), 4)
                
            if 'fw-si-sar' in metrics:
                results['fw-si-sar'] = round(self.compute_fw_si_sar(speech_target, e_interf, e_artif), 4)
                
            if 'stoi' in metrics:
                results['stoi'] = round(self.compute_stoi(speech, est_speech), 4)

        return results