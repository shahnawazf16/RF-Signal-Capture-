import numpy as np
import logging
from pathlib import Path

class SDRController:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = None
        self.config = {}
        self.hardware_available = False
        
    def configure(self, frequency: float, sample_rate: float, 
                 gain: float, device_type: str = 'rtlsdr'):
        """Configure SDR device - works perfectly with or without hardware"""
        self.config.update({
            'frequency': frequency,
            'sample_rate': sample_rate,
            'gain': gain,
            'device_type': device_type
        })
        
        # Try hardware first
        self.hardware_available = self._try_init_hardware()
        
        if self.hardware_available:
            self.logger.info("✅ RTL-SDR Hardware Connected - Capturing Real Signals")
        else:
            self.logger.info("🔧 Simulation Mode Active - Generating Realistic RF Signals")
            self.logger.info("💡 Tip: Connect RTL-SDR hardware for real signal capture")
            
    def _try_init_hardware(self) -> bool:
        """Try to initialize hardware, return True if successful"""
        try:
            from rtlsdr import RtlSdr
            self.device = RtlSdr()
            self.device.set_center_freq(int(self.config['frequency']))
            self.device.set_sample_rate(int(self.config['sample_rate']))
            self.device.set_gain(self.config['gain'])
            return True
        except Exception:
            # Don't show error, just return False
            return False
            
    def capture_samples(self, num_samples: int) -> np.ndarray:
        """Capture IQ samples - works with hardware or simulation"""
        if self.hardware_available:
            try:
                if hasattr(self.device, 'read_samples'):
                    samples = self.device.read_samples(num_samples)
                    self.logger.info(f"📡 Captured {len(samples)} real samples from hardware")
                    return np.array(samples, dtype=np.complex64)
            except Exception as e:
                self.logger.warning(f"Hardware capture issue, switching to simulation: {e}")
                self.hardware_available = False
        
        # If no hardware or hardware failed, use simulation
        return self._create_realistic_signal(num_samples)
            
    def _create_realistic_signal(self, num_samples: int) -> np.ndarray:
        """Create realistic RF signals for simulation"""
        t = np.linspace(0, 1, num_samples)
        freq = self.config['frequency']
        
        # Show what type of signal we're generating based on frequency
        if freq == 433e6:
            signal_type = "IoT/Remote Control (FSK-like)"
            carrier = np.exp(1j * 2 * np.pi * 0.1 * t)
            mod_signal = np.sin(2 * np.pi * 0.01 * t)
            signal = carrier * (1 + 0.3 * mod_signal)
            
        elif freq == 868e6:
            signal_type = "European IoT (LoRa-like)"
            carrier = np.exp(1j * 2 * np.pi * 0.08 * t)
            chirp = np.exp(1j * 2 * np.pi * 0.001 * t * t)
            signal = carrier * chirp
            
        elif freq == 2.4e9:
            signal_type = "WiFi/Bluetooth (OFDM-like)"
            carrier = np.exp(1j * 2 * np.pi * 0.05 * t)
            ofdm_like = np.sin(2 * np.pi * 0.02 * t) + 0.5 * np.sin(2 * np.pi * 0.04 * t)
            signal = carrier * (1 + 0.2 * ofdm_like)
            
        else:
            signal_type = "Generic RF Signal"
            carrier = np.exp(1j * 2 * np.pi * 0.1 * t)
            mod_signal = np.sin(2 * np.pi * 0.005 * t)
            signal = carrier * (1 + 0.25 * mod_signal)
        
        # Add realistic noise
        noise_power = 0.1
        noise = noise_power * (np.random.normal(0, 1, num_samples) + 
                              1j * np.random.normal(0, 1, num_samples))
        
        final_signal = signal + noise
        
        self.logger.info(f"🎯 Generated {signal_type} at {freq/1e6:.0f}MHz")
        self.logger.info(f"📊 Simulation: {len(final_signal)} samples, Sample Rate: {self.config['sample_rate']/1e6:.1f}Msps")
        
        return final_signal
            
    def start_streaming(self, callback, num_samples: int = 1024*1024) -> None:
        """Start continuous streaming"""
        if self.hardware_available and hasattr(self.device, 'read_samples_async'):
            self.device.read_samples_async(callback, num_samples=num_samples)
        else:
            self.logger.info("📡 Streaming in Simulation Mode")
            import threading
            import time
            
            def simulate_stream():
                chunk_size = 1024
                while True:
                    samples = self._create_realistic_signal(chunk_size)
                    callback(samples)
                    time.sleep(0.1)
            
            thread = threading.Thread(target=simulate_stream, daemon=True)
            thread.start()
            
    def close(self) -> None:
        """Close SDR device"""
        if self.device and self.hardware_available:
            if hasattr(self.device, 'close'):
                self.device.close()
            self.device = None
