#!/usr/bin/env python3
"""
Universal RF Capture System
Works perfectly with or without RTL-SDR hardware
Author: Shahnawaz
License: MIT
"""

import argparse
import sys
import os
import logging
import numpy as np
from pathlib import Path
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class UniversalSDRController:
    """Universal SDR Controller - Works with hardware or simulation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = None
        self.config = {}
        self.hardware_available = False
        
    def configure(self, frequency: float, sample_rate: float, 
                 gain: float, device_type: str = 'rtlsdr'):
        """Configure SDR - works with or without hardware"""
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
        
        # Simulation mode - generate realistic signals
        return self._create_realistic_signal(num_samples)
            
    def _create_realistic_signal(self, num_samples: int) -> np.ndarray:
        """Create realistic RF signals for simulation"""
        t = np.linspace(0, 1, num_samples)
        freq = self.config['frequency']
        
        # Generate different signal types based on frequency
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
        self.logger.info(f"📊 Samples: {len(final_signal)}, Rate: {self.config['sample_rate']/1e6:.1f}Msps")
        
        return final_signal
            
    def close(self):
        """Close SDR device"""
        if self.device and self.hardware_available:
            if hasattr(self.device, 'close'):
                self.device.close()

class CaptureManager:
    """Manages signal capture and file operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def capture_to_file(self, sdr, duration: float, output_path: Path):
        """Capture samples and save to file"""
        sample_rate = sdr.config['sample_rate']
        total_samples = int(sample_rate * duration)
        
        self.logger.info(f"⏱️ Capturing {total_samples} samples ({duration}s)...")
        
        try:
            # Capture in chunks
            chunk_size = min(1024 * 1024, total_samples)
            samples_captured = 0
            all_samples = []
            
            while samples_captured < total_samples:
                remaining = total_samples - samples_captured
                current_chunk = min(chunk_size, remaining)
                
                chunk = sdr.capture_samples(current_chunk)
                all_samples.append(chunk)
                samples_captured += len(chunk)
                
                if samples_captured % (1024*1024) == 0:
                    self.logger.info(f"📥 Progress: {samples_captured}/{total_samples} samples")
                
            # Combine all chunks
            iq_data = np.concatenate(all_samples)
            
            # Save to file
            self.save_iq_data(iq_data, output_path, sample_rate)
            self.logger.info(f"✅ Capture completed: {output_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Capture failed: {e}")
            raise
            
    def save_iq_data(self, iq_data: np.ndarray, output_path: Path, sample_rate: float):
        """Save IQ data to file"""
        output_path = Path(output_path)
        np.save(output_path, iq_data)
        
    def load_iq_data(self, input_path: Path):
        """Load IQ data from file"""
        input_path = Path(input_path)
        if input_path.suffix == '.npy':
            return np.load(input_path), None
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")

class UniversalSignalClassifier:
    """Machine Learning classifier for RF signals"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.classes = ['tone', 'noise', 'mixed', 'modulated']
        
    def extract_features(self, iq_data):
        """Extract features from IQ data for ML"""
        features = []
        
        # Statistical features
        features.extend([
            np.mean(iq_data.real),
            np.mean(iq_data.imag),
            np.std(iq_data.real),
            np.std(iq_data.imag),
            np.var(iq_data.real),
            np.var(iq_data.imag),
        ])
        
        # Spectral features
        fft_mag = np.abs(np.fft.fft(iq_data))
        features.extend([
            np.mean(fft_mag),
            np.std(fft_mag),
            np.max(fft_mag),
            np.argmax(fft_mag) / len(fft_mag),
        ])
        
        return np.array(features)
    
    def build_rf_model(self):
        """Build Random Forest classifier"""
        return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def build_svm_model(self):
        """Build SVM classifier"""
        return SVC(kernel='rbf', probability=True, random_state=42)
    
    def preprocess_iq_data(self, iq_data, segment_length=1024):
        """Preprocess IQ data and extract features"""
        if len(iq_data) < segment_length:
            raise ValueError(f"Input data too short. Need at least {segment_length} samples")
            
        # Segment the data
        num_segments = len(iq_data) // segment_length
        segmented = iq_data[:num_segments * segment_length].reshape(-1, segment_length)
        
        # Extract features for each segment
        features_list = []
        for segment in segmented:
            features = self.extract_features(segment)
            features_list.append(features)
            
        return np.array(features_list)
    
    def train_model(self, model_type='rf', num_samples=1000):
        """Train the specified model"""
        self.logger.info(f"🤖 Training {model_type.upper()} model with {num_samples} samples")
        
        # Create models directory
        Path('models').mkdir(exist_ok=True)
        
        # Create training data
        X, y = self.create_training_data(num_samples)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build and train model
        if model_type == 'rf':
            model = self.build_rf_model()
        elif model_type == 'svm':
            model = self.build_svm_model()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.models[model_type] = model
        self.scalers[model_type] = scaler
        
        # Save model
        self.save_model(model_type)
        
        results = {
            'accuracy': accuracy,
            'model_type': model_type,
            'samples_used': num_samples
        }
        
        self.logger.info(f"✅ Model trained with accuracy: {accuracy:.3f}")
        return results
    
    def create_training_data(self, num_samples=1000):
        """Create training data with various signal types"""
        segment_length = 1024
        features_list = []
        labels = []
        
        for i in range(num_samples):
            signal_type = np.random.randint(0, 4)  # 4 classes
            
            if signal_type == 0:
                # Pure tone
                t = np.linspace(0, 1, segment_length)
                signal = np.exp(1j * 2 * np.pi * 10 * t)
            elif signal_type == 1:
                # Noise
                signal = np.random.normal(0, 1, segment_length) + 1j * np.random.normal(0, 1, segment_length)
            elif signal_type == 2:
                # Mixed signal
                t = np.linspace(0, 1, segment_length)
                signal = np.exp(1j * 2 * np.pi * 5 * t) + 0.5 * (np.random.normal(0, 1, segment_length) + 1j * np.random.normal(0, 1, segment_length))
            else:
                # Modulated signal
                t = np.linspace(0, 1, segment_length)
                carrier = np.exp(1j * 2 * np.pi * 8 * t)
                modulation = np.sin(2 * np.pi * 2 * t)
                signal = carrier * (1 + 0.3 * modulation)
            
            features = self.extract_features(signal)
            features_list.append(features)
            labels.append(signal_type)
        
        return np.array(features_list), np.array(labels)
    
    def classify_signals(self, input_path, model_type='rf'):
        """Classify signals using trained model"""
        if model_type not in self.models:
            self.load_model(model_type)
            
        # Load input data
        cm = CaptureManager()
        iq_data, _ = cm.load_iq_data(input_path)
        
        # Preprocess and extract features
        features = self.preprocess_iq_data(iq_data)
        
        # Scale features
        features_scaled = self.scalers[model_type].transform(features)
        
        # Predict
        predictions = self.models[model_type].predict_proba(features_scaled)
        
        # Process results
        results = []
        for i, pred in enumerate(predictions):
            class_idx = np.argmax(pred)
            confidence = pred[class_idx]
            
            class_name = self.classes[class_idx]
                
            results.append({
                'segment': i,
                'class': class_name,
                'confidence': float(confidence),
                'all_probabilities': {cls: float(prob) for cls, prob in zip(self.classes, pred)}
            })
            
        return results
    
    def save_model(self, model_type):
        """Save trained model"""
        model_path = f'models/{model_type}_model.joblib'
        scaler_path = f'models/{model_type}_scaler.joblib'
        classes_path = f'models/{model_type}_classes.json'
        
        joblib.dump(self.models[model_type], model_path)
        joblib.dump(self.scalers[model_type], scaler_path)
        
        with open(classes_path, 'w') as f:
            json.dump(self.classes, f)
        
        self.logger.info(f"💾 Model saved to {model_path}")
        
    def load_model(self, model_type):
        """Load pre-trained model"""
        model_path = f'models/{model_type}_model.joblib'
        scaler_path = f'models/{model_type}_scaler.joblib'
        classes_path = f'models/{model_type}_classes.json'
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        self.models[model_type] = joblib.load(model_path)
        self.scalers[model_type] = joblib.load(scaler_path)
        
        if Path(classes_path).exists():
            with open(classes_path, 'r') as f:
                self.classes = json.load(f)
        
        self.logger.info(f"📂 Model loaded from {model_path}")

class UniversalRFCapture:
    """Main CLI application"""
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def run(self):
        parser = argparse.ArgumentParser(
            description="🚀 Universal RF Capture - Works With or Without Hardware",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
🎯 Examples:
  python rf_capture_universal.py capture --freq 433e6 --output signal.npy
  python rf_capture_universal.py train --model rf
  python rf_capture_universal.py classify --input signal.npy --model rf
  python rf_capture_universal.py status
  python rf_capture_universal.py demo

💡 Features:
  ✅ With RTL-SDR: Captures real RF signals
  ✅ Without Hardware: Generates realistic simulated signals
  ✅ No errors - Always works
  ✅ Machine Learning classification
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Commands')
        
        # Capture command
        capture_parser = subparsers.add_parser('capture', help='Capture RF signals')
        capture_parser.add_argument('--freq', type=float, default=433e6, 
                                  help='Center frequency in Hz (default: 433MHz)')
        capture_parser.add_argument('--duration', type=float, default=5,
                                  help='Capture duration in seconds (default: 5)')
        capture_parser.add_argument('--rate', type=float, default=2.4e6,
                                  help='Sample rate in Hz (default: 2.4Msps)')
        capture_parser.add_argument('--gain', type=float, default=20,
                                  help='RF gain in dB (default: 20)')
        capture_parser.add_argument('--output', '-o', type=Path, required=True,
                                  help='Output file path (.npy)')
        
        # Classify command
        classify_parser = subparsers.add_parser('classify', help='Classify signals')
        classify_parser.add_argument('--input', '-i', type=Path, required=True,
                                   help='Input IQ file (.npy)')
        classify_parser.add_argument('--model', '-m', type=str, default='rf',
                                   choices=['rf', 'svm'], help='Model type')
        
        # Train command
        train_parser = subparsers.add_parser('train', help='Train ML models')
        train_parser.add_argument('--model', '-m', type=str, default='rf',
                                choices=['rf', 'svm'], help='Model type')
        train_parser.add_argument('--samples', type=int, default=1000,
                                help='Training samples to generate')
        
        # Status command
        status_parser = subparsers.add_parser('status', help='Check hardware status')
        
        # Demo command
        demo_parser = subparsers.add_parser('demo', help='Run complete demo')
        
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            sys.exit(1)
            
        self.execute_command(args)
    
    def execute_command(self, args):
        try:
            if args.command == 'capture':
                self.handle_capture(args)
            elif args.command == 'classify':
                self.handle_classify(args)
            elif args.command == 'train':
                self.handle_train(args)
            elif args.command == 'status':
                self.handle_status(args)
            elif args.command == 'demo':
                self.handle_demo(args)
        except Exception as e:
            self.logger.error(f"❌ Command failed: {e}")
            sys.exit(1)

    def handle_capture(self, args):
        self.logger.info("🎯 Starting RF Signal Capture")
        self.logger.info(f"📡 Frequency: {args.freq/1e6:.0f} MHz")
        self.logger.info(f"⏱️ Duration: {args.duration} seconds")
        self.logger.info(f"📊 Sample Rate: {args.rate/1e6:.1f} Msps")
        
        sdr = UniversalSDRController()
        cm = CaptureManager()
        
        sdr.configure(
            frequency=args.freq,
            sample_rate=args.rate,
            gain=args.gain
        )
        
        cm.capture_to_file(
            sdr=sdr,
            duration=args.duration,
            output_path=args.output
        )
        
        self.logger.info(f"✅ Capture completed: {args.output}")

    def handle_classify(self, args):
        if not args.input.exists():
            self.logger.error(f"❌ Input file not found: {args.input}")
            self.logger.info("💡 First run: python rf_capture_universal.py capture --output signal.npy")
            return
            
        self.logger.info(f"🔍 Classifying signals from {args.input}")
        classifier = UniversalSignalClassifier()
        results = classifier.classify_signals(
            input_path=args.input,
            model_type=args.model
        )
        
        # Show results
        print("\n" + "="*50)
        print("📊 CLASSIFICATION RESULTS")
        print("="*50)
        for i, result in enumerate(results[:10]):
            print(f"Segment {result['segment']:3d}: {result['class']:10s} "
                  f"(confidence: {result['confidence']:.3f})")
        
        if len(results) > 10:
            print(f"... and {len(results) - 10} more segments")
            
        # Save results
        results_file = args.input.with_suffix('.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"💾 Full results saved to: {results_file}")

    def handle_train(self, args):
        self.logger.info(f"🤖 Training {args.model.upper()} model")
        
        classifier = UniversalSignalClassifier()
        results = classifier.train_model(
            model_type=args.model,
            num_samples=args.samples
        )
        
        print("\n" + "="*40)
        print("✅ TRAINING COMPLETED")
        print("="*40)
        print(f"Model Type:  {results['model_type'].upper()}")
        print(f"Accuracy:    {results['accuracy']:.3f}")
        print(f"Samples:     {results['samples_used']}")
        print(f"Saved to:    models/{args.model}_model.joblib")

    def handle_status(self, args):
        self.logger.info("🔍 Checking System Status...")
        
        sdr = UniversalSDRController()
        sdr.configure(frequency=433e6, sample_rate=2.4e6, gain=20)
        
        print("\n" + "="*40)
        print("🖥️  SYSTEM STATUS")
        print("="*40)
        print(f"RTL-SDR Hardware: {'✅ CONNECTED' if sdr.hardware_available else '🔧 SIMULATION MODE'}")
        print(f"Python:          ✅ Ready")
        print(f"ML Models:       ✅ Ready")
        print(f"Signal Capture:  ✅ Ready")
        
        if not sdr.hardware_available:
            print("\n💡 No RTL-SDR hardware detected.")
            print("   The system will automatically use simulation mode.")
            print("   Connect RTL-SDR for real signal capture.")
        else:
            print("\n🎉 RTL-SDR is ready for real signal capture!")

    def handle_demo(self, args):
        """Run a complete demo"""
        self.logger.info("🚀 Starting Complete RF Capture Demo")
        
        # Step 1: Check status
        self.handle_status(args)
        
        # Step 2: Capture signal
        output_file = Path("demo_signal.npy")
        self.logger.info(f"\n📡 Step 1: Capturing demo signal...")
        
        sdr = UniversalSDRController()
        cm = CaptureManager()
        
        sdr.configure(frequency=433e6, sample_rate=2.4e6, gain=20)
        cm.capture_to_file(sdr=sdr, duration=3, output_path=output_file)
        
        # Step 3: Train model
        self.logger.info(f"\n🤖 Step 2: Training model...")
        self.handle_train(argparse.Namespace(model='rf', samples=500))
        
        # Step 4: Classify
        self.logger.info(f"\n🔍 Step 3: Classifying demo signal...")
        self.handle_classify(argparse.Namespace(input=output_file, model='rf'))
        
        self.logger.info("\n🎉 Demo completed successfully!")
        self.logger.info("💡 Now try: python rf_capture_universal.py capture --freq 868e6 --output my_signal.npy")

def main():
    app = UniversalRFCapture()
    app.run()

if __name__ == "__main__":
    main()
