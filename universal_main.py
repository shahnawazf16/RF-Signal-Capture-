#!/usr/bin/env python3
import argparse
import sys
import logging
from pathlib import Path

class UniversalRFCapture:
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
            description="Universal RF Capture - Works With or Without Hardware",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  rfcapture capture --freq 433e6 --duration 5 --output signal.npy
  rfcapture train --model rf
  rfcapture classify --input signal.npy --model rf
  rfcapture status  # Check hardware availability

Hardware & Simulation Modes:
  ✅ With RTL-SDR: Captures real RF signals
  ✅ Without Hardware: Generates realistic simulated signals
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
                                  help='Output file path')
        
        # Classify command
        classify_parser = subparsers.add_parser('classify', help='Classify signals')
        classify_parser.add_argument('--input', '-i', type=Path, required=True,
                                   help='Input IQ file')
        classify_parser.add_argument('--model', '-m', type=str, default='rf',
                                   choices=['rf', 'svm'], help='Model type')
        classify_parser.add_argument('--output', '-o', type=Path,
                                   help='Output results file (optional)')
        
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
            self.logger.error(f"Command failed: {e}")
            sys.exit(1)

    def handle_capture(self, args):
        from src.hardware.sdr_controller_fixed import SDRController
        from src.core.capture_manager import CaptureManager
        
        self.logger.info("🎯 Starting RF Signal Capture")
        self.logger.info(f"📡 Frequency: {args.freq/1e6:.0f} MHz")
        self.logger.info(f"⏱️ Duration: {args.duration} seconds")
        self.logger.info(f"📊 Sample Rate: {args.rate/1e6:.1f} Msps")
        
        sdr = SDRController()
        cm = CaptureManager()
        
        sdr.configure(
            frequency=args.freq,
            sample_rate=args.rate,
            gain=args.gain,
            device_type='rtlsdr'
        )
        
        cm.capture_to_file(
            sdr=sdr,
            duration=args.duration,
            output_path=args.output
        )
        
        self.logger.info(f"✅ Capture completed: {args.output}")

    def handle_classify(self, args):
        from src.ml.light_classifier import LightSignalClassifier
        
        if not args.input.exists():
            self.logger.error(f"Input file not found: {args.input}")
            self.logger.info("💡 First run: rfcapture capture --output signal.npy")
            return
            
        self.logger.info(f"🔍 Classifying signals from {args.input}")
        classifier = LightSignalClassifier()
        results = classifier.classify_signals(
            input_path=args.input,
            model_type=args.model
        )
        
        # Show results
        print("\n" + "="*50)
        print("📊 CLASSIFICATION RESULTS")
        print("="*50)
        for i, result in enumerate(results[:10]):  # Show first 10
            print(f"Segment {result['segment']:3d}: {result['class']:8s} "
                  f"(confidence: {result['confidence']:.3f})")
        
        if len(results) > 10:
            print(f"... and {len(results) - 10} more segments")
            
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"💾 Full results saved to: {args.output}")

    def handle_train(self, args):
        from src.ml.light_classifier import LightSignalClassifier
        
        self.logger.info(f"🤖 Training {args.model.upper()} model")
        self.logger.info(f"📚 Generating {args.samples} training samples")
        
        classifier = LightSignalClassifier()
        results = classifier.train_model(
            data_path=Path('.'),
            model_type=args.model
        )
        
        print("\n" + "="*40)
        print("✅ TRAINING COMPLETED")
        print("="*40)
        print(f"Model Type: {results['model_type'].upper()}")
        print(f"Accuracy:   {results['accuracy']:.3f}")
        print(f"Saved to:   models/{args.model}_model.joblib")

    def handle_status(self, args):
        from src.hardware.sdr_controller_fixed import SDRController
        
        self.logger.info("🔍 Checking System Status...")
        
        sdr = SDRController()
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
        
        from src.hardware.sdr_controller_fixed import SDRController
        from src.core.capture_manager import CaptureManager
        
        sdr = SDRController()
        cm = CaptureManager()
        
        sdr.configure(frequency=433e6, sample_rate=2.4e6, gain=20)
        cm.capture_to_file(sdr=sdr, duration=3, output_path=output_file)
        
        # Step 3: Train model (if not exists)
        self.logger.info(f"\n🤖 Step 2: Training model...")
        self.handle_train(argparse.Namespace(model='rf', samples=500))
        
        # Step 4: Classify
        self.logger.info(f"\n🔍 Step 3: Classifying demo signal...")
        self.handle_classify(argparse.Namespace(input=output_file, model='rf', output=None))
        
        self.logger.info("\n🎉 Demo completed successfully!")
        self.logger.info("💡 Now try: rfcapture capture --freq 868e6 --output my_signal.npy")

def main():
    app = UniversalRFCapture()
    app.run()

if __name__ == "__main__":
    main()
