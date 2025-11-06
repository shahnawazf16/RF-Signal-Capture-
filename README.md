# RF-Signal-Capture-
RF-Signal-Capture
# Universal RF Capture - Advanced RF Signal Analysis with Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-brightgreen)
![SDR](https://img.shields.io/badge/SDR-Compatible-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

A universal command-line interface (CLI) software for RF signal capture, analysis, and classification using Software Defined Radio (SDR) hardware and advanced Machine Learning algorithms. **Works perfectly with or without hardware!**

## 🚀 Key Features

### 📡 Universal Hardware Support
- **✅ With RTL-SDR**: Captures real RF signals from hardware
- **✅ Without Hardware**: Generates realistic simulated signals
- **🔧 Zero Errors**: No crashes, no LIBUSB errors - always works
- **🔄 Auto-Detection**: Automatically switches between hardware/simulation modes

### 🤖 Machine Learning Powered
- **Random Forest** & **SVM** classifiers for signal classification
- **Real-time signal analysis** and classification
- **Feature extraction** (spectral, statistical features)
- **Model training** and evaluation with accuracy metrics

### 🎯 Multiple Frequency Support
- **433 MHz** - IoT devices, remote controls (FSK-like signals)
- **868 MHz** - European IoT, LoRa-like signals
- **2.4 GHz** - WiFi, Bluetooth (OFDM-like signals)
- **Custom frequencies** with appropriate signal simulation

### 💾 Professional Output
- **NumPy (.npy)** - Efficient binary data storage
- **JSON (.json)** - Human-readable classification results
- **Model files** - Save/load trained ML models
- **Progress tracking** - Real-time capture progress

## 🎉 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/shahnawazf16/Universal-RF-Capture.git
cd Universal-RF-Capture

# Install dependencies
pip install numpy scikit-learn joblib pyrtlsdr

Basic Usage
bash

# Check system status
python rf_capture_universal.py status

# Run complete demo
python rf_capture_universal.py demo

# Capture signals (works without hardware!)
python rf_capture_universal.py capture --freq 433e6 --output signal.npy

# Train machine learning model
python rf_capture_universal.py train --model rf

# Classify captured signals
python rf_capture_universal.py classify --input signal.npy --model rf

📋 Complete Command Reference
Capture RF Signals
bash

# Basic capture at 433MHz
python rf_capture_universal.py capture --freq 433e6 --output my_signal.npy

# Custom duration and sample rate
python rf_capture_universal.py capture --freq 868e6 --duration 10 --rate 2.4e6 --output iot_signal.npy

# WiFi frequency capture
python rf_capture_universal.py capture --freq 2.4e9 --output wifi_signal.npy

Machine Learning Operations
bash

# Train Random Forest model
python rf_capture_universal.py train --model rf --samples 1000

# Train SVM model
python rf_capture_universal.py train --model svm --samples 500

# Classify signals
python rf_capture_universal.py classify --input signal.npy --model rf

# Classify with custom model
python rf_capture_universal.py classify --input iot_signal.npy --model svm

System Management
bash

# Check hardware status
python rf_capture_universal.py status

# Run complete demonstration
python rf_capture_universal.py demo

🏗️ Project Structure
text

Universal-RF-Capture/
├── rf_capture_universal.py     # Main application (single file!)
├── models/                     # Trained ML models
│   ├── rf_model.joblib
│   ├── rf_scaler.joblib
│   └── rf_classes.json
├── demo_signal.npy            # Demo capture file
├── demo_signal.json           # Demo classification results
├── README.md                  # This file
└── LICENSE                    # MIT License

🔧 How It Works
🎯 Universal Design

The system automatically detects whether RTL-SDR hardware is available:

With Hardware Connected:
text

✅ RTL-SDR Hardware Connected - Capturing Real Signals
📡 Captured 12,000,000 real samples from hardware

Without Hardware (Simulation Mode):
text

🔧 Simulation Mode Active - Generating Realistic RF Signals
🎯 Generated IoT/Remote Control (FSK-like) at 433MHz
📊 Samples: 12,000,000, Rate: 2.4Msps

🤖 Machine Learning Pipeline

    Signal Capture - Real or simulated IQ data

    Feature Extraction - Spectral and statistical features

    Classification - Random Forest or SVM models

    Results - Signal type identification with confidence scores

📡 Realistic Signal Simulation

    433MHz: FSK-like signals (IoT devices, remote controls)

    868MHz: LoRa-like signals (European IoT)

    2.4GHz: OFDM-like signals (WiFi, Bluetooth)

    Noise modeling with realistic SNR levels

📊 Output Examples
Classification Results
text

==================================================
📊 CLASSIFICATION RESULTS
==================================================
Segment   0: tone       (confidence: 0.650)
Segment   1: tone       (confidence: 0.700)
Segment   2: tone       (confidence: 0.700)
Segment   3: tone       (confidence: 0.680)
Segment   4: tone       (confidence: 0.700)
... and 7,021 more segments
💾 Full results saved to: signal.json

Training Results
text

========================================
✅ TRAINING COMPLETED
========================================
Model Type:  RF
Accuracy:    1.000
Samples:     1000
Saved to:    models/rf_model.joblib

🛠️ Technical Details
Dependencies

    numpy - Scientific computing

    scikit-learn - Machine learning

    joblib - Model serialization

    pyrtlsdr - RTL-SDR support (optional)

Supported Platforms

    ✅ Linux (Ubuntu, Debian, etc.)

    ✅ Windows (with Python 3.8+)

    ✅ macOS

    ✅ Raspberry Pi

Signal Types Classified

    tone - Pure carrier signals

    noise - Random noise patterns

    mixed - Signal + noise combinations

    modulated - Various modulation types

🚀 Advanced Usage
Custom Signal Processing
python

from rf_capture_universal import UniversalSDRController, UniversalSignalClassifier

# Custom capture and analysis
sdr = UniversalSDRController()
sdr.configure(frequency=433e6, sample_rate=2.4e6, gain=20)
samples = sdr.capture_samples(100000)

classifier = UniversalSignalClassifier()
features = classifier.extract_features(samples)

Integration with Other Tools

The system outputs standard NumPy and JSON formats that can be used with:

    MATLAB - .npy file import

    Jupyter Notebooks - Data analysis

    GNU Radio - Signal processing

    Custom applications - JSON API

🐛 Troubleshooting
Common Non-Issues

    "Simulation Mode Active" - This is normal when no hardware is connected

    "No RTL-SDR hardware detected" - Expected behavior, system still works

    Confidence scores 0.5-0.7 - Realistic for simulated data

Performance Tips

    Reduce --duration for faster testing

    Use --samples 500 for quick model training

    Start with python rf_capture_universal.py demo for verification

🤝 Contributing

We welcome contributions! Areas for improvement:

    Additional SDR hardware support

    New machine learning models

    Signal processing algorithms

    Documentation improvements

Development Setup
bash

git clone https://github.com/shahnawazf16/Universal-RF-Capture.git
cd Universal-RF-Capture
python rf_capture_universal.py demo  # Test the system

📞 Support & Contact

Author: Shahnawaz
Email: shahnawazzai@gmail.com
Cell: +92 333 2522802
GitHub: https://github.com/shahnawazf16

Repository: https://github.com/shahnawazf16/Universal-RF-Capture
📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

MIT License Features:

    ✅ Free to use for personal and commercial projects

    ✅ Permission to modify and distribute

    ✅ No warranty - use at your own risk

    ✅ Must include original license and copyright

🙏 Acknowledgments

    RTL-SDR community for hardware inspiration

    Scikit-learn team for machine learning tools

    NumPy and SciPy communities for scientific computing

🎯 Future Roadmap

    Web interface for remote monitoring

    Docker containerization

    Additional SDR hardware support (HackRF, USRP)

    Real-time spectrum analysis

    Deep learning models (CNN, LSTM)

    Mobile application interface

<div align="center">
⭐ If you find this project useful, please give it a star! ⭐

Happy RF Signal Processing! 🎉
</div> ```
And here's the LICENSE file:
text

MIT License

Copyright (c) 2024 Shahnawaz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Files to upload to GitHub:

    rf_capture_universal.py - Your main Python script

    README.md - The documentation above

    LICENSE - The MIT license text

    Any generated files (optional):

        demo_signal.npy

        demo_signal.json

        models/ folder with trained models

Repository Structure:
text

Universal-RF-Capture/
├── rf_capture_universal.py
├── README.md
├── LICENSE
├── demo_signal.npy
├── demo_signal.json
└── models/
    ├── rf_model.joblib
    ├── rf_scaler.joblib
    └── rf_classes.json

Your GitHub repository is now ready with professional documentation! 🚀
