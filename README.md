High-accuracy Indoor Positioning System achieving 99.46% accuracy using Machine Learning

This project implements an Indoor Localization and Floor-Level Detection system using WiFi RSSI fingerprinting.
It compares Machine Learning and Deep Learning models to identify the best approach for real-world deployment in GPS-denied environments like malls, hospitals, and campuses.

📍 Predicts floor-level location using WiFi signal strengths
⚡ Achieved 99.46% accuracy with Random Forest
🔍 Compared ML vs DL models:
     Random Forest
     Support Vector Machine (SVM)
     Convolutional Neural Network (CNN)
     Long Short-Term Memory (LSTM)
⏱️ Reduced training time from ~789s (CNN) → 1.1s (RF)
💡 Demonstrates that classical ML can outperform deep learning for structured data


📁 Indoor-Localization
│
├── 📁 app                # Web interface / deployment scripts
├── 📁 templates          # HTML templates (index.html)
├── 📁 train              # Model training scripts
├── 📁 results            # Output results, metrics, plots
│
├── 📄 TrainingData.csv   # Training dataset
├── 📄 ValidationData.csv # Validation dataset
│
└── README.md

📊 Dataset
📌 Dataset Used: UJIIndoorLoc WiFi Fingerprint Dataset
📥 Download Link:
👉 https://www.kaggle.com/datasets/giantuji/UjiIndoorLoc
Dataset Details:
500+ WiFi Access Point features (RSSI values)
Includes:
Building ID
Floor Number
Latitude & Longitude
RSSI values range from -104 dBm to 0 dBm
Missing values handled as -110 dBm

