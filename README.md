# 🎙️ Speech Emotion Recognition System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Librosa-Audio%20Processing-9B59B6?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/RAVDESS-Dataset-E74C3C?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/MFCC-Feature%20Extraction-2ECC71?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/CodeAlpha-Internship-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<p align="center">
  A deep learning and signal processing project that recognizes <strong>human emotions from speech audio</strong> using audio feature extraction techniques (MFCC, Chroma, Mel Spectrogram) and machine learning classification — built as part of the <strong>CodeAlpha Data Science Internship</strong>.
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Audio Feature Extraction](#-audio-feature-extraction)
- [Model Details](#-model-details)
- [Results](#-results)
- [Technologies Used](#-technologies-used)
- [Getting Started](#-getting-started)
- [How to Run](#-how-to-run)
- [Key Concepts Covered](#-key-concepts-covered)
- [Real-World Applications](#-real-world-applications)
- [Author](#-author)

---

## 🧠 Overview

This project builds a **Speech Emotion Recognition (SER)** system that classifies the emotional state of a speaker from raw audio recordings. It combines **audio signal processing** with **machine learning** to extract meaningful acoustic features from speech and train a classifier that distinguishes emotions such as happy, sad, angry, fearful, and more.

Unlike traditional ML projects that work with tabular or image data, this project operates directly on **raw `.wav` audio files** — making it a unique and advanced portfolio project that demonstrates skills in signal processing, feature engineering, and classification.

This project was developed as part of the **CodeAlpha Data Science Internship** program.

---

## 🎯 Problem Statement

Human speech carries rich emotional information beyond the words being spoken — tone, pitch, rhythm, and intensity all encode how a person feels. Automatically recognizing these emotions from audio has broad applications in:

- Making AI assistants more emotionally aware
- Detecting distress in call centre conversations
- Enabling empathetic human-computer interaction
- Supporting mental health monitoring systems

This project builds an end-to-end pipeline that takes a raw speech audio file as input and outputs the predicted emotion of the speaker.

---

## 📊 Dataset

**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song

| Property           | Details                                                        |
|--------------------|----------------------------------------------------------------|
| Full Name          | Ryerson Audio-Visual Database of Emotional Speech and Song     |
| Total Files        | 1,440 audio files (.wav format)                               |
| Actors             | 24 professional actors (12 Male, 12 Female)                   |
| Emotions           | 8 emotional classes                                            |
| Audio Type         | Speech recordings at 48kHz                                     |
| Source             | Kaggle / Zenodo (Open Access)                                  |
| Format             | `.wav` files organized in actor folders                        |

### Emotion Classes

| Emotion Code | Emotion Label   | Description                                |
|--------------|-----------------|--------------------------------------------|
| 01           | **Neutral**     | Calm, normal speech                        |
| 02           | **Calm**        | Relaxed, composed tone                     |
| 03           | **Happy**       | Positive, joyful expression                |
| 04           | **Sad**         | Sorrowful, low-energy speech               |
| 05           | **Angry**       | High-intensity, aggressive tone            |
| 06           | **Fearful**     | Tense, anxious speech                      |
| 07           | **Disgust**     | Repulsed, negative emotional tone          |
| 08           | **Surprised**   | Sudden reaction, unexpected                |

### RAVDESS Filename Convention

Each audio file encodes metadata in its filename:
```
03-01-06-01-02-01-12.wav
│   │   │   │   │   │   └─ Actor ID (01–24)
│   │   │   │   │   └───── Repetition (01 or 02)
│   │   │   │   └───────── Statement (01 or 02)
│   │   │   └───────────── Intensity (01=Normal, 02=Strong)
│   │   └───────────────── Emotion (01–08)
│   └───────────────────── Vocal Channel (01=Speech, 02=Song)
└───────────────────────── Modality (01=AV, 02=Video-only, 03=Audio-only)
```

---

## 📁 Project Structure

```
CodeAlpha_Speech_Recognition_System/
│
├── Speech_Emotion_Recognition.ipynb       ← Main project notebook (2.25 MB)
│   ├── 1.  Import Libraries
│   ├── 2.  Load RAVDESS Audio Files
│   ├── 3.  Parse Emotion Labels from Filenames
│   ├── 4.  Waveform Visualization
│   ├── 5.  Spectrogram Visualization
│   ├── 6.  Audio Feature Extraction
│   │         ├── MFCC (Mel-Frequency Cepstral Coefficients)
│   │         ├── Chroma Features
│   │         └── Mel Spectrogram
│   ├── 7.  Build Feature Matrix (X) & Label Vector (y)
│   ├── 8.  Train-Test Split
│   ├── 9.  Train MLP Classifier / SVM Model
│   ├── 10. Evaluate Model (Accuracy, Classification Report)
│   └── 11. Predict Emotion from New Audio File
│
└── README.md
```

---

## 🔊 Audio Feature Extraction

The core of this project is transforming raw audio waveforms into structured numerical features that a machine learning model can understand. Three key feature types are extracted using **Librosa**:

### 1. MFCC — Mel-Frequency Cepstral Coefficients
```
• Captures the short-term power spectrum of sound
• Mimics how the human ear perceives frequency (logarithmic scale)
• Most widely used feature in speech and emotion recognition
• Extracts 40 coefficients per audio file
```

### 2. Chroma Features
```
• Represents the 12 pitch classes (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
• Captures tonal and harmonic content of speech
• Useful for distinguishing emotional tones
• Extracts 12 chroma values per audio file
```

### 3. Mel Spectrogram
```
• Visual representation of audio frequency over time on a Mel scale
• Captures both temporal and frequency characteristics
• Provides a rich, texture-like summary of audio energy
• Extracts 128 features per audio file
```

### Combined Feature Vector
Each audio file is represented as a **180-dimensional feature vector**:
```
[40 MFCCs] + [12 Chroma features] + [128 Mel Spectrogram features] = 180 features
```

---

## 🤖 Model Details

### Algorithm: MLP Classifier (Multi-Layer Perceptron)

A **Multi-Layer Perceptron** neural network is used for classification, which is well-suited for high-dimensional audio feature vectors due to its ability to learn complex non-linear decision boundaries.

**Training Configuration:**

| Parameter          | Value                              |
|--------------------|------------------------------------|
| Algorithm          | MLPClassifier (sklearn)            |
| Hidden Layer Sizes | (300,)                             |
| Activation         | ReLU                               |
| Solver             | Adam                               |
| Max Iterations     | 500                                |
| Evaluation Metric  | Accuracy + Classification Report   |
| Train-Test Split   | 75% Train / 25% Test               |
| Random State       | 9                                  |

### Preprocessing Pipeline

| Step                          | Details                                                   |
|-------------------------------|-----------------------------------------------------------|
| Audio Loading                 | `librosa.load()` with 22,050 Hz sample rate              |
| Feature Extraction            | MFCC (n=40), Chroma (n=12), Mel Spectrogram (n=128)     |
| Feature Aggregation           | Mean across time axis → fixed-length vector               |
| Label Parsing                 | Extracted from RAVDESS filename (position 3, 0-indexed)  |
| Train-Test Split              | Stratified 75/25 split                                   |
| Standard Scaling              | Applied to normalize feature values                       |

---

## 📈 Results

| Metric              | Score      |
|---------------------|------------|
| Test Accuracy       | ~72–78%    |
| Best Performing Emotions | Happy, Angry, Fearful |
| Hardest to Distinguish   | Calm vs Neutral      |

> Speech Emotion Recognition is an inherently challenging task — even humans disagree on emotional labels ~20% of the time. An accuracy of 72–78% on 8 emotion classes (random baseline = 12.5%) represents strong performance.

---

## 🛠️ Technologies Used

| Technology       | Purpose                                             |
|------------------|-----------------------------------------------------|
| Python 3.8+      | Programming Language                                |
| Librosa          | Audio loading, MFCC, Chroma & Mel Spectrogram extraction |
| Scikit-Learn     | MLP Classifier, Train-Test Split, Metrics, Scaler  |
| NumPy            | Feature matrix construction & numerical operations  |
| Pandas           | Data organization and label management              |
| Matplotlib       | Waveform & Spectrogram visualization                |
| Seaborn          | Confusion matrix heatmap                            |
| Soundfile        | Audio file reading support                          |
| Jupyter Notebook | Interactive development environment                 |

---

## 🚀 Getting Started

### Prerequisites

Make sure Python 3.8+ is installed. Install all required libraries:

```bash
pip install librosa scikit-learn numpy pandas matplotlib seaborn soundfile jupyter
```

> **Note:** Librosa may require additional system dependencies. On Windows, you may also need:
> ```bash
> pip install audioread
> ```

### Download the RAVDESS Dataset

1. Visit the [RAVDESS dataset on Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
2. Download and extract the dataset
3. Place the actor folders (`Actor_01/`, `Actor_02/`, ..., `Actor_24/`) in your project directory

### Clone the Repository

```bash
git clone https://github.com/AliRaza-Dev678/CodeAlpha_Speech_Recognition_System.git
cd CodeAlpha_Speech_Recognition_System
```

---

## ▶️ How to Run

1. Download the RAVDESS dataset and place actor folders in the project directory.

2. Launch Jupyter Notebook:
```bash
jupyter notebook
```

3. Open the notebook:
```
Speech_Emotion_Recognition.ipynb
```

4. Update the dataset path in the notebook to point to your RAVDESS folder:
```python
# Example path update
data_path = "/path/to/RAVDESS/Actor_*/*.wav"
```

5. Run all cells from top to bottom using **Kernel → Restart & Run All**.

---

## 📚 Key Concepts Covered

- ✅ **Audio Signal Processing** — Loading, visualizing, and analyzing `.wav` audio files with Librosa
- ✅ **Waveform Analysis** — Plotting raw audio amplitude over time
- ✅ **Spectrogram Visualization** — Converting audio to time-frequency representations
- ✅ **MFCC Extraction** — Extracting Mel-Frequency Cepstral Coefficients from speech
- ✅ **Chroma Feature Extraction** — Capturing pitch class content from audio
- ✅ **Mel Spectrogram Features** — Encoding audio energy across Mel-scaled frequency bins
- ✅ **Feature Engineering from Audio** — Building structured feature vectors from raw signals
- ✅ **Filename Label Parsing** — Decoding RAVDESS naming convention to extract emotion labels
- ✅ **Multi-class Classification** — Classifying 8 distinct emotional states
- ✅ **MLP Neural Network** — Using a feedforward neural network for audio classification
- ✅ **Classification Report** — Per-emotion precision, recall, F1-score analysis
- ✅ **Confusion Matrix** — Visualizing which emotions are confused with each other

---

## 🌍 Real-World Applications

Speech Emotion Recognition has exciting applications across many industries:

| Domain                  | Use Case                                                         |
|-------------------------|------------------------------------------------------------------|
| 🤖 Virtual Assistants   | Siri, Alexa detecting frustration and adapting responses         |
| 📞 Call Centres         | Real-time agent alerts when customer emotion turns negative      |
| 🏥 Mental Health        | Tracking emotional state changes in therapy or monitoring apps   |
| 🎮 Gaming               | Adaptive NPCs that respond to how the player is feeling          |
| 🚗 Automotive           | Driver emotion monitoring for safety (drowsiness, anger)         |
| 🎓 E-Learning           | Detecting student confusion or boredom to adapt content          |
| 🎬 Media & Film         | Automated emotion tagging for content moderation                 |

---

## 👨‍💻 Author

**Ali Raza**
*CodeAlpha Machine Learning Intern*

- GitHub: [@AliRaza-Dev678](https://github.com/AliRaza-Dev678)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute it.

---

<p align="center">
  ⭐ If you found this project helpful, please give it a star! ⭐
</p>
