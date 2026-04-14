# Topological Analysis for Deepfake Audio Detection

## Overview
This project explores the use of **Topological Data Analysis (TDA)** for detecting synthetic (deepfake) audio.  
The primary goal is to evaluate whether topological features derived from audio representations provide **complementary information** to classical signal processing features such as MFCC.

The experiments are conducted on the **ASVspoof 2019 Logical Access (LA)** dataset.

---

## Key Idea
We compare three types of feature representations:

1. **MFCC-based features**
   - 13 MFCC coefficients
   - Aggregated using mean and standard deviation → 26 features

2. **Topological features (TDA)**
   - MFCC frames treated as a point cloud
   - Subsampling to reduce complexity
   - Persistent homology (H0 and H1) computed using Ripser
   - Diagram summary statistics (mean, std, max, count of lifetimes) → 8 features

3. **Hybrid representation**
   - Concatenation of MFCC and TDA features

---

## Models
The following models are evaluated:
- Logistic Regression (with feature scaling)
- Random Forest

---

## Dataset
- **ASVspoof 2019 LA**
- Balanced sampling: bonafide vs spoof
- Train/dev split follows official protocol

---

## Pipeline

1. Load audio (mono, 16 kHz, fixed duration)
2. Extract MFCC features
3. Build MFCC frame-based point cloud
4. Apply TDA (persistent homology)
5. Construct feature vectors
6. Train models
7. Evaluate using ROC-AUC and EER

---

## Metrics
- ROC-AUC
- Equal Error Rate (EER)
- Accuracy (optional)

---

## Preliminary Results
| Model | Features | ROC-AUC | EER |
|------|--------|--------|------|
| Random Forest | MFCC | ~0.95 | ~0.11 |
| Logistic Regression | TDA | ~0.72 | high |
| Random Forest | Hybrid | ~0.96 | ~0.10 |

Key observations:
- MFCC features are strong baseline
- TDA alone is weak
- TDA provides **consistent improvement when combined with MFCC**

---

## Research Questions

- Do topological features provide useful information for deepfake detection?
- Are TDA features effective on their own, or only in combination with classical features?
- Does TDA improve robustness under signal degradation (noise, compression)?
- Can richer representations (e.g. persistence images) improve performance?

---

## Future Work

- Robustness analysis (noise, compression)
- Persistence images / landscapes
- Sliding window / delay embeddings
- Additional models (e.g. boosting, neural networks)

---

## Tech Stack
- Python
- NumPy / Pandas
- Librosa (audio processing)
- Ripser (persistent homology)
- Scikit-learn (models & evaluation)




