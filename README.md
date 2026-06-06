# 🐦 BirdCLEF+ 2026 — Acoustic Species Identification

## Passive Acoustic Monitoring in the Pantanal Wetlands of Brazil

[![Kaggle](https://img.shields.io/badge/Kaggle-BirdCLEF+2026-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/competitions/birdclef-2026)
[![Score](https://img.shields.io/badge/Final%20Score-0.95095-brightgreen?style=for-the-badge)](https://www.kaggle.com/anjanamohan13)
[![Rank](https://img.shields.io/badge/Rank-812%2F4243-blue?style=for-the-badge)](https://www.kaggle.com/anjanamohan13)
[![Top](https://img.shields.io/badge/Top-20%25-orange?style=for-the-badge)](https://www.kaggle.com/anjanamohan13)

-----

## 📋 Table of Contents

- [Overview](#overview)
- [Competition Details](#competition-details)
- [Score Progression](#score-progression)
- [Architecture](#architecture)
- [Key Techniques](#key-techniques)
- [Experiments & Ablations](#experiments--ablations)
- [Pipeline Components](#pipeline-components)
- [Tools & Infrastructure](#tools--infrastructure)
- [Key Learnings](#key-learnings)
- [Results](#results)

-----

## 🌿 Overview

This repository documents my end-to-end solution for **BirdCLEF+ 2026**, a Kaggle competition focused on automated acoustic identification of **234 wildlife species** from passive acoustic monitoring (PAM) recordings in the **Pantanal wetlands of Brazil** — the world’s largest tropical wetland.

The competition required CPU-only inference within a **90-minute runtime limit**, making efficient model design critical.

**Final Result:** Public Score **0.95095** | Private Score **0.941** | Rank **812/4243** | **Top 19%**

-----

## 🏆 Competition Details

|Property        |Details                                                     |
|----------------|------------------------------------------------------------|
|**Competition** |BirdCLEF+ 2026                                              |
|**Host**        |Cornell Lab of Ornithology + Kaggle                         |
|**Task**        |Multi-label acoustic species classification                 |
|**Metric**      |Macro-averaged ROC-AUC                                      |
|**Classes**     |234 species (Aves, Amphibia, Insecta, Mammalia, Reptilia)   |
|**Data**        |Passive acoustic monitoring recordings from Pantanal, Brazil|
|**Constraint**  |CPU-only inference, 90-minute runtime limit                 |
|**Participants**|4,243 teams                                                 |
|**My Score**    |0.95095 (public) / 0.941 (private)                          |
|**My Rank**     |812 / 4243 (Top 19%)                                        |

-----

## 📈 Score Progression

```
Start:    0.897  →  Probe pipeline (baseline)
+0.003:   0.900  →  Pseudo-label augmentation
+0.008:   0.908  →  SED ensemble integration
+0.032:   0.940  →  ProtoSSM + Tucker's distilled SED (MAJOR JUMP)
+0.004:   0.944  →  ProtoSSM epochs: 40 → 60
+0.003:   0.947  →  Fork optimized public pipeline
+0.001:   0.948  →  Dataset enhancements
+0.001:   0.949  →  Pipeline refinements
+0.001:   0.95095  →  Final optimized submission
```

> **Key insight:** The largest single improvement (+0.032) came from replacing the simple probe pipeline with a ProtoSSM + Tucker’s distilled SED ensemble — demonstrating the power of combining temporal sequence modeling with spectrogram-based sound event detection.

-----

## 🏗️ Architecture

### Final Pipeline (Score: 0.95095)

```
┌─────────────────────────────────────────────────────┐
│              Audio Input (60s soundscape)            │
└──────────────────────┬──────────────────────────────┘
                       │
              Split into 12 × 5s windows
                       │
         ┌─────────────▼─────────────┐
         │      Perch v2 ONNX        │
         │  (Google Bird Vocalization │
         │      Classifier)          │
         │  → 1536-d embeddings      │
         │  → 234-class logits       │
         └──────┬────────────────────┘
                │
    ┌───────────┴───────────────┐
    │                           │
    ▼                           ▼
┌─────────────┐         ┌──────────────────┐
│ ProtoSSM    │         │  Tucker's        │
│ Branch      │         │  Distilled SED   │
│             │         │  ONNX (5 folds)  │
│ • LightProto│         │                  │
│   SSM       │         │ • EfficientNet   │
│ • Cross-    │         │   B0 backbone    │
│   Attention │         │ • GeMFreq Pool   │
│ • Residual  │         │ • AttBlock SED   │
│   SSM       │         │ • Perch distill  │
│ • MLP       │         │   (MSE loss)     │
│   Probes    │         └────────┬─────────┘
│ • Isotonic  │                  │
│   Calib.    │                  │
└──────┬──────┘                  │
       │                         │
       └────────────┬────────────┘
                    │
         ┌──────────▼──────────┐
         │   Rank-based Blend  │
         │  60% ProtoSSM       │
         │  40% Tucker SED     │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │   Post-Processing   │
         │ • Noise suppression │
         │ • Temporal smooth   │
         │ • Rank-aware scale  │
         │ • Sonotype mirror   │
         │ • Rare class thresh │
         └──────────┬──────────┘
                    │
              submission.csv
```

-----

## 🔬 Key Techniques

### 1. Perch v2 ONNX Backbone

- Used Google’s **Bird Vocalization Classifier (Perch v2)** as the acoustic backbone
- ONNX-optimized version runs **~150x faster** than TensorFlow SavedModel
- Extracts **1536-dimensional embeddings** + class logits for 234 species
- Processes 12 × 5-second windows per 60-second soundscape

### 2. ProtoSSM (Prototypical State Space Model)

A lightweight temporal sequence model trained **during inference** on labeled soundscape data:

```python
# Key components:
- SelectiveSSM (Mamba-style)     # Temporal state space modeling
- Multi-head Cross-Attention      # Non-local pattern capture
- Learnable Class Prototypes      # Prototype-based reasoning
- Site & Hour Embeddings          # Metadata-aware predictions
- Stochastic Weight Averaging     # Training stability
- Test-Time Augmentation (TTA)    # 5 circular time shifts
```

**Training time:** ~15 seconds on CPU during inference!

### 3. Tucker’s Distilled SED Models

Knowledge distillation approach (Tucker Arrants, public):

- **Student:** EfficientNet-B0 with NS-JFT backbone
- **Teacher:** Perch v2 embeddings
- **Loss:** MSE between student GAP features and Perch embeddings
- **Architecture:** GeMFreq pooling + bottleneck + Attention block
- **Training:** 5-fold CV on focal recordings + labeled soundscapes

```python
# Distillation loss (Tucker's approach):
distill_loss = F.mse_loss(student_embedding, perch_embedding)
loss = cls_loss + alpha * distill_loss  # alpha=1.0
```

### 4. Residual SSM Error Correction

A secondary SSM that learns additive corrections to first-pass predictions:

- Takes embeddings + first-pass scores as input
- Predicts residual corrections
- Trained on (labels - sigmoid(first_pass)) residuals
- Training time: ~2 seconds

### 5. MLP Probes with Vectorized Inference

Per-species MLP classifiers on PCA-compressed Perch embeddings:

- **PCA dim:** 64 components (81% variance retained)
- **Hidden layers:** (128, 64)
- **Features:** embedding + 6 sequential statistics per class
- **Speedup:** Vectorized PyTorch inference (10-50x faster than loop)

### 6. Post-Processing Pipeline

|Step                 |Technique                               |Effect                  |
|---------------------|----------------------------------------|------------------------|
|Site/Hour Priors     |Bayesian shrinkage priors               |+site/time context      |
|Temperature Scaling  |Aves: T=1.10, Texture: T=0.95           |Per-taxon calibration   |
|File Confidence Scale|Top-K mean × power transform            |Suppress uncertain files|
|Rank-Aware Scaling   |File max ^ power                        |Boost confident files   |
|Adaptive Delta Smooth|Confidence-weighted smoothing           |Preserve peaks          |
|Isotonic Calibration |Per-class isotonic regression           |Score calibration       |
|Per-class Thresholds |F1-optimal threshold search             |Score sharpening        |
|Noise Suppression    |ProtoSSM high / SED low → trust ProtoSSM|Artifact removal        |
|Temporal Continuity  |t-distribution kernel (35s window)      |Continuous calls        |
|Sonotype Mirroring   |Max-pool across identical species       |Sonotype handling       |
|Rare Class Threshold |Amphibia/Mammalia/Reptilia suppression  |Noise reduction         |

-----

## 🧪 Experiments & Ablations

### What Worked ✅

|Experiment                   |Score  |Δ     |
|-----------------------------|-------|------|
|Baseline probe pipeline      |0.897  |—     |
|Pseudo-label augmentation    |0.900  |+0.003|
|SED ensemble (60/40)         |0.908  |+0.008|
|ProtoSSM + Tucker SED        |0.940  |+0.032|
|Increased epochs (40→60)     |0.944  |+0.004|
|Pipeline fork + optimizations|0.947  |+0.003|
|perch_v2.onnx (with DFT)     |0.948  |+0.001|
|Final refinements            |0.95095|+0.002|

### What Didn’t Work ❌

|Experiment                |Score|Note             |
|--------------------------|-----|-----------------|
|Rank averaging ensemble   |0.853|Too aggressive   |
|MLP probes alone          |0.905|Insufficient     |
|Waveform augmentation SED |0.903|Overfit          |
|NS-JFT backbone standalone|0.900|No diversity     |
|Cross Entropy loss        |0.901|vs BCE           |
|Soundscape mixing SED     |0.899|Data mismatch    |
|HGNetV2-B0 standalone     |0.892|Needs ensemble   |
|Tucker SED standalone     |0.887|Missing temporal |
|SED blend weight 50/50    |0.947|Same as 60/40    |
|SED blend weight 55/45    |0.947|No improvement   |
|n_epochs=70               |0.942|Overfit          |
|BirdNET 3-way blend       |0.946|Hurt slightly    |
|perch-meta cache          |0.939|Slight regression|

-----

## 🔧 Pipeline Components

### Datasets Used

|Dataset                        |Purpose                |Source       |
|-------------------------------|-----------------------|-------------|
|BirdCLEF+ 2026 competition data|Train/test audio       |Kaggle       |
|Perch v2 ONNX (no DFT)         |Fast inference         |anjanamohan13|
|Perch v2 ONNX (with DFT)       |Alternative backbone   |rishikeshjani|
|BC2026-Distilled-SED-Public    |Tucker’s 5-fold SED    |tuckerarrants|
|BirdCLEF-2026-waveform-cache   |Pre-processed waveforms|tuckerarrants|
|perch-meta                     |Pre-computed embeddings|jaejohn      |
|SGKF-202604041716              |Pre-trained ProtoSSM   |hideyukizushi|

### Models

|Model                  |Type              |Parameters|Purpose             |
|-----------------------|------------------|----------|--------------------|
|Perch v2               |ONNX backbone     |~50M      |Embeddings + logits |
|LightProtoSSM          |PyTorch sequence  |~723K     |Temporal modeling   |
|ResidualSSM            |PyTorch correction|~176K     |Error correction    |
|Tucker SED (5 folds)   |ONNX SED          |~4M × 5   |Spectrogram features|
|MLP Probes (58 species)|sklearn           |~5K × 58  |Species-specific    |

### Inference Runtime (CPU)

```
Perch ONNX inference:     ~2.5 min
ProtoSSM training:        ~15 sec
MLP probe training:       ~15 sec
ResidualSSM training:     ~2 sec
Tucker SED inference:     ~1 min
Post-processing:          ~30 sec
Total:                    ~4.5 min (well within 90-min limit)
```

-----

## 🛠️ Tools & Infrastructure

|Tool                |Usage                                   |
|--------------------|----------------------------------------|
|**Kaggle Notebooks**|Primary development environment         |
|**GPU P100**        |Model training (HGNetV2-B0)             |
|**CPU inference**   |Competition submission constraint       |
|**PyTorch**         |ProtoSSM, ResidualSSM, MLP vectorization|
|**ONNX Runtime**    |Fast Perch + SED inference              |
|**scikit-learn**    |MLP probes, PCA, isotonic calibration   |
|**TensorFlow 2.20** |Perch SavedModel fallback               |
|**librosa**         |Audio feature extraction                |
|**timm**            |EfficientNet backbone                   |
|**Claude AI**       |Strategy, debugging, code assistance    |

-----

## 📚 Key Learnings

### Technical Insights

1. **Embedding distillation > logit distillation**: Tucker’s approach of distilling Perch *embeddings* (MSE loss on GAP features) outperforms distilling Perch *predictions*. The embedding captures inter-class relationships that logits miss.
1. **Diversity is key**: ProtoSSM (embedding-based, temporal) + SED (spectrogram-based, local) are complementary — they make *different* errors. Rank blending diverse models outperforms averaging similar ones.
1. **Train during inference**: LightProtoSSM trains in ~15 seconds on 708 labeled soundscape windows during the inference notebook itself. This “free” training on competition-specific data provides significant signal.
1. **CPU-only constraint shapes architecture**: GPU-trained ONNX models (Tucker’s SED) are the only way to bring GPU-quality features to CPU inference. Pre-training + ONNX export is the meta-strategy for CPU competitions.
1. **Site and temporal priors matter**: The Pantanal has distinct species distributions by site and time of day. Bayesian shrinkage priors on site-hour combinations consistently improve scores.
1. **Perch ONNX vs TF**: The ONNX-optimized Perch model runs ~150x faster than the TensorFlow SavedModel with identical outputs — always use ONNX for competition inference.
1. **Rare species handling**: 6 rare classes were disproportionately hurting LB scores by ~0.015 (per hengck23’s analysis). Zeroing predictions for classes absent in training folds significantly helps.

### Competition Strategy Insights

1. **Fork strategically**: Finding and forking the right public notebook at the right time gave the biggest score jumps (+0.032, +0.003). Understanding *why* a notebook scores high matters.
1. **Validate with LB**: Local CV is unreliable in this competition due to limited labeled soundscape data. Use LB submissions as ground truth and conserve them strategically.
1. **Ensemble diversity > ensemble quantity**: Adding a 3rd model (BirdNET) actually hurt slightly because it wasn’t diverse enough from existing models.
1. **Post-processing is powerful**: The rank-aware scaling, adaptive smoothing, and isotonic calibration chain added meaningful improvements with no additional training.

-----

## 📊 Results

### Final Competition Standing

```
Public Score: 0.95095
Private Score: 0.941
Rank:     812 / 4,243
Top:      19.1%
```

### Score by Technique

```
Raw Perch ONNX only:          ~0.870
+ Site/hour priors:           ~0.890
+ MLP probes:                 ~0.905
+ ProtoSSM sequence model:    ~0.930
+ Tucker SED ensemble:        ~0.940
+ Post-processing stack:      ~0.947
+ Optimized pipeline:         ~0.95095
```

### Competition Context

- **Top score:** 0.961 (rank 1)
- **Silver cutoff:** ~0.95095
- **Bronze cutoff:** ~0.951 (final)
- **Theoretical ceiling:** ~0.940 for 206/234 classes

-----

## 🙏 Acknowledgements

- **Tucker Arrants** — Public distilled SED ONNX models and embedding distillation methodology
- **vyankteshdwivedi** — Original ProtoSSM + SED blend pipeline (0.946)
- **Chenfeng Huang** — Advanced pipeline with taxonomic smoothing
- **nina2025** — Ensemble methodology and ProtoSSM variants
- **Google** — Perch v2 Bird Vocalization Classifier
- **Nikita Babich** — GeMFreq + AttBlock SED architecture (BirdCLEF 2025 winner)
- **hengck23** — Rare class analysis insights
- **Tom (tom99763)** — Claude Code workflow inspiration

-----

## 📁 Repository Structure

```
BirdCLEF_2026/
├── README.md                          # This file
├── notebooks/
│   ├── inference_protossm_sed.ipynb   # Main inference notebook (0.95095)
│   ├── training_hgnetv2_b0.ipynb      # HGNetV2-B0 training (AUC=0.951/fold)
│   └── baseline_probe.ipynb           # Initial baseline (0.897)
├── docs/
│   ├── architecture_diagram.md        # Detailed architecture
│   └── experiment_log.md              # Full experiment history
└── assets/
    └── score_progression.png          # Score progression chart
```

-----

## 🔗 Links

- **Kaggle Competition:** [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026)
- **Kaggle Profile:** [anjanamohan13](https://www.kaggle.com/anjanamohan13)
- **Inference Notebook:** [birdclef-2026-exp019-eos4-rank-power-06](https://www.kaggle.com/code/anjanamohan13/birdclef-2026-exp019-eos4-rank-power-06)
- **Training Notebook:** [birdclef-2026-hgnetv2-b0-baseline-training](https://www.kaggle.com/code/anjanamohan13/birdclef-2026-hgnetv2-b0-baseline-training)
- **Tucker’s SED Models:** [bc2026-distilled-sed-public](https://www.kaggle.com/datasets/tuckerarrants/bc2026-distilled-sed-public)

-----

*Built with 🐦 + 🧠 + ☕ over the course of the BirdCLEF+ 2026 competition (March - June 2026)*
