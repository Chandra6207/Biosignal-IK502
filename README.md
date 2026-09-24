# Biosignal-IK502

# IK 502 — PPG-Derived Heart Rate Variability (HRV) Analysis

## Introduction

This project implements a Python-based processing and analysis pipeline for **Photoplethysmography (PPG)** data to derive **Heart Rate Variability (HRV)** measures.

The analysis compares HRV across three study phases:

1. **Pre-Intervention**
2. **During-Intervention**
3. **Post-Intervention**

This project was developed as part of **IK 502 — Introduction to Biosignals**.

---

## Objective

The objective of this project is to process raw PPG sensor data and perform HRV analysis across the three experimental phases.

The analysis pipeline includes:

- Data import
- Study-protocol segmentation
- PPG signal filtering
- Pulse/peak detection
- Beat-to-beat interval extraction
- Time-domain HRV analysis
- Frequency-domain HRV analysis
- Comparison of HRV measures across study phases
- Visualization of processed PPG and HRV results

---

## Study Protocol

The recording consists of three major phases:

| Phase | Duration |
|---|---:|
| Pre-Intervention | 5 minutes |
| During-Intervention | 10 minutes |
| Post-Intervention | 5 minutes |

The **During-Intervention** phase is further divided into:

| Segment | Duration |
|---|---:|
| Listening 1 | 3 minutes |
| Task while listening | 4 minutes |
| Listening 2 | 3 minutes |

The segmentation is implemented directly in the analysis script.

---

## Data

The PPG recordings were obtained from an **EmotiBit** sensor.

The raw data contain a `PG` channel, which is used for the PPG analysis.

The three recordings used in the analysis are:

- Pre-Intervention
- During-Intervention
- Post-Intervention

The raw physiological data are **not included in this GitHub repository**. The `data/` directory is excluded through `.gitignore`.

---

## Signal Processing Pipeline

The analysis follows this processing pipeline:

```text
Raw PPG
   │
   ▼
Data Import
   │
   ▼
Study Protocol Segmentation
   │
   ▼
Bandpass Filtering
   │
   ▼
PPG Pulse Peak Detection
   │
   ▼
Beat-to-Beat Interval Extraction
   │
   ▼
RR Interval Validation
   │
   ├───────────────────┐
   ▼                   ▼
Time-Domain        Frequency-Domain
HRV Analysis       HRV Analysis
   │                   │
   ├── SDNN            ├── LF Power
   └── RMSSD           └── HF Power
   │                   │
   └─────────┬─────────┘
             ▼
       Phase Comparison
