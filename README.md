# Project Argus: Domain-Agnostic Edge AI for ICS Security

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow_Keras-orange)
![Build](https://img.shields.io/badge/Status-Active_Research-success)

**Author:** Jhanvi Rani  
**Links:** [Read the Full Technical Report Here](https://docs.google.com/document/d/1dVBla_5wmcBrzfIwh6G9aXoX7YKcEwjJY_0r4tI9dAQ/edit?usp=sharing)

---



https://github.com/user-attachments/assets/e67f9d37-7e05-42eb-9575-afe646a766ca



---
## Overview
Project Argus is a lightweight, domain-agnostic anomaly detection framework engineered to protect air-gapped Industrial Control Systems (ICS) and safety-critical infrastructure (e.g., nuclear reactors, water treatment facilities) from sophisticated cyber-physical attacks. 

By shifting the computational burden of volatility tracking from the deep learning model to a deterministic feature-engineering layer, Argus successfully detects stealth attacks while maintaining a parameter footprint small enough for native Edge microprocessor deployment.

## The Core Innovation: Defeating Replay Attacks
Standard time-series neural networks suffer from a "Low-Entropy Blind Spot." When an adversary executes a Replay Attack (freezing a sensor's output to hide physical machinery sabotage) standard AI predicts the resulting flatline perfectly, yielding a near-zero Mean Squared Error (MSE) and failing to trigger an alarm.

**The Argus Solution:** Prior to tensor sequence creation, Argus implements an automated **Multivariate Rolling Variance** layer. 
* By mathematically quantifying the *absence* of natural thermodynamic noise, the system actively tracks signal volatility. 
* When a sensor is spoofed and flatlines, its variance drops to exactly `0.0`. 
* The LSTM-Autoencoder instantly flags this lack of physical entropy as a mathematical impossibility, triggering a high-probability anomaly alert.

## System Architecture
* **Dynamic Scaling:** Automatically reads incoming telemetry matrices, counts base sensors, and engineers N x 2 variance features on the fly.
* **Inference Engine:** A highly optimized Long Short-Term Memory Autoencoder (LSTM-AE).
* **UI/UX Layer:** A Streamlit-based diagnostic dashboard featuring a dynamic multi-select rendering protocol to prevent browser memory overload (200MB threshold) when analyzing high-frequency, multi-dimensional data.

## Empirical Validation
The architecture is validated against the **HAI 22.04 (HIL-based Augmented ICS) Security Dataset**.
During testing, the Argus pipeline successfully ingested 87 raw sensor streams, dynamically engineered 174 total features, and accurately detected the tightly coupled, stealth cyber-attacks hidden within the dataset without requiring code modification.

*(Dataset available via the [official icsdataset repository](https://github.com/icsdataset/hai))*

## Quick Start Guide

### 1. Train the Universal Model
Place your normal-operation telemetry CSV in the root directory. The pipeline will automatically conform to the data dimensionality and compile the custom AI.
```bash
python train_model.py <path_to_normal_data.csv>
```

### 2. Launch the Edge Inference Dashboard
Once the model and configuration arrays are saved, initialize the diagnostic interface to test against anomalous telemetry.
```bash
streamlit run app.py
```
