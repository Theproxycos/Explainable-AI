# XAI Robustness in Network Intrusion Detection

Research project evaluating the robustness of Explainable AI (XAI) 
methods — SHAP and LIME — under adversarial attacks in cybersecurity 
intrusion detection systems.

## Dataset
CSE-CIC-IDS2018 (Canadian Institute for Cybersecurity)

## Stack
- XGBoost — baseline classifier
- SHAP (TreeExplainer) — explainability layer
- LIME (LimeTabularExplainer) — explainability layer
- ART + FGSM — adversarial attack generation
- Scikit-learn — preprocessing and feature selection

## Project Phases
1. Data Engineering & Baseline (XGBoost > 95% accuracy)
2. XAI Layer Implementation (SHAP vs LIME latency)
3. Adversarial XAI Attacks (FGSM via ART)
4. Benchmarking & Metrics (Jaccard, Euclidean, Throughput)
5. Legal Alignment (NIST AI RMF, EU AI Act)

## Academic Context
MSc in Cybersecurity — Applied Machine Learning for Security
