# ❄️ Climate Resilient Transportation System — Snowy Road Condition Detection

> IEEE-grounded sensor fusion system for classifying winter road surface conditions using
> image features, meteorological data, temperature, and polarimetric radar analysis.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Papers](https://img.shields.io/badge/IEEE-3_Papers-orange)](references)

---

## 📖 Overview

This project implements a **multi-modal road surface classification and friction estimation
system** for winter driving safety. It fuses:

| Modality | Features | Source |
|---|---|---|
| RGB Camera Images | 33 features (color moments + GLCM texture) | Yang & Lei [1] |
| Meteorological Data | UV, illuminance, wind speed, humidity, temps | Yang & Lei [1] |
| Temperature | Surface & air temperature | Yang & Lei [1] |
| Polarimetric Radar | Target Entropy H, Auxiliary Angle α | Vassilev [2] |
| Computer Vision CNN | Friction factor regression (SIWNet) | Ojala & Seppänen [3] |

### Surface Classes
- **DR** — Dry
- **FS** — Fresh Snow
- **TI** — Transparent Ice
- **GS** — Granular Snow
- **MI** — Mixed Ice

---

## 📐 Equations (27 total from 3 IEEE papers)

All equations are implemented in `src/model.py`:

| # | Name | Formula | Source |
|---|---|---|---|
| 1 | GLCM Energy | `E = ΣΣ [G(i,j)]²` | [1] Eq.1 |
| 2 | Moment of Inertia | `I = Σ n²·[Σ|i−j|=n G(i,j)]` | [1] Eq.2 |
| 3 | Inv. Diff. Moment | `L = ΣΣ G(i,j)/[1+(i−j)²]` | [1] Eq.3 |
| 4 | GLCM Entropy | `EN = −ΣΣ G(i,j)·log G(i,j)` | [1] Eq.4 |
| 5 | Correlation | `C = ΣΣ(ij)G−μᵢμⱼ / σᵢσⱼ` | [1] Eq.5 |
| 6 | VIP Score | `VIP = √[k·Σr²(y,cₕ)w²ₕⱼ/Σr²]` | [1] Eq.6 |
| 7 | Average Precision | `AP = (1/n)·ΣPᵢ` | [1] Eq.7 |
| 8 | DRAP 3→4 | `DRAP = (AP₃−AP₄)/AP₃` | [1] Eq.8 |
| 9 | DRAP 4→5 | `DRAP = (AP₄−AP₅)/AP₄` | [1] Eq.9 |
| 10 | Coherence Vector | `k = [SHH+SVV, SHH−SVV, 2SHV]ᵀ` | [2] Eq.3 |
| 11 | Coherence Matrix | `T̂ = MM†/N` | [2] Eq.5 |
| 12 | Eigendecomposition | `T̂ = Σλᵢ[eᵢ·eᵢ†]` | [2] Eq.6 |
| 13 | Probability Weights | `Pᵢ = λᵢ/(λ₁+λ₂+λ₃)` | [2] Eq.8 |
| 14 | Target Entropy H | `H = −ΣPᵢ·log₃(Pᵢ)` | [2] Eq.7 |
| 15 | Auxiliary Angle α | `α = ΣPᵢ·arccos\|eᵢ₁\|` | [2] Eq.9 |
| 16 | Truncated Normal PDF | `p = φ((x−μ)/σ)/[Φ(b)−Φ(a)]` | [3] Eq.1 |
| 17 | Normal PDF φ | `φ = (1/σ√2π)·exp(−(x−μ)²/2σ²)` | [3] Eq.2 |
| 18 | Normal CDF Φ | `Φ = (1/2)[1+erf((b−μ)/σ√2π)]` | [3] Eq.3 |
| 19 | Neg. Log-Likelihood | `−ln p = ln σ+(μ−x)²/2σ²+ln…` | [3] Eq.4 |
| 20 | Batch Loss L | `L = Σᵢ −ln p(f̂ᵢ,σ̂ᵢ,a,b;fᵢ)` | [3] Eq.5 |
| 21 | Friction Normalization | `f = (grip−g_min)/(g_max−g_min)` | [3] Sec.III-A |
| 22 | Prediction Interval | `PI = [F⁻¹((1−c)/2), F⁻¹((1+c)/2)]` | [3] Sec.III-D |
| 23 | Interval Score IS | `IS = (u−l)+(2/α)[(l−y)⁺+(y−u)⁺]` | [3] Sec.III-D |
| 24 | CRPS | `CRPS = E\|X−y\|−(1/2)E\|X−X'\|` | [3] Sec.III-D |
| 25 | MAE | `MAE = (1/n)Σ\|fᵢ−f̂ᵢ\|` | [3] Table II |
| 26 | RMSE | `RMSE = √[(1/n)Σ(fᵢ−f̂ᵢ)²]` | [3] Table II |
| 27 | Radar Surface Score | `S = 0.6·(1−H)+0.4·(1−α/90)` | Derived [2] |

---

## 📊 12 Graphs Generated

| # | Graph | Key Equations |
|---|---|---|
| 1 | AP Comparison: All Methods × Scenarios | Eq. 7 |
| 2 | DRAP Analysis (3→4 and 4→5 classes) | Eq. 8, 9 |
| 3 | Polarimetric Entropy vs Alpha Scatter | Eq. 14, 15 |
| 4 | Friction Factor Distribution by Class | Eq. 21 |
| 5 | SIWNet Prediction Intervals | Eq. 16–22 |
| 6 | Confusion Matrix (IMTFM, 5-class PLS) | — |
| 7 | Temperature vs Friction Scatter | Eq. 21 |
| 8 | VIP Feature Importance | Eq. 6 |
| 9 | MAE & RMSE Model Comparison | Eq. 25, 26 |
| 10 | GLCM Texture Features per Class | Eq. 1–5 |
| 11 | Radar Surface Score vs Friction | Eq. 27, 14 |
| 12 | Interval Score & CRPS Comparison | Eq. 23, 24 |

---

## 🗂️ Project Structure

```
snowy-road-detection/
├── src/
│   ├── model.py            # 27 equations + data generation + PLS classification
│   └── generate_graphs.py  # 12 research graphs (matplotlib)
├── data/
│   └── road_condition_data.csv   # 600-sample synthetic dataset
├── results/
│   ├── graph1_ap_comparison.png
│   ├── graph2_drap_analysis.png
│   ├── graph3_entropy_alpha_scatter.png
│   ├── graph4_friction_distribution.png
│   ├── graph5_prediction_intervals.png
│   ├── graph6_confusion_matrix.png
│   ├── graph7_temperature_friction.png
│   ├── graph8_vip_features.png
│   ├── graph9_mae_rmse_comparison.png
│   ├── graph10_texture_features.png
│   ├── graph11_radar_friction_correlation.png
│   └── graph12_interval_crps.png
├── dashboard/
│   └── snowy-road-dashboard.jsx   # React dashboard (Recharts + Tailwind)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/snowy-road-detection.git
cd snowy-road-detection

# Install dependencies
pip install -r requirements.txt

# Run model & generate data
python src/model.py

# Generate all 12 graphs
python src/generate_graphs.py

# View graphs in results/
```

---

## 📦 Requirements

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
scipy>=1.11
matplotlib>=3.7
```

---

## 📚 References

[1] S. Yang and C. Lei, "Research on the Classification Method of Complex Snow and Ice Cover
    on Highway Pavement Based on Image-Meteorology-Temperature Fusion,"
    *IEEE Sensors Journal*, vol. 24, no. 2, pp. 1784–1791, Jan. 2024.

[2] V. Vassilev, "Road Surface Characterization Using a 77–81 GHz Polarimetric Radar,"
    *IEEE Transactions on Intelligent Transportation Systems*, vol. 25, no. 9,
    pp. 12829–12834, Sep. 2024.

[3] R. Ojala and A. Seppänen, "Lightweight Regression Model with Prediction Interval
    Estimation for Computer Vision-Based Winter Road Surface Condition Monitoring,"
    *IEEE Transactions on Intelligent Vehicles*, vol. 10, no. 4, pp. 2206–2218, Apr. 2025.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
