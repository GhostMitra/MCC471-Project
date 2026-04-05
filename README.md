# 🌱 AI-Driven Crop Recommendation for Climate-Resilient & Sustainable Agriculture

### Course: Sustainability, Climate Actions & Environmental Sciences (MCC471)
### Topic Area: D — Climate Change & Environmental Data Analytics

> An end-to-end machine learning project with data preprocessing, exploratory
> climate-soil analysis, model training, evaluation, and **Explainable AI (SHAP + LIME)** —
> built for Jupyter Lab and aligned with UN Sustainable Development Goals.

---

## 🌍 SDG Alignment

| SDG | Goal | How This Project Contributes |
|-----|------|------------------------------|
| **SDG 2** | Zero Hunger | Reduces crop failure by recommending climate-suited crops |
| **SDG 6** | Clean Water & Sanitation | Rainfall-aware recommendations prevent water waste |
| **SDG 12** | Responsible Consumption & Production | Prevents over-application of N, P, K fertilisers |
| **SDG 13** | Climate Action | Temperature & Rainfall are primary model drivers |
| **SDG 15** | Life on Land | Matches crop demands to soil composition; protects soil health |

---

## 📁 Repository Structure

```
├── Crop_Recommendation_MCC471.ipynb    ← Main Jupyter notebook (MCC471 version)
├── Train_Dataset.csv                   ← Training data  (18,079 samples)
├── Test_Dataset.csv                    ← Test data      (18,079 samples)
├── requirements.txt                    ← All dependencies
└── README.md                           ← This file
```

---

## 🌾 Project Overview

Climate change is reshaping agricultural ecosystems — shifting rainfall patterns,
rising temperatures, and increasing soil stress make traditional crop selection
unreliable. This project applies **Machine Learning and Explainable AI (XAI)** to
analyse environmental and soil data, recommending optimal crops that are both
**climate-adapted and resource-efficient**.

By treating Rainfall and Temperature as first-class model features alongside soil
nutrients, this project directly operationalises **SDG 13 (Climate Action)** and
**SDG 2 (Zero Hunger)** — demonstrating that data-driven precision agriculture is
a viable pathway toward sustainable food systems.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | [IEEE DataPort – Crop Recommendation Dataset](https://ieee-dataport.org/documents/crop-recommendation-dataset) |
| **Size** | 18,079 samples (80/20 internal split) |
| **Classes** | 40 crop types across diverse agro-climatic zones |
| **Missing values** | None |

### Features

| Column | Description | Unit | SDG Dimension |
|---|---|---|---|
| `N` | Nitrogen content in soil | mg/kg | 🌿 SDG 12 / SDG 15 |
| `P` | Phosphorus content in soil | mg/kg | 🌿 SDG 12 / SDG 15 |
| `K` | Potassium content in soil | mg/kg | 🌿 SDG 12 / SDG 15 |
| `pH` | Soil acidity / alkalinity | — | 🌿 SDG 15 |
| `rainfall` | Annual rainfall | mm | 🌡️ SDG 13 / SDG 6 |
| `temperature` | Mean annual temperature | °C | 🌡️ SDG 13 |
| `Crop` | **Target** — recommended crop | 40 classes | — |

---

## 🤖 Models Trained

| Rank | Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| 🥇 | **Decision Tree** *(selected)* | **95.50%** | **95.56%** | **95.50%** | **95.52%** |
| 🥈 | Random Forest | 95.50% | 95.43% | 95.50% | 95.45% |
| 🥉 | KNN | 94.91% | 94.88% | 94.91% | 94.89% |
| 4 | SVM | 93.00% | 88.52% | 93.00% | 90.27% |
| 5 | Naive Bayes | 92.85% | 88.37% | 92.85% | 90.12% |

> **Best model: Decision Tree** — selected for highest precision (95.56%),
> full interpretability (human-readable rules → SDG 2 farmer adoption),
> and fast inference suitable for mobile/edge deployment in rural areas.

---

## 📓 Notebook Structure (`Crop_Recommendation_MCC471.ipynb`)

| Section | Title | SDG Focus |
|---|---|---|
| — | Cover page & SDG alignment table | All |
| 1 | Imports & configuration | — |
| 2 | Load dataset | — |
| 3a | Statistical summary | SDG 13 |
| 3b | Feature distributions (soil & climate) | SDG 12, 13, 15 |
| 3c | Crop class distribution | SDG 2 |
| 3d | Feature boxplots & variability | SDG 13 |
| 3e | Feature correlation heatmap | SDG 13 |
| **3f** | **🌡️ Climate-Crop Analysis** *(new)* | **SDG 13, SDG 6** |
| **3g** | **🌿 Soil Nutrient Overuse Analysis** *(new)* | **SDG 12, SDG 15** |
| 4 | Data preprocessing | — |
| 5 | Train 5 ML models | SDG 2 |
| 6a–f | Model evaluation — table, charts, confusion matrix, feature importance | SDG 2, 13 |
| 7a–h | **SHAP** — global bar, beeswarm, violin, force plots, dependence, heatmap, waterfall | SDG 13, 15 |
| 8a–f | **LIME** — batch implementation, surrogate quality, global importance, individual explanations, heatmap | SDG 2, 13 |
| 9 | SHAP vs LIME consistency check | SDG 13 |
| **10** | **🌍 Sustainability Impact Assessment** *(new)* | **All SDGs** |
| 11 | Summary & conclusions | All |
| 12 | 🌾 Climate-Adaptive Crop Advisory Tool (live predictor) | SDG 2, 13 |
| 13 | Challenges & limitations | — |
| 14 | Future scope | SDG 2, 6, 11, 13 |

---

## 🔍 Explainable AI — SHAP & LIME

Both XAI methods are implemented **from scratch** using only `scikit-learn` and
`numpy` — no external `shap` or `lime` packages are required to run the notebook.

### Why XAI matters for sustainability

A black-box model that farmers cannot understand will not be adopted in the field.
SHAP and LIME make the system **transparent and trustworthy** — explaining *why*
a specific crop is recommended given the exact climate and soil conditions of a
field. This directly supports **SDG 2** (farmer uptake) and **SDG 13** (validating
climate drivers).

### SHAP — SHapley Additive exPlanations

```
Theory  : Cooperative game theory — each feature receives its fair share
          of the prediction as the average marginal contribution across
          all possible feature orderings.

Method  : Tree-path contributions
          For each tree, decision paths are traced and the change in
          leaf-class probability at each split is credited to the
          responsible feature. Averaged across all trees.

SDG use : Quantifies the % contribution of Climate vs Soil features →
          directly measures SDG 13 (Climate Action) impact on recommendations.

Plots   : Global bar (Climate vs Soil colour-coded) · Beeswarm · Violin ·
          Force plots · Dependence · Per-crop heatmap · Waterfall
```

### LIME — Local Interpretable Model-Agnostic Explanations

```
Theory  : Any complex model can be approximated linearly in a small
          local neighbourhood around a single prediction.

Algorithm:
  1. Perturb — generate Gaussian noise samples around the field input
  2. Query  — get crop probabilities from the trained model
  3. Weight — rank by proximity (Gaussian kernel)
  4. Fit    — weighted Ridge regression (local surrogate)
  5. Read   — surrogate coefficients = local climate/soil importance

SDG use : Provides per-field explanations that agricultural extension
          workers can communicate directly to farmers → SDG 2.

Plots   : Surrogate quality distribution · Global importance ·
          Individual field explanations · Per-crop coefficient heatmap ·
          Coefficient distributions
```

### SHAP vs LIME Comparison

| Property | SHAP | LIME |
|---|---|---|
| **Scope** | Global + Local | Local (aggregated globally) |
| **Theory** | Game theory (Shapley values) | Local linear approximation |
| **Consistency** | Guaranteed by axioms | Depends on kernel & sampling |
| **Speed** | O(n × depth × n_trees) | O(n × n_perturbations) |
| **SDG 13 use** | Confirms climate variable dominance globally | Shows climate impact locally per field |
| **SDG 2 use** | Model-level audit for policymakers | Field-level explanation for farmers |
| **Output** | Signed feature contributions | Linear surrogate coefficients |

---

## 🌱 Key Sustainability Findings

| Finding | Environmental Significance | SDG |
|---|---|---|
| Climate features (Rainfall + Temperature) are top drivers | Climate is the primary crop suitability determinant | SDG 13 |
| Potassium (K) is the top soil driver | Guides precision fertilisation; prevents overuse | SDG 12, 15 |
| Crops cluster into clear rainfall bands (<100mm / >1000mm) | Tool guides climate migration of crops under shifting rainfall | SDG 13, 6 |
| SHAP–LIME agreement > 0.85 | Robust, trustworthy insights safe for policy recommendations | SDG 2 |

---

## 🚀 Getting Started

### 1. Clone / download the project
```bash
git clone https://github.com/your-username/MCC471-Crop-Recommendation.git
cd MCC471-Crop-Recommendation
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Lab
```bash
jupyter lab
```

### 5. Open the notebook
`Crop_Recommendation_MCC471.ipynb`

> Make sure `Train_Dataset.csv` and `Test_Dataset.csv` are in the **same folder**
> as the notebook before running.

### 6. Run all cells
`Kernel → Restart Kernel and Run All Cells`

---

## 💡 Usage — Climate-Adaptive Crop Advisory Tool

```python
# Scenario 1 — Rice-growing region (high rainfall, warm climate)
recommend_crop(N=80, P=40, K=40, pH=5.66, rainfall=297.66, temperature=29.57)

# Output:
# ══════════════════════════════════════════════════════════
#   🌱 CLIMATE-ADAPTIVE CROP ADVISORY TOOL
#   Course: MCC471 — Sustainability & Climate Sciences
# ──────────────────────────────────────────────────────────
#   🌿 Soil    : N=80 mg/kg  P=40 mg/kg  K=40 mg/kg  pH=5.66
#   🌡️  Climate : Rainfall=297.66 mm   Temperature=29.57°C
# ──────────────────────────────────────────────────────────
#   📋 Top 3 Crop Recommendations:
#   #1  rice                ██████████████████████████████  98.7%
#   #2  maize               ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.8%
#   #3  wheat               ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0.5%
# ══════════════════════════════════════════════════════════

# Scenario 2 — Semi-arid region (low rainfall, mild temperature)
recommend_crop(N=70, P=40, K=45, pH=5.54, rainfall=75.32, temperature=22.68)

# Scenario 3 — Climate-stressed zone (low water, cool)
recommend_crop(N=20, P=30, K=20, pH=7.00, rainfall=55.0, temperature=18.0)
```

---

## 🌍 Sustainability Impact Assessment

| Impact Area | Expected Benefit | SDG |
|---|---|---|
| Food Security | Reduces crop failure via climate-adaptive recommendations | SDG 2 |
| Water Conservation | Avoids water-intensive crops in low-rainfall zones | SDG 6 |
| Fertiliser Efficiency | Prevents N/P/K overuse by matching to actual soil state | SDG 12 |
| Climate Resilience | Adapts recommendations to temperature and rainfall shifts | SDG 13 |
| Soil Health | Matches crop nutrient demands to soil composition | SDG 15 |

---

## 🔭 Future Scope

| Extension | SDG Impact |
|---|---|
| Integrate IoT soil sensors + real-time weather APIs | SDG 13 — live climate adaptation |
| Add **carbon footprint score** per crop recommendation | SDG 13 — emissions-aware farming |
| Include **water consumption estimate** per crop | SDG 6 — precision water management |
| **LSTM forecasting** for seasonal climate trends | SDG 13 — forward-looking recommendations |
| Mobile app with multilingual support for rural farmers | SDG 2 — inclusive farmer access |
| District/state-level policy layer with GIS integration | SDG 11, 13 — urban-rural sustainability |
| Extend to global agro-climatic zones (Africa, SE Asia) | SDG 2 — food security at scale |

---

## 🛠 Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| ML Framework | scikit-learn |
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| XAI | SHAP & LIME (custom implementations; no external packages needed) |
| Environment | Jupyter Lab |

---

## ⚠️ Challenges & Limitations

| Challenge | Detail |
|---|---|
| Static climate data | Dataset uses historical averages; real deployment needs live weather feeds |
| No temporal dimension | Cannot model seasonal shifts without time-series (LSTM) extension |
| Regional specificity | Represents Indian agro-climatic zones; not globally generalisable as-is |
| Missing variables | No pest risk, market price, water table, or soil moisture data |
| Rural deployment | Mobile/edge infrastructure needed for smallholder farmer access |

---

## 📄 References

- **Dataset:** [IEEE DataPort – Crop Recommendation Dataset](https://ieee-dataport.org/documents/crop-recommendation-dataset)
- **SHAP paper:** Lundberg & Lee, *NeurIPS 2017* — "A Unified Approach to Interpreting Model Predictions"
- **LIME paper:** Ribeiro et al., *KDD 2016* — "Why Should I Trust You? Explaining the Predictions of Any Classifier"
- **SDG Framework:** United Nations, [Sustainable Development Goals](https://sdgs.un.org/goals)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
