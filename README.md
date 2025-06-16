---
title: GenAI-ETGenderDashboard
emoji: 👁️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 👁️ GenAI-ETGenderDashboardGradio

An interactive Gradio dashboard for analyzing **eye-tracking metrics** with a focus on **gender-based differences in visual attention** while distinguishing between **AI-generated** and **real images**.

---

## 📌 Research Focus

**Research Question 3:**
> *Are there gender-based differences in gaze behavior when distinguishing between AI and real images?*

This dashboard supports the investigation of gaze behavior across gender, testing whether males and females exhibit statistically and visually distinguishable patterns in how they attend to different Areas of Interest (AOIs) when identifying AI-generated content.

---

## 🎯 Objectives

- To compare gaze behavior across gender while participants differentiate between AI and real images.
- To determine whether **eye-tracking metrics** differ significantly by gender using statistical tests.
- To **visually highlight** any gender-driven differences in attention distribution and scanning strategies.

---

## 🧪 Null & Alternative Hypotheses

**H₀ (Null):**  
There are no significant differences in gaze behavior metrics between male and female participants when distinguishing between AI-generated and real images.

**H₁ (Alternative):**  
There are significant differences in at least one gaze metric between genders, potentially varying across image content and AOIs.

---

## 👨‍🔬 Methodology Summary

- **Participants:** 84 students (53 male, 31 female), balanced using SMOTE.
- **Apparatus:** Tobii Pro Fusion eye tracker at 50Hz.
- **Stimuli:** 6 AI vs real image pairs; participants identify the AI-generated image.
- **Metrics collected:**
  - Total Fixation Duration
  - Fixation Count
  - Time to First Fixation
  - Total Visit Duration

- **AOIs** were manually defined using **Tobii Pro Lab**., grouped into facial features, backgrounds, and semantic image parts.  

---

## 📊 Dashboard Features

### ➤ Gender-Based Visualization Tools

- **Bar Charts:** Metric-by-metric comparison by gender across all AOIs and questions (Q1–Q6)
- **Scatter Plots:** Relationship between fixation duration and fixation count
- **Box Plots / Histograms / Violin Plots:** Distribution of visual attention metrics
- **Correlation Heatmaps:** Visual strategy patterns by gender

> 🧠 **After reviewing all visualizations**, it becomes clear that gender does influence gaze behavior — particularly in *Time to First Fixation* and *Total Visit Duration*, with noticeable divergence in Q2 to Q6.

[![View on Hugging Face](https://img.shields.io/badge/View%20on-HuggingFace-blueviolet?logo=huggingface&logoColor=white)](https://huggingface.co/spaces/RextonRZ/GenAI-ETGenderDashboard)

---

## 📈 Statistical Testing Methods

All visual results are statistically validated using:

| Test | Use Case |
|------|----------|
| **Shapiro–Wilk Test** | Assess metric normality per gender group |
| **Independent Samples t-test** | For normally distributed data (p ≥ 0.05) |
| **Mann–Whitney U Test** | For non-normal data (p < 0.05) |
| **Pearson Correlation** | Explore inter-metric dependencies |
| **SMOTE** | Synthetic balancing for gender parity |

**Key Result:** Mann–Whitney U tests revealed significant gender-based differences in visual attention for Q2 to Q6 , **rejecting the null hypothesis**. There are significant differences in at least one gaze metric between genders, potentially varying across image content and AOIs.

---

## 📂 File Overview

| File | Description |
|------|-------------|
| `app.py` | Main Gradio dashboard code |
| `Full Visualisations for RQ3.pdf` | Static graphs for RQ3 |
| `Assignment Report.pdf` | Full academic write-up of methodology, data collection, and results |
| `AOI Mapping.pdf` | Detailed AOI definitions per image |
| `requirements.txt` | Libraries for local or HF deployment |
| `GenAIEyeTrackingDatasetRQ3.ipynb` | Pre-dashboard notebook containing data cleaning, gender balancing (SMOTE), statistical tests and simple visualizations to explore gender-based differences |

---

## 🚀 Technologies Used

- **Frontend:** Gradio 4.44.0
- **Visualization:** Plotly, Seaborn
- **Data Processing:** Pandas, NumPy
- **Statistical Testing:** Scipy, Scikit-learn
- **File Handling:** OpenPyXL
- **Balancing:** imbalanced-learn (SMOTE)

---

## 💡 Key Insight

> **After thoroughly analyzing all the graphs and statistical outcomes, gender differences clearly affect visual attention based on certain gaze metrics. In particular, early fixation behavior (Time to First Fixation) and overall engagement (Total Visit Duration) reveal that males and females interact with AI versus real images in subtly distinct ways.**

---

## 📜 License

MIT License – feel free to reuse and build upon this work with attribution.

---
## 👥 Contributors

- Rui Zhe Ooi  
- Shin Yen Lee  
- Rui Zhe Khor  
- Iman Soffea  
- Min Zi Teoh  
- Jia Xin Low  
(Supervised by Dr. Unaizah Hanum)

---

## ❤️ Acknowledgements

Thanks to all participants and the Faculty of Computer Science, University of Malaya, for supporting this research.

---
