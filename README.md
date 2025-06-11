
# Cancer Type Prediction Web Tool 🧬

This is an AI-powered web application for predicting cancer types based on gene expression data. It integrates transcriptomic and proteomic insights to provide real-time diagnostic support using machine learning models.

---

## 📌 Project Overview

This project was developed as part of our B.Tech Mini Project titled:

**“Integrated Proteome and Transcriptome Analysis for Pan Cancer Pathway Identification and Biomarker Discovery”**  

---

## 💡 Features

- 🔍 Predicts **Cancer Type** using gene symbol and expression status (Up/Down/Unknown).
- 📊 Displays top 5 predicted cancer types with confidence scores.
- 🧠 Uses **XGBoost** or **Random Forest** models trained on multi-omics data.
- 📈 Shows **feature importance visualization** for model interpretability.

---



## 🧪 How It Works

- The user enters a **gene symbol** and optional **expression status**.
- The app encodes inputs using saved label encoders.
- The model predicts the most probable cancer type and returns the top predictions with probabilities.
- A bar chart shows the predicted distribution.
- Feature importance is visualized as a static plot.

---




