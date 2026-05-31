# Feature Scaling, Autocorrelation & Logistic Regression — ML Fundamentals Notebook

A hands-on Python notebook covering essential machine learning preprocessing and analysis techniques, including Z-score normalization, autocorrelation analysis across multiple real-world domains, and logistic regression with feature importance.

---

## Overview

This notebook is a practical introduction to foundational machine learning preprocessing and signal analysis techniques. It walks through several self-contained examples using realistic data from domains such as computer vision, healthcare, IoT/robotics, weather forecasting, and finance. Each example is easy to understand and extend.

---

## Notebook Structure

Cell 1 — Standard Scaling (Z-score normalization): visualizes original vs. scaled features side by side. Domain: General / Computer Vision

Cell 2 — Autocorrelation Plot: stem plot of lag-based correlation for a simulated signal. Domain: Signal Processing

Cell 3 — Robotic Arm Vibration Analysis: autocorrelation of a noisy repeating vibration cycle over 10 seconds. Domain: Robotics / IoT

Cell 4 — Patient Vital Signs Normalization: scales heart rate, body temperature, and blood pressure to a common range. Domain: Healthcare

Cell 5 — Daily Temperature Autocorrelation: detects weekly seasonality in 30 days of simulated temperature data. Domain: Weather / Time Series

Cell 6 — Loan Approval with Logistic Regression: trains a model and extracts feature importance weights. Domain: Finance / Credit Scoring

---

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- scikit-learn

Install all dependencies:

pip install numpy pandas matplotlib scikit-learn

---

## Getting Started

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
jupyter notebook Untitled1.ipynb

No external datasets needed — all data is generated inline inside the notebook.

---

## Concepts Covered

Feature Scaling
- Z-score normalization using StandardScaler
- Why scaling matters when features have different units or magnitudes

Autocorrelation Analysis
- Computing lag-based autocorrelation with pandas autocorr()
- Identifying repeating patterns and seasonal cycles in time series

Logistic Regression & Feature Importance
- Binary classification with LogisticRegression from scikit-learn
- Interpreting model coefficients as feature weights
- Scaling features before training

---

## License

This project is open source and available under the MIT License.
