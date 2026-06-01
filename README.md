# 🌦️ AZ-104 Data Cleaning & Machine Learning Pipeline
### Live Weather Data — Lodi, NJ

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/webuild262-boop/az104-weather-ml-pipeline/blob/main/az104_data_cleaning.ipynb) ![Python](https://img.shields.io/badge/Python-3.13.2-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange) ![Open-Meteo](https://img.shields.io/badge/API-Open--Meteo-green) ![Coursera](https://img.shields.io/badge/Coursera-Foundations%20of%20AI%20%26%20ML-lightblue)

---

## 🚀 Run This Pipeline — No Install Required

| Method | How | Best For |
|--------|-----|----------|
| Google Colab | Click the badge above then Run All | Anyone, zero setup |
| GitHub Codespaces | Code → Codespaces → New | VS Code in browser |
| Local Python | Clone → pip install -r requirements.txt | Developers |

---

## 📌 Project Overview

A complete end-to-end data engineering and machine learning pipeline in Python. Fetches live hourly weather data for Lodi, NJ, cleans it, engineers features, trains a Random Forest classifier (95.8% accuracy), and generates a 7-day rain forecast.

Built for the Coursera Foundations of AI and Machine Learning course.

---

## 🗂️ Pipeline Steps

| Step | Name | Description |
|------|------|-------------|
| 1 | Fetch Live Data | Pulls 360 hourly rows from Open-Meteo API |
| 2 | Explore Data | Inspects shape, dtypes, distributions |
| 3 | Inject Missing Values | Simulates ~8% dirty data |
| 4 | Visualise Nulls | missingno bar chart and matrix |
| 5 | Impute Missing Values | Mean for temp/humidity, median for precip/wind |
| 6 | Detect & Cap Outliers | IQR method with before/after boxplots |
| 7 | Feature Engineering | hour, day_of_week, is_daytime, feels_like_temp |
| 8 | Encode & Scale | LabelEncoder, StandardScaler, MinMaxScaler |
| 9 | Summary Statistics | Descriptive stats on cleaned dataset |
| 10 | Export | Saves CSV and PNG charts |
| 11 | Rain Prediction Model | Random Forest — 95.8% accuracy |
| 12 | 7-Day Live Forecast | Hourly rain probabilities for next 7 days |

---

## ⚙️ Local Setup

git clone https://github.com/webuild262-boop/az104-weather-ml-pipeline.git
cd az104-weather-ml-pipeline
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

---

## 👤 Author

Allen — Lodi, NJ | Coursera: Foundations of AI and Machine Learning | May 2026
