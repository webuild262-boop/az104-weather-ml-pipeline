# ============================================================
#  AZ-104 Data Lab — Real-World Data Cleaning & Preprocessing
#  Data Source : Open-Meteo Weather API (Lodi, NJ)
#  Libraries   : pandas, numpy, scikit-learn, missingno
#  Author      : Allen | Date: May 31, 2026
# ============================================================

import requests
import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# SECTION 1 — FETCH REAL-WORLD DATA FROM OPEN-METEO API
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  STEP 1: Fetching live weather data for Lodi, NJ...")
print("=" * 60)

LAT = 40.8776
LON = -74.0826

url = (
    "https://api.open-meteo.com/v1/forecast"
    f"?latitude={LAT}&longitude={LON}"
    "&hourly=temperature_2m,relative_humidity_2m,"
    "precipitation,windspeed_10m,weathercode"
    "&timezone=America%2FNew_York"
    "&past_days=14&forecast_days=1"
)

try:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    raw = response.json()
    df = pd.DataFrame(raw["hourly"])
    print(f"  ✅ Data fetched! Shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
except Exception as e:
    print(f"  ⚠️  API call failed ({e}). Loading offline sample data...\n")
    np.random.seed(42)
    n = 72
    times = pd.date_range("2026-05-17", periods=n, freq="h")
    df = pd.DataFrame({
        "time"                 : times.strftime("%Y-%m-%dT%H:%M"),
        "temperature_2m"       : np.random.normal(18, 5, n),
        "relative_humidity_2m" : np.random.normal(65, 15, n),
        "precipitation"        : np.random.exponential(0.5, n),
        "windspeed_10m"        : np.random.normal(12, 4, n),
        "weathercode"          : np.random.choice([0,1,2,3,45,61,80], n),
    })

# ─────────────────────────────────────────────────────────────
# SECTION 2 — INITIAL EXPLORATION
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  STEP 2: Initial Data Exploration")
print("=" * 60)

print(f"\n📐 Shape      : {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\n📋 Columns    : {list(df.columns)}")
print(f"\n🔢 Data Types :\n{df.dtypes}")
print(f"\n📊 First 5 rows:\n{df.head()}")
print(f"\n📈 Statistics :\n{df.describe()}")

# ─────────────────────────────────────────────────────────────
# SECTION 3 — INJECT MISSING VALUES (for practice)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 3: Injecting Missing Values for Practice")
print("=" * 60)

np.random.seed(7)
for col in ["temperature_2m","relative_humidity_2m",
            "precipitation","windspeed_10m"]:
    idx = np.random.choice(df.index, size=int(len(df)*0.08), replace=False)
    df.loc[idx, col] = np.nan

print(f"\n🔍 Missing values per column:\n{df.isnull().sum()}")
print(f"\n📉 Missing % per column:\n{round(df.isnull().mean()*100,2)}")

# ─────────────────────────────────────────────────────────────
# SECTION 4 — VISUALISE MISSING DATA (missingno)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 4: Visualising Missing Data with missingno")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
msno.bar(df, ax=axes[0], color="#2196F3", fontsize=11)
axes[0].set_title("Missing Data — Bar Chart", fontsize=13, fontweight="bold")
msno.matrix(df, ax=axes[1], sparkline=False, fontsize=11)
axes[1].set_title("Missing Data — Matrix", fontsize=13, fontweight="bold")
plt.suptitle("Lodi, NJ — Open-Meteo Weather (Missing Value Analysis)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("missing_data_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✅ Chart saved → missing_data_analysis.png")

# ─────────────────────────────────────────────────────────────
# SECTION 5 — IMPUTE MISSING VALUES
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 5: Handling Missing Values (Imputation)")
print("=" * 60)

mean_imputer = SimpleImputer(strategy="mean")
df[["temperature_2m","relative_humidity_2m"]] = mean_imputer.fit_transform(
    df[["temperature_2m","relative_humidity_2m"]])
print("  ✅ temperature_2m & relative_humidity_2m → imputed with MEAN")

median_imputer = SimpleImputer(strategy="median")
df[["precipitation","windspeed_10m"]] = median_imputer.fit_transform(
    df[["precipitation","windspeed_10m"]])
print("  ✅ precipitation & windspeed_10m         → imputed with MEDIAN")
print(f"\n  Missing values remaining: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────────────────────
# SECTION 6 — HANDLE OUTLIERS (IQR method)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 6: Detecting & Capping Outliers (IQR Method)")
print("=" * 60)

def cap_outliers(dataframe, column):
    Q1, Q3 = dataframe[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    n_out = ((dataframe[column] < lower)|(dataframe[column] > upper)).sum()
    dataframe[column] = dataframe[column].clip(lower=lower, upper=upper)
    print(f"  {column:<30} → {n_out:>3} outliers capped [{lower:.2f}, {upper:.2f}]")
    return dataframe

for col in ["temperature_2m","relative_humidity_2m",
            "precipitation","windspeed_10m"]:
    df = cap_outliers(df, col)

# ─────────────────────────────────────────────────────────────
# SECTION 7 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 7: Feature Engineering")
print("=" * 60)

df["time"]         = pd.to_datetime(df["time"])
df["hour"]         = df["time"].dt.hour
df["day_of_week"]  = df["time"].dt.day_name()
df["date"]         = df["time"].dt.date
df["is_daytime"]   = df["hour"].between(6, 20).astype(int)
df["feels_like_temp"] = (
    df["temperature_2m"] -
    0.4*(df["temperature_2m"]-10)*(1-df["relative_humidity_2m"]/100)
).round(2)
weather_map = {0:"Clear",1:"Mainly Clear",2:"Partly Cloudy",
               3:"Overcast",45:"Foggy",61:"Light Rain",80:"Showers"}
df["weather_label"] = df["weathercode"].map(weather_map).fillna("Unknown")
print("  ✅ hour, day_of_week, date, is_daytime added")
print("  ✅ feels_like_temp calculated")
print("  ✅ weather_label mapped from weathercode")

# ─────────────────────────────────────────────────────────────
# SECTION 8 — ENCODE CATEGORICAL VARIABLES
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 8: Encoding Categorical Variables")
print("=" * 60)

le = LabelEncoder()
df["day_of_week_encoded"] = le.fit_transform(df["day_of_week"])
print(f"  ✅ day_of_week label encoded. Classes: {list(le.classes_)}")
df = pd.get_dummies(df, columns=["weather_label"], prefix="weather")
print("  ✅ weather_label one-hot encoded → weather_* columns added")

# ─────────────────────────────────────────────────────────────
# SECTION 9 — SCALE NUMERIC FEATURES
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 9: Scaling Numeric Features")
print("=" * 60)

scale_cols = ["temperature_2m","relative_humidity_2m",
              "precipitation","windspeed_10m","feels_like_temp"]

df_std = pd.DataFrame(
    StandardScaler().fit_transform(df[scale_cols]),
    columns=[f"{c}_zscore" for c in scale_cols])

df_mm = pd.DataFrame(
    MinMaxScaler().fit_transform(df[scale_cols]),
    columns=[f"{c}_minmax" for c in scale_cols])

df = pd.concat([df, df_std, df_mm], axis=1)
print("  ✅ Z-score (StandardScaler) → *_zscore columns added")
print("  ✅ Min-Max (MinMaxScaler)   → *_minmax columns added")

# ─────────────────────────────────────────────────────────────
# SECTION 10 — EXPORT CLEANED DATA
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 10: Exporting Cleaned Dataset")
print("=" * 60)

df.drop(columns=["time","date","day_of_week"], errors="ignore")\
  .to_csv("lodi_nj_weather_cleaned.csv", index=False)
print("  ✅ Cleaned data saved → lodi_nj_weather_cleaned.csv")
print(f"  📐 Final shape: {df.shape[0]} rows x {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ✅  PIPELINE COMPLETE — SUMMARY")
print("=" * 60)
for k, v in {
    "Total Rows"              : df.shape[0],
    "Total Columns"           : df.shape[1],
    "Missing Values Remaining": int(df.isnull().sum().sum()),
    "Features Engineered"     : 5,
    "Scalers Used"            : 2,
    "Output Files"            : "cleaned CSV + PNG chart",
}.items():
    print(f"  {k:<30}: {v}")
print("=" * 60)
