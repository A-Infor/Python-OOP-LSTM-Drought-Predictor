# 🌱 Python OOP LSTM Drought Predictor

> *A clean, object-oriented Long Short-Term Memory (LSTM) neural network that forecasts the **SPEI** (Standardized Precipitation–Evapotranspiration Index) — one of the most widely used indicators of meteorological drought — for municipalities in northern Minas Gerais, Brazil.*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](#-quick-start)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](#-tech-stack)
[![License](https://img.shields.io/badge/License-Unspecified-lightgrey)](#-license)
[![Minas Gerais](https://img.shields.io/badge/Region-Northern%20Minas%20Gerais%20(BR)-green)](#-the-region)

---

## 🌍 What is this project?

A split of [**JVSREco19/RedeNeuralSecas**](https://github.com/JVSREco19/RedeNeuralSecas) that seeks to be more universally applicable.

Droughts are among the most destructive natural hazards on the planet. In Brazil, the semi-arid north of **Minas Gerais** is no exception — communities, agriculture and water reservoirs are constantly under pressure from prolonged dry spells. This repository implements a **deep-learning pipeline** that, given a historical monthly SPEI time series, learns its temporal patterns and forecasts future drought conditions.

What makes it special is **not** only the model — it is the **clean, modular Object-Oriented design** that splits responsibilities into self-contained classes (`Dataset`, `Cluster`, `NeuralNetwork`, `PerformanceEvaluator`, `Plotter`). The result is a project that is easy to read, easy to extend, and pleasant to experiment with.

> 🎓 Originally built to support a master's dissertation, the codebase serves as a research-friendly reference for anyone interested in drought forecasting, time-series deep learning, or OOP best practices in scientific Python.

---

## 🗺️ The Region

The study focuses on **two neighbouring clusters** of municipalities in northern Minas Gerais, each containing three cities:

| Cluster                 | Municipalities (training & target) |
|-------------------------|-------------------------------------|
| **Rio Pardo de Minas**  | Rio Pardo de Minas · Montezuma · Fruta de Leite |
| **São Francisco**       | São Francisco · Pintópolis · Japonvar |

The model is trained on the **central** city of each cluster and then **transferred** to its two **bordering** neighbours. This produces a *cross-municipality* evaluation that measures how well drought knowledge generalises across geographically related regions.

---

## ⚡ How the model works (in a nutshell)

1. **Ingest** a historical monthly SPEI series from an `.xlsx` spreadsheet (`Series 1` → `SPEI Real`).
2. **Cluster-normalise** every city's SPEI using the global min/max of its cluster, so that all cities share the same activation-friendly scale.
3. **Window the data** with a sliding-window technique: 12 consecutive months are sliced, the last 6 are the prediction target, the first 6 are the input.
4. **Train** a small LSTM (9 hidden units) followed by 3 fully-connected Dense layers (6 units, sigmoid), for 800 epochs, on 80 % of the time series.
5. **Predict** drought values on the held-out 20 % *and* on the entire series of the two neighbouring cities.
6. **Evaluate** with MAE, MSE, RMSE and R², log everything, and **visualise** predictions, residuals, R² scatter plots and the prediction distribution.

The result: a forecast that learns long-term temporal dependencies in drought behaviour and that you can apply, unchanged, to a neighbour's data.

---

## 🏗️ Architecture

```
Python-OOP-LSTM-Drought-Predictor/
│
├── main.py                          # 🚪 Entry point – orchestrates every experiment
├── config.json                      # ⚙️  Model hyperparameters
│
├── Data/
│   ├── cluster RIO PARDO DE MINAS/  # 📊 3 .xlsx time-series
│   └── cluster SÃO FRANCISCO/       # 📊 3 .xlsx time-series
│
├── classes/
│   ├── __init__.py                  # 🔌 Re-exports the public API
│   ├── dataset.py                   # 📁  Dataset    – I/O, splitting, sliding windows
│   ├── cluster.py                   # 🗂️  Cluster    – groups cities, normalises SPEI
│   ├── neural_network.py            # 🧠 NeuralNetwork – LSTM + Dense model, training loop
│   ├── performance_evaluator.py     # 📏 PerformanceEvaluator – MAE / MSE / RMSE / R²
│   └── plotter.py                   # 🎨 Plotter    – all matplotlib visualisations
│
└── Images/                          # 🖼️  Auto-generated at runtime
```

### Class responsibilities

| Class                  | Role                                                                                          |
|------------------------|-----------------------------------------------------------------------------------------------|
| `Dataset`              | Loads an `.xlsx`, holds the raw SPEI array, splits train/test, builds sliding-window pairs.   |
| `Cluster`              | Groups several `Dataset`s, computes cluster-wide min/max, applies min-max normalisation.       |
| `NeuralNetwork`        | Builds the Keras `Sequential` model, trains it, runs inference on central/bordering cities.   |
| `PerformanceEvaluator` | Computes MAE / MSE / RMSE / R² for both central and bordering scenarios.                      |
| `Plotter`              | Generates dataset plots, training curves, residual plots, R² scatter, prediction distribution. |

---

## ⚙️ Configuration

All hyperparameters live in **`config.json`** — no code change required to retune the model:

```json
{
  "total_points"    : 12,
  "dense_units"     : 6,
  "hidden_units"    : 9,
  "numberOfEpochs"  : 800,
  "parcelDataTrain" : 0.8
}
```

| Key                | Meaning                                                                  |
|--------------------|--------------------------------------------------------------------------|
| `total_points`     | Sliding window length (months).                                          |
| `dense_units`      | Number of months to predict *and* the size of each Dense layer.         |
| `hidden_units`     | LSTM hidden-state size.                                                  |
| `numberOfEpochs`   | Training epochs.                                                         |
| `parcelDataTrain`  | Fraction of the series used for training (the rest is the test split).  |

The model is created with `ReLU` activations on the LSTM and `sigmoid` on the Dense layers, `mse` loss and the `adam` optimiser, and is compiled with `mae`, `rmse`, `mse` and `r2` as metrics.

---

## 🚀 Quick start

### 1. Clone

```bash
git clone https://github.com/A-Infor/Python-OOP-LSTM-Drought-Predictor.git
cd Python-OOP-LSTM-Drought-Predictor
```

### 2. Install dependencies

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib openpyxl
```

> **Tip:** a dedicated `requirements.txt` is on the roadmap. For now the imports you'll find inside `main.py` and `classes/*.py` are the source of truth.

### 3. Run

```bash
python main.py
```

`main.py` is self-contained: it builds both clusters, trains an LSTM per central city, applies it to its two neighbours, and prints a complete metrics summary. Plots are written to `Images/cluster <CLUSTER>/model <CITY>/…` automatically.

---

## 📈 What you'll see

* **Dataset plots** — observed SPEI series with the train/test split highlighted.
* **Training history** — loss / metrics curves for each city.
* **Residual plots** — predicted vs. error per portion (80 % / 20 % / 100 %).
* **R² scatter plots** — observed vs. predicted, by portion.
* **Prediction-distribution plots** — how close predicted values are to the real distribution.
* **Console logs** — per-city metrics for central and bordering runs.

---

## 🧪 Reproducing an experiment in 4 lines

```python
from classes import Dataset, Plotter, NeuralNetwork

dataset = Dataset("Rio Pardo de Minas", "Rio Pardo de Minas")
plotter = Plotter(dataset)
model   = NeuralNetwork("config.json", dataset, plotter)
model.use_neural_network()
```

To apply the trained model to a neighbouring city:

```python
bordering = Dataset("Montezuma", "Rio Pardo de Minas")
model.use_neural_network(dataset=bordering, plotter=Plotter(bordering))
```

---

## 🛠️ Tech stack

* **Python 3.9+** – the whole pipeline.
* **TensorFlow / Keras** – the LSTM model.
* **pandas + NumPy** – data wrangling and sliding windows.
* **scikit-learn** – chronological train/test split.
* **matplotlib** – all visualisations.
* **openpyxl** – reading the `.xlsx` time series.

---

## 🤝 Contributing

Issues, suggestions and pull requests are warmly welcome.

Feel free to open an issue first to discuss significant changes.

---

## 📚 Citation & context

If you use this project in academic work, please reference the original dissertation it was built for. 

---

## 📄 License

No license has been declared in this repository yet. Until one is added, the source code is **all rights reserved** by the author under default copyright law. If you'd like to reuse, redistribute, or build on this work, please contact the repository owner first — or, even better, open an issue/PR adding a `LICENSE` file (common choices are MIT, Apache-2.0, BSD-3-Clause, or GPL-3.0).
