# Machine Learning & Forecasting Coursework

**溫翎傑｜NTUST 工業管理所｜課堂練習與作業（2024–2025）**

這個 repo 收的是課堂上的練習與作業：從頭實作神經網路、分群和基因演算法，以及股價時間序列預測。比較完整的專案另外放在獨立 repo：[Quality2025](https://github.com/Jerry25346/Quality2025)、[Soft-Computing-HW3](https://github.com/Jerry25346/Soft-Computing-HW3)、[soft-computing-assignment-2](https://github.com/Jerry25346/soft-computing-assignment-2)。

---

## Highlight — predicting student GPA

`Interactive - GPA.py.ipynb`

Predicts GPA for 2,392 students in a student-performance dataset (13 input columns), with a 64/16/20 train/validation/test split. Scaling and encoding are fitted on the training set only, so no test information leaks into preprocessing.

| Model | Test RMSE | Test R² |
|---|---:|---:|
| Random forest | 0.252 | **0.923** |
| SVR | 0.260 | 0.918 |
| Neural network (NumPy, tuned hidden size and learning rate) | 0.264 | 0.916 |
| Decision tree | 0.317 | 0.879 |
| KNN | 0.450 | 0.755 |

The notebook also uses SHAP to show which features drive the predictions.

## Neural networks from scratch (NumPy)

| File | What it does |
|---|---|
| `Interactive - perceptron.py.ipynb` | Perceptron with decision-boundary plots |
| `Interactive - back.XOR.py.ipynb` | Two-layer network trained with backpropagation on XOR (MSE loss, Xavier initialization) |
| `Interactive - back.XOR _bce.py.ipynb` | Same network with binary cross-entropy loss |
| `Interactive - back_iris.py.ipynb` | Backpropagation network for 3-class Iris classification (one-hot targets) |
| `Readme.ipynb` | Notes on the Gaussian-noise layer used in the denoising autoencoder of Quality2025 |

## Clustering

| File | What it does |
|---|---|
| `k_means.py`, `Interactive - k_means.py.ipynb` | K-means implemented from scratch on the Iris data |
| `kkkk-meams.py` | Elbow method with scikit-learn's KMeans to choose the number of clusters |
| `Iris.csv` | Data used by both |

## Genetic algorithm

| File | What it does |
|---|---|
| `#優化最多1.py` | Binary GA (roulette selection, crossover, mutation, elitism) on the OneMax problem |
| `#gemnetric algorithm.py` | Outline of the GA steps |

## Time-series forecasting (Production Management course, R / Quarto)

| File | What it does |
|---|---|
| `生產管理功課.qmd` | NVIDIA closing price: descriptive statistics, additive decomposition (weekly cycle of 5 trading days) and forecasting |
| `associative_model.pdf`, `tsmcccccccc.pdf` | Group report: ARIMAX model linking NVIDIA monthly log returns to the SOXX semiconductor index |

## Imbalanced data

| File | What it does |
|---|---|
| `renderrrrrr.qmd`, `renderrrrrr.pdf` | Notes and code on SMOTE, SMOTE + Tomek links and SMOTE + ENN for class imbalance |

## Statistics

| File | What it does |
|---|---|
| `myreport (1).pdf` | Statistical-thinking report with summary statistics and missing-value analysis (Palmer Penguins data, R) |

## Tools

Python (NumPy, pandas, scikit-learn, SHAP, matplotlib, seaborn, plotly) · R (forecast, tseries, dplyr, ggplot2) · Quarto
