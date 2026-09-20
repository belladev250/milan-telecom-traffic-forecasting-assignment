
## Hardware Requirements

The notebook is designed to run on **Google Colab with T4 GPU**:

| Component | Specification |
| :--- | :--- |
| Platform | Google Colaboratory |
| GPU | NVIDIA T4 (16 GB VRAM) |
| RAM | ~12.7 GB available |
| Storage | ~20 GB free |

The code will also run on CPU, but training will be significantly slower (5-10x).

## Running the Notebook

### Step 1: Get the Dataset

1. Go to the Harvard Dataverse link above
2. Download all 52 files named `sms-call-internet-mi-2013-11-01.txt` through `sms-call-internet-mi-2013-12-22.txt`
3. Upload these files to your Google Drive in a folder called `milan_telecom_data/`

### Step 2: Open the Notebook in Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click `File` → `Upload notebook`
3. Select `milan_telecom_analysis.ipynb` from your computer

### Step 3: Configure the Runtime

1. Click `Runtime` → `Change runtime type`
2. Set **Hardware accelerator** to `T4 GPU`
3. Click `Save`

### Step 4: Run the Notebook

1. Run the first cell to mount your Google Drive
2. Update the folder path if your data is in a different location
3. Click `Runtime` → `Run all`

**Expected runtime:** ~5-7 minutes on first run (includes data loading and model training). Subsequent runs will be faster because the processed data is cached as a Parquet file in your Drive.

## What the Notebook Does

The notebook is organised into six main sections:

| Section | Description |
| :--- | :--- |
| **1. Setup & Data Loading** | Mounts Drive, loads 52 files with memory optimisation, saves to Parquet |
| **2. Exploratory Data Analysis** | PDF plots, time series, stationarity tests, decomposition, ACF/PACF, heatmaps, anomaly detection |
| **3. SARIMA Model** | Hourly aggregation, model fitting, forecasting on test week (Dec 16-22) |
| **4. LSTM Model** | MinMax scaling, sliding windows (144 steps), 2-layer LSTM training |
| **5. TCN Model** | Same preprocessing, 4-block TCN with dilations [1,2,4,8] |
| **6. Comparative Analysis** | Side-by-side metrics table, failure analysis, discussion |

## Memory Optimisation

The following strategies were used to handle the 16 GB dataset within Colab's memory limits:

| Optimisation | Impact |
| :--- | :--- |
| Column selection (3 of 8 columns) | 60% less data loaded into memory |
| Downcasting int64→int32, float64→float32 | 33% memory reduction per file |
| Chunk-by-chunk loading | Prevents memory spikes during concatenation |
| Parquet serialisation | 20× storage reduction (16 GB → 781 MB) |

**Result:** 273 million rows loaded in 224 seconds using ~5.5 GB of RAM.

## Key Results

### Area 5161 (Highest Traffic Cell)

| Model | MAE | MAPE | RMSE | Train Time |
| :--- | :--- | :--- | :--- | :--- |
| SARIMA | 324.42 | 29.09% | 487.91 | 10.4 sec |
| LSTM | 126.19 | 12.86% | 186.22 | 6.7 sec |
| TCN | 125.94 | 15.05% | 189.09 | 14.2 sec |

### Area 4159 (Mid-Range)

| Model | MAE | MAPE | RMSE | Train Time |
| :--- | :--- | :--- | :--- | :--- |
| SARIMA | 76.97 | 42.19% | 121.65 | 7.2 sec |
| LSTM | 18.89 | 8.85% | 24.26 | 5.8 sec |
| TCN | 19.26 | 8.00% | 26.49 | 13.9 sec |

### Area 4556 (Secondary)

| Model | MAE | MAPE | RMSE | Train Time |
| :--- | :--- | :--- | :--- | :--- |
| SARIMA | 119.61 | 27.96% | 150.58 | 9.8 sec |
| LSTM | 33.23 | 8.15% | 42.51 | 5.7 sec |
| TCN | 31.73 | 7.35% | 41.86 | 13.8 sec |

## Key Findings

### Model Comparison
- LSTM and TCN both significantly outperform SARIMA, reducing MAPE by 60-70%
- LSTM is recommended overall: best accuracy on the highest-traffic area, fastest training (~6 sec)
- TCN slightly better on lower-traffic areas but trains ~2x slower
- SARIMA remains useful as a fast, interpretable baseline

### Data Characteristics
- Strong daily and weekly seasonality dominates the signal
- Traffic follows a log-normal-like distribution across the city
- All three focus areas are statistically stationary (confirmed by ADF tests, p < 0.05)
- Traffic is heavily concentrated in the city centre, decreasing radially

### Failure Analysis
All models struggled during December 19-22 (the week before Christmas). This is a distributional shift problem - the test period had unusually high traffic due to holiday shopping and events that didn't appear in the training data. Adding calendar features (holiday indicators) would likely fix this.

## Dependencies

No manual installation is required if running on Google Colab. All libraries come pre-installed:

- Python 3.11
- pandas 2.x
- numpy 1.x
- matplotlib 3.x
- seaborn 0.12.x
- statsmodels 0.14.x
- scikit-learn 1.3.x
- torch 2.x (PyTorch)
- tqdm

To verify everything is available, run the first code cell in the notebook which checks for all required libraries.

## Video Demonstration

A complete walkthrough of this project is available at:

****https://www.youtube.com/watch?v=cUe89h9-MYU

The video covers:
- Data loading and memory optimisation strategy
- Key findings from exploratory data analysis
- Model architectures for SARIMA, LSTM, and TCN
- Training procedures and hyperparameter tuning
- Results comparison and interpretation
- Failure analysis (why Dec 19-22 was problematic)
- Demonstration of running the notebook in Colab

## Results Reproduced From This Repository

To reproduce all results:

1. Open the notebook in Colab with T4 GPU
2. Upload the raw data to Google Drive
3. Run all cells (`Runtime` → `Run all`)

The notebook will:
- Load and process the data (takes ~4 minutes first time)
- Generate all figures from Task 2
- Train all three models
- Produce the metrics tables shown above
- Save figures to your Drive (optional)

## Known Limitations

1. **SARIMA requires hourly aggregation** - Full 10-minute resolution with seasonal period s=144 is computationally infeasible due to seasonal lag matrix size. The hourly approximation produces step-shaped forecasts.

2. **Pre-Christmas distributional shift** - All models performed poorly on Dec 19-22. The training period (Nov 1 - Dec 15) doesn't contain holiday patterns.

3. **Univariate only** - Models only use the target cell's historical traffic. Adding traffic from neighbouring cells (spatial features) could improve accuracy.

4. **No calendar features** - The models don't know about holidays or day-of-week. Adding these as explicit inputs would help with the Christmas period failure.

## Possible Improvements

- Add day-of-week and hour-of-day embeddings
- Include public holiday indicators as features
- Use multivariate inputs with neighbouring grid cells
- Implement attention mechanism or Transformer architecture
- Train for more epochs (50-100) with early stopping
- Bayesian hyperparameter optimisation

## References

1. Barlacchi, G., et al. (2015). "A multi-source dataset of urban life in the city of Milan and the Province of Trentino." *Scientific Data*, 2, 150055.

2. Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory." *Neural Computation*, 9(8), 1735-1780.

3. Bai, S., Kolter, J. Z., & Koltun, V. (2018). "An empirical evaluation of generic convolutional and recurrent networks for sequence modeling." *arXiv:1803.01271*.

## Contact

**Author:** Bella Melissa Ineza


---

*Submitted as part of ML Techniques I formative assignment.*
