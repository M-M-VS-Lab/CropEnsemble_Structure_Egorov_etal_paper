# Crop model ensembles should account for shared model structure

Code and derived data for

> Egorov, F., & Helman, D. (2026). Crop model ensembles should account for shared model structure. *Global Change Biology* (submitted).

Corresponding author: David Helman (david.helman@mail.huji.ac.il), The Hebrew University of Jerusalem.

Everything in the paper that is computed (Figs 1–3, Figs S1–S3, Tables S2–S4 and S6) can be regenerated from the scripts in `code/` and the small derived data files in `data/`. The raw data are public (see *Data sources*); the scripts that extract the derived files from them are included.

## Contents

```
code/
  fig1.py                     Fig. 1 (classification heat map) from data/TableS1_scores.csv
  fig2_hsc_ensemble.py        Fig. 2 helper: loads the AgMIP Hot Serial Cereal archive, model classes, code-sharing groups, similarity weights
  fig2_timeseries.py          Fig. 2 (daily biomass, 29 wheat models, winter unheated vs. spring heated sowings)
  figS1.py                    Fig. S1 and Table S3 (three ways of averaging the wheat ensemble)
  extract_gdhy_conus.py       GDHY maize yields -> data/gdhy_maize_conus_1981_2010.csv
  extract_isimip_conus.py     ISIMIP2a/GGCMI maize simulations -> data/isimip_maize_conus_1981_2010.csv.gz
  compute_weather_features.py WATCH-WFDEI daily data -> data/weather_features_conus_1981_2010.csv (growing-season predictors)
  extract_soil_conus.py       ISIMIP3 HWSD soil input -> data/soil_conus.csv
  ca_pipeline.py              Maize analysis: panel, detrending, bias correction, error regressors, condition-aware weights, benchmarks, tests
  make_figures.py             Fig. 3, Fig. S2, Fig. S3 from results/
  benchmarks.py               Fixed optimal weights (NNLS), ML-only and stacking benchmarks, spatial-block bootstrap intervals, error correlations and N_eff (Table S2 notes, Text S10, Table S6)
data/
  TableS1_scores.csv                    scores of the 11 processes for the 37 classified models (Table S1; input of fig1.py)
  gdhy_maize_conus_1981_2010.csv        observed rainfed maize yield (t/ha) per 0.5-deg cell and year (GDHY v1.2/v1.3)
  isimip_maize_conus_1981_2010.csv.gz   simulated maize yield per model, harmonization run, cell and year (ISIMIP2a)
  weather_features_conus_1981_2010.csv  1 May-30 Sep weather predictors per cell and year (WATCH-WFDEI)
  soil_conus.csv                        HWSD topsoil predictors per cell
  us_cells.csv                          cells whose centre lies in the contiguous United States
results/
  panel.csv                   the analysis panel (1,522 cells x 27 years, 19 models, observations, predictors)
  table_baseline.csv, table_d1983.csv, table_d1988.csv   RMSE of every strategy in each test
  single_models_*.csv         RMSE of each of the 19 models in each test
  predictions_*.csv           predictions of every strategy per cell and year
  weights_lgb_*.csv           condition-aware (LightGBM) weights per cell and year
  TableS2.csv, TableS4.csv, TableS3.csv   the supplementary tables
  TableS2_benchmarks.csv, NNLS_weights.csv, bootstrap_CI.csv   benchmark rows and bootstrap intervals quoted in Table S2 and Section 5.2
  TableS6.csv, error_correlation_matrix.csv   error correlations among the 19 simulations and the effective ensemble size
  lgb_params.json             LightGBM hyperparameters chosen by cross-validation
figures/                      Fig1, Fig2, Fig3, FigS1, FigS2, FigS3 (PNG)
```

## Method in brief (details in the Supporting Information, Texts S2–S10)

* Domain: rainfed maize, contiguous United States east of 104.5° W, 0.5° cells with observations and all 19 simulations in every year, at least two neighbouring cells, 1982–2008 (1,522 cells).
* Observations are detrended per cell with a linear trend fitted on training years only; each model is bias-corrected per cell with its mean training-year bias (max[0, ·]).
* For each of the 19 models a regressor (LightGBM; also XGBoost and random forest) predicts the model's absolute error from 25 predictors (14 weather, 8 soil, latitude, longitude, the model's own bias-corrected yield).
* Weights: softmax of the negative predicted errors, w_k = exp(−τ Ê_k) / Σ_j exp(−τ Ê_j); τ chosen on a validation set drawn from the training years.
* Tests: baseline (train 1982–2004, test 2005–2008) and two leave-one-drought-year-out tests (1983 withheld with 1988 in training; 1988 withheld with 1983 in training).
* Benchmarks: unweighted mean and median, single best model (hindsight), best model per cell (training years), group average, similarity weighting (Knutti et al., 2017), fixed optimal weights (non-negative least squares on the training years), a LightGBM yield model without crop models, and a stacked LightGBM model; 95% spatial-block-bootstrap intervals (2.5° blocks, 1,000 draws).

## Reproducing the results

Python 3.10+ with `numpy pandas scipy scikit-learn lightgbm xgboost netCDF4 matplotlib basemap basemap-data` (`pip install -r requirements.txt`).

Maize analysis from the derived data in `data/` (no download needed):

```
python code/ca_pipeline.py      # writes results/   (about 2–3 h on 2 cores; random forest is the slow part)
python code/make_figures.py     # writes figures/Fig3.png, FigS2.png, FigS3.png (+ .tif)
python code/benchmarks.py       # writes results/TableS2_benchmarks.csv, bootstrap_CI.csv, TableS6.csv (about 10 min)
python code/fig1.py             # writes figures/Fig1.png from data/TableS1_scores.csv
```

Wheat figures from the AgMIP Hot Serial Cereal archive (download `HSC-AgMIP_data` from the Dataverse record below and unpack it):

```
python code/fig2_timeseries.py  path/to/HSC-AgMIP_data     # Fig. 2
python code/figS1.py            path/to/HSC-AgMIP_data     # Fig. S1 and Table S3
```

Rebuilding the derived maize data from the raw sources (optional):

```
python code/extract_gdhy_conus.py   path/to/gdhy_v1.2_v1.3_20190128/maize   data/gdhy_maize_conus_1981_2010.csv
python code/extract_isimip_conus.py path/to/isimip2a_yield-mai-noirr_nc4    data/gdhy_maize_conus_1981_2010.csv  data/isimip_maize_conus_1981_2010.csv
python code/compute_weather_features.py   # expects the WATCH-WFDEI CONUS cut-outs (tas, tasmax, tasmin, pr, rsds, huss, ps; 1981–2010) in data/wfdei/
python code/extract_soil_conus.py         # expects data/soil/hwsd_soil_data_all_land.nc
```

## Data sources

* ISIMIP2a/GGCMI Phase 1 maize simulations (yield-mai-noirr, WATCH-WFDEI, nobc, hist CO2; default, fullharm and harmnon runs): Müller et al. (2023), https://doi.org/10.48364/ISIMIP.729341
* WATCH-WFDEI daily climate: Lange & Büchner (2020), https://doi.org/10.48364/ISIMIP.886955
* HWSD v1.2 soil (ISIMIP3 soil input): Volkholz & Müller (2020), https://doi.org/10.48364/ISIMIP.942125
* GDHY v1.2 and v1.3 observed yields: Iizumi (2019), https://doi.org/10.1594/PANGAEA.909132
* AgMIP Hot Serial Cereal wheat observations and simulations: Martre et al. (2018), https://doi.org/10.7910/DVN/ECSFZG

## License

Code: MIT License (see `LICENSE`). Derived data files are redistributed under the terms of their sources (CC BY 4.0).
