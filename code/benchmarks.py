"""Additional benchmarks, bootstrap intervals and error-correlation statistics for the
condition-aware maize ensemble (Egorov & Helman, Perspective; Section 5.2, Table S2 notes,
Text S8, Text S10 and Table S6 of the Supporting Information).

Run AFTER code/ca_pipeline.py (needs results/panel.csv, results/lgb_params.json and
results/predictions_*.csv):   python code/benchmarks.py      (about 10 min)

Outputs (results/):
  TableS2_benchmarks.csv  RMSE of fixed optimal weights (non-negative least squares, fitted on the
                          training years), of a LightGBM yield model without crop models (ML-only) and
                          of a stacked LightGBM model (24 predictors + 19 simulated yields), for the
                          baseline test and the two drought tests; plus the NNLS weights.
  bootstrap_CI.csv        95% spatial-block-bootstrap intervals (2.5 deg blocks, 1,000 draws) for the
                          change in RMSE of the condition-aware (LGB) ensemble relative to the MME mean.
  TableS6.csv             Mean pairwise correlation of the errors within and between the EPIC, GEPIC
                          and other simulations over the training years, and the effective number of
                          independent members N_eff (Pennell & Reichler, 2011).
"""
import os, sys, json
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.optimize import nnls
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ca_pipeline import M19, WX, SOIL, Y0, Y1, TRAIN_END, TEST0, DROUGHT, prep, rmse, OUT

D = pd.read_csv(os.path.join(OUT, 'panel.csv'))
params = json.load(open(os.path.join(OUT, 'lgb_params.json')))
yrs = list(range(Y0, Y1 + 1))
FEAT = WX + SOIL + ['lat', 'lon']
mk = lambda: lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=0, **params)

TESTS = {'baseline': ([y for y in yrs if y <= TRAIN_END], [y for y in yrs if y >= TEST0])}
for d in DROUGHT:
    TESTS[f'd{d}'] = ([y for y in yrs if y != d], [d])

# ------------------------------------------------ 1. fixed weights, ML-only, stacking
rows, wrows = [], []
for tag, (train, test) in TESTS.items():
    P = prep(D, train); TR = P[P.year.isin(train)]; TE = P[P.year.isin(test)].reset_index(drop=True)
    obs = TE.obs_d.values
    w, _ = nnls(TR[M19].values, TR.obs_d.values); w = w / w.sum()
    r_nnls = rmse(TE[M19].values @ w, obs)
    r_ml = rmse(mk().fit(TR[FEAT].values, TR.obs_d.values).predict(TE[FEAT].values), obs)
    r_stack = rmse(mk().fit(TR[FEAT + M19].values, TR.obs_d.values).predict(TE[FEAT + M19].values), obs)
    r_mean = rmse(TE[M19].values.mean(1), obs)
    for name, r in [('Fixed optimal weights (NNLS)', r_nnls), ('ML, no crop models', r_ml),
                    ('Stacking (24 predictors + 19 simulations)', r_stack)]:
        rows.append({'test': tag, 'strategy': name, 'RMSE': round(r, 4), 'vs_mean_%': round(100 * (r / r_mean - 1), 2)})
    wrows.append(pd.Series(np.round(w, 4), index=M19, name=tag))
    print(tag, 'MME mean %.3f | NNLS %.3f | ML-only %.3f | stacking %.3f' % (r_mean, r_nnls, r_ml, r_stack), flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'TableS2_benchmarks.csv'), index=False)
pd.concat(wrows, axis=1).to_csv(os.path.join(OUT, 'NNLS_weights.csv'))

# ------------------------------------------------ 2. spatial block bootstrap
ci = []
for tag in TESTS:
    pr = pd.read_csv(os.path.join(OUT, f'predictions_{tag}.csv')); rng = np.random.RandomState(0)
    pr['blk'] = (pr.lat // 2.5).astype(int).astype(str) + '_' + (pr.lon // 2.5).astype(int).astype(str)
    blks = list(pr.blk.unique()); groups = {k: pr[pr.blk == k] for k in blks}
    g = []
    for b in range(1000):
        sub = pd.concat([groups[k] for k in rng.choice(blks, len(blks), replace=True)])
        g.append(100 * (rmse(sub['Condition-aware (LGB)'], sub.obs_d) / rmse(sub['MME mean'], sub.obs_d) - 1))
    ci.append({'test': tag, 'mean_change_%': round(np.mean(g), 1), 'CI2.5_%': round(np.percentile(g, 2.5), 1),
               'CI97.5_%': round(np.percentile(g, 97.5), 1), 'blocks': len(blks), 'draws': 1000})
    print(tag, 'CA(LGB) vs MME mean: %.1f%% [%.1f, %.1f]' % (np.mean(g), np.percentile(g, 2.5), np.percentile(g, 97.5)), flush=True)
pd.DataFrame(ci).to_csv(os.path.join(OUT, 'bootstrap_CI.csv'), index=False)

# ------------------------------------------------ 3. error correlations and N_eff (Table S6)
train = TESTS['baseline'][0]; P = prep(D, train); TR = P[P.year.isin(train)]
E = TR[M19].values - TR.obs_d.values[:, None]; R = np.corrcoef(E.T)
grp = {'EPIC (7)': [m for m in M19 if m.startswith('epic')], 'GEPIC (5)': [m for m in M19 if m.startswith('gepic')],
       'Others (7)': [m for m in M19 if not m.startswith(('epic', 'gepic'))]}
idx = {g: [M19.index(m) for m in ms] for g, ms in grp.items()}
neff = lambda C: float((np.linalg.eigvalsh(C).sum() ** 2) / (np.linalg.eigvalsh(C) ** 2).sum())
tab = pd.DataFrame(index=list(grp), columns=list(grp), dtype=float)
for g1 in grp:
    for g2 in grp:
        a = R[np.ix_(idx[g1], idx[g2])]
        tab.loc[g1, g2] = a[~np.eye(len(idx[g1]), dtype=bool)].mean() if g1 == g2 else a.mean()
tab = tab.round(2)
tab.loc['N_eff (all 19)', tab.columns[0]] = round(neff(R), 1)
tab.loc['N_eff (7 others)', tab.columns[0]] = round(neff(R[np.ix_(idx['Others (7)'], idx['Others (7)'])]), 1)
tab.to_csv(os.path.join(OUT, 'TableS6.csv'))
pd.DataFrame(R, index=M19, columns=M19).round(3).to_csv(os.path.join(OUT, 'error_correlation_matrix.csv'))
print(tab)
