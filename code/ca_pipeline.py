"""Condition-aware weighting of 19 GGCMI/ISIMIP2a rainfed-maize simulations over the US
(Egorov & Helman, Perspective). Rebuilds Fig. 3, Figs S2-S3, Tables S2 and S4 from raw data.

Run from anywhere: python code/ca_pipeline.py  (about 2-3 h on 2 cores; RF is the slow part)
Inputs (data/):  isimip_maize_conus_1981_2010.csv.gz  (model,harm,year,lat,lon,sim)
                 gdhy_maize_conus_1981_2010.csv        (year,lat,lon,obs_yield)
                 weather_features_conus_1981_2010.csv  (growing-season predictors from WATCH-WFDEI)
                 soil_conus.csv                        (HWSD topsoil predictors)
                 us_cells.csv                          (0.5-deg cells whose centre lies in the contiguous US)
Outputs (results/): panel.csv, predictions_*.csv, weights_*.csv, tableS2.csv, tableS4.csv, cell_rmse_*.csv
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore')
rng = np.random.RandomState(0)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE) if os.path.basename(HERE) == 'code' else HERE   # repository root
DATA = os.path.join(BASE, 'data')
OUT = os.path.join(BASE, 'results'); os.makedirs(OUT, exist_ok=True)

M19 = ['epic-boku', 'epic-boku-p', 'epic-boku-br', 'epic-boku-pt', 'epic-boku-hg', 'epic-iiasa', 'epic-tamu',
       'gepic', 'gepic-pm', 'gepic-br', 'gepic-pt', 'gepic-p', 'pdssat',
       'cgms-wofost', 'pegasus', 'lpjml', 'lpj-guess', 'orchidee-crop', 'clm-crop']
WX = ['tmean', 'tmax', 'tmin', 'dtr', 'gdd', 'edd', 'hot_days', 'pr_sum', 'pr_anom', 'pr_julaug', 'dry_spell',
      'rsds', 'huss', 'vpd']
SOIL = ['sand', 'silt', 'clay', 'oc', 'ph', 'bulk_density', 'awc', 'cec']
Y0, Y1 = 1982, 2008          # years in which observations and all 19 simulations exist
TRAIN_END, TEST0 = 2004, 2005
DROUGHT = [1983, 1988]
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

# ---------------------------------------------------------------- 1. panel
def build_panel():
    s = pd.read_csv(os.path.join(DATA, 'isimip_maize_conus_1981_2010.csv.gz'))
    o = pd.read_csv(os.path.join(DATA, 'gdhy_maize_conus_1981_2010.csv'))
    rows = []
    for m in M19:
        g = s[s.model == m]; hs = set(g.harm)
        h = 'fullharm' if 'fullharm' in hs else ('default' if 'default' in hs else sorted(hs)[0])
        rows.append(g[g.harm == h].assign(model=m))
    S = pd.concat(rows).drop(columns='harm')
    W = S.pivot_table(index=['lat', 'lon', 'year'], columns='model', values='sim')
    D = W.join(o.set_index(['lat', 'lon', 'year']).obs_yield, how='inner').dropna()
    D = D[(D.index.get_level_values('year') >= Y0) & (D.index.get_level_values('year') <= Y1)]
    D = D[D.index.get_level_values('lon') >= -104.5]
    nyr = Y1 - Y0 + 1
    cnt = D.groupby(level=['lat', 'lon']).size(); full = cnt[cnt == nyr].index
    D = D[D.index.droplevel('year').isin(full)].reset_index()
    # neighbour filter: keep cells with >= 2 adjacent (8-neighbourhood) cells of the domain
    cells = D[['lat', 'lon']].drop_duplicates().reset_index(drop=True)
    tree = cKDTree(cells[['lat', 'lon']].values)
    nn = tree.query_ball_point(cells[['lat', 'lon']].values, r=0.5 * np.sqrt(2) + 1e-6)
    keep = cells[[len(x) - 1 >= 2 for x in nn]]
    D = D.merge(keep, on=['lat', 'lon'])
    # keep only cells whose centre lies in the contiguous United States (the GDHY grid also covers Canada/Mexico)
    us = pd.read_csv(os.path.join(DATA, 'us_cells.csv')); D = D.merge(us, on=['lat', 'lon'])
    wx = pd.read_csv(os.path.join(DATA, 'weather_features_conus_1981_2010.csv'))
    so = pd.read_csv(os.path.join(DATA, 'soil_conus.csv'))
    D = D.merge(wx, on=['year', 'lat', 'lon'], how='left').merge(so, on=['lat', 'lon'], how='left')
    assert D[WX[:8] + SOIL].notna().all().all(), 'missing predictors'
    print(f'panel: {D[["lat","lon"]].drop_duplicates().shape[0]} cells x {nyr} years = {len(D)} cell-years')
    return D

# ------------------------------------------------- 2. detrend / bias-correct
def prep(D, train):
    out = []
    for (la, lo), g in D.groupby(['lat', 'lon']):
        g = g.copy(); tr = g[g.year.isin(train)]
        a, b = np.polyfit(tr.year, tr.obs_yield, 1); ybar = tr.obs_yield.mean()
        g['obs_d'] = ybar + (g.obs_yield - (a * g.year + b))
        for m in M19:
            g[m] = np.maximum(0.0, g[m] - (tr[m].mean() - ybar))
        g['pr_anom'] = g.pr_sum - tr.pr_sum.mean()
        out.append(g)
    return pd.concat(out, ignore_index=True)

# --------------------------------------------------------- 3. error models
def fit_predict(P, train, test, algo='lgb', params=None):
    """Return predicted absolute errors (n_test x 19) and actual test frame."""
    TR = P[P.year.isin(train)]; TE = P[P.year.isin(test)]
    E = np.zeros((len(TE), len(M19)))
    for j, m in enumerate(M19):
        X = TR[WX + SOIL + ['lat', 'lon', m]].values; y = np.abs(TR.obs_d - TR[m]).values
        Xt = TE[WX + SOIL + ['lat', 'lon', m]].values
        if algo == 'lgb':
            p = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=0)
            if params: p.update(params)
            mdl = lgb.LGBMRegressor(**p)
        elif algo == 'xgb':
            import xgboost as xgb
            mdl = xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=6, min_child_weight=5,
                                   subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=0, n_jobs=8)
        else:
            mdl = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.5, n_jobs=8, random_state=0)
        mdl.fit(X, y); E[:, j] = np.maximum(mdl.predict(Xt), 0)
    return E, TE

def tune_lgb(P, train):
    """Small CV grid (year-grouped, 3 folds) on training years; pooled over 4 representative models."""
    grid = [dict(num_leaves=15, min_child_samples=80), dict(num_leaves=31, min_child_samples=40),
            dict(num_leaves=63, min_child_samples=20)]
    yrs = np.array(sorted(train)); folds = np.array_split(yrs, 3)
    best, bscore = None, 1e9
    for p in grid:
        sc = []
        for f in folds:
            tr = [y for y in train if y not in f]
            TR = P[P.year.isin(tr)]; VA = P[P.year.isin(f)]
            for m in ['epic-boku', 'gepic', 'pdssat', 'lpj-guess', 'pegasus']:
                mdl = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, subsample=0.8, subsample_freq=1,
                                        colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=0, **p)
                mdl.fit(TR[WX + SOIL + ['lat', 'lon', m]].values, np.abs(TR.obs_d - TR[m]).values)
                sc.append(rmse(mdl.predict(VA[WX + SOIL + ['lat', 'lon', m]].values), np.abs(VA.obs_d - VA[m])))
        s = np.mean(sc); print('  lgb grid', p, round(s, 4))
        if s < bscore: best, bscore = p, s
    return best

def softmax_w(E, tau):
    z = -tau * (E - E.min(1, keepdims=True)); w = np.exp(z); return w / w.sum(1, keepdims=True)

def tune_tau(P, train, algo, params, val=None):
    """Validation = last 4 training years (or a given list); grid over tau."""
    yrs = sorted(train); val = val or yrs[-4:]; tr = [y for y in yrs if y not in val]
    E, VA = fit_predict(P, tr, val, algo, params); Y = VA[M19].values
    taus = np.concatenate([np.arange(0.1, 1, 0.1), np.arange(1, 10.5, 0.5)])
    sc = [rmse((softmax_w(E, t) * Y).sum(1), VA.obs_d) for t in taus]
    return float(taus[int(np.argmin(sc))]), dict(zip(np.round(taus, 2).tolist(), np.round(sc, 4).tolist()))

# --------------------------------------------------------------- 4. tests
def run_test(D, train, test, tag, lgb_params, val=None):
    P = prep(D, train); TR = P[P.year.isin(train)]; TE = P[P.year.isin(test)].reset_index(drop=True)
    Y = TE[M19].values; obs = TE.obs_d.values
    res = {}
    res['MME mean'] = Y.mean(1); res['MME median'] = np.median(Y, 1)
    single = {m: rmse(TE[m], obs) for m in M19}; kbest = min(single, key=single.get)
    res['Single best model'] = TE[kbest].values
    # best model per cell (training years)
    bp = TR.groupby(['lat', 'lon']).apply(lambda g: min(M19, key=lambda m: rmse(g[m], g.obs_d)))
    res['Best model per grid cell'] = np.array([TE.loc[i, bp[(TE.lat[i], TE.lon[i])]] for i in range(len(TE))])
    # group average (code-sharing groups) and Knutti similarity weighting
    grp = {m: ('EPIC' if m.startswith('epic') else 'GEPIC' if m.startswith('gepic') else 'LPJ' if m.startswith('lpj') else m) for m in M19}
    res['Group average'] = TE[M19].T.groupby(lambda m: grp[m]).mean().T.mean(1).values
    Dm = np.sqrt(((Y[:, :, None] - Y[:, None, :]) ** 2).mean(0)); off = ~np.eye(19, dtype=bool)
    sig = Dm[off].mean(); wk = 1 / (1 + (np.exp(-(Dm / sig) ** 2) * off).sum(1)); wk /= wk.sum()
    res['Similarity weighting'] = Y @ wk
    weights = {}
    for algo in ['rf', 'xgb', 'lgb']:
        tau, curve = tune_tau(P, train, algo, lgb_params, val)
        E, _ = fit_predict(P, train, test, algo, lgb_params); W = softmax_w(E, tau)
        res[f'Condition-aware ({algo.upper()})'] = (W * Y).sum(1); weights[algo] = W
        print(f'  [{tag}] {algo}: tau={tau}', flush=True)
    tab = pd.DataFrame({k: {'RMSE': rmse(v, obs), **{f'RMSE_{y}': rmse(v[TE.year == y], obs[TE.year == y]) for y in test}}
                        for k, v in res.items()}).T
    tab['vs_mean_%'] = 100 * (tab.RMSE / tab.loc['MME mean', 'RMSE'] - 1)
    tab['vs_median_%'] = 100 * (tab.RMSE / tab.loc['MME median', 'RMSE'] - 1)
    tab.loc['Single best model', 'model'] = kbest
    for y in test:
        sy = {m: rmse(TE.loc[TE.year == y, m], obs[TE.year == y]) for m in M19}
        tab.loc['Single best model', f'model_{y}'] = min(sy, key=sy.get)
    tab.to_csv(os.path.join(OUT, f'table_{tag}.csv'))
    pd.Series(single).sort_values().to_csv(os.path.join(OUT, f'single_models_{tag}.csv'))
    pred = TE[['lat', 'lon', 'year', 'obs_d']].copy()
    for k, v in res.items(): pred[k] = v
    pred.to_csv(os.path.join(OUT, f'predictions_{tag}.csv'), index=False)
    Wd = TE[['lat', 'lon', 'year']].copy()
    for j, m in enumerate(M19): Wd[m] = weights['lgb'][:, j]
    Wd.to_csv(os.path.join(OUT, f'weights_lgb_{tag}.csv'), index=False)
    print(tab[['RMSE', 'vs_mean_%', 'vs_median_%']].round(3))
    return tab

if __name__ == '__main__':
    D = build_panel(); D.to_csv(os.path.join(OUT, 'panel.csv'), index=False)
    yrs = list(range(Y0, Y1 + 1))
    train = [y for y in yrs if y <= TRAIN_END]; test = [y for y in yrs if y >= TEST0]
    P0 = prep(D, train); lgb_params = tune_lgb(P0, train); print('lgb params', lgb_params)
    json.dump(lgb_params, open(os.path.join(OUT, 'lgb_params.json'), 'w'))
    if not os.path.exists(os.path.join(OUT, 'table_baseline.csv')): run_test(D, train, test, 'baseline', lgb_params)
    # leave-one-drought-year-out: the other drought year is kept in training and in the tau-validation set
    for t in DROUGHT:
        other = [y for y in DROUGHT if y != t][0]; train_t = [y for y in yrs if y != t]
        run_test(D, train_t, [t], f'd{t}', lgb_params, val=[other] + [y for y in train_t if y >= 2006])
