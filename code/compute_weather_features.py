"""Growing-season (1 May-30 Sep) weather predictors per 0.5-deg cell and year
from WATCH-WFDEI daily CONUS cutouts (data/wfdei/*.nc4). Output: data/weather_features_conus_1981_2010.csv
Columns: year, lat, lon, tmean, tmax, tmin, dtr, gdd, edd, hot_days, pr_sum, pr_julaug, dry_spell, rsds, huss, vpd
(pr_anom = pr_sum anomaly vs. cell mean is computed later in the benchmark script, using training years only)."""
import glob, os, sys, datetime as dt
import numpy as np, pandas as pd, netCDF4 as nc
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'wfdei')
OUT = os.path.join(ROOT, '..', 'weather_features_conus_1981_2010.csv')
VARS = ['tas','tasmax','tasmin','pr','rsds','huss','ps']
rows = []
for dec in ['1981_1990','1991_2000','2001_2010']:
    D = {}
    for v in VARS:
        f = glob.glob(os.path.join(ROOT, f'{v}_watch-wfdei_{dec}_*.nc4'))[0]
        ds = nc.Dataset(f)
        if v == 'tas':
            t = ds.variables['time'][:].astype(float)
            dates = pd.to_datetime(dt.datetime(1860,1,1)) + pd.to_timedelta(t, unit='D')
            lat = ds.variables['lat'][:].astype(float); lon = ds.variables['lon'][:].astype(float)
        a = ds.variables[v][:]
        a = np.ma.filled(a.astype('float32'), np.nan)
        D[v] = a; ds.close()
    years = dates.year.values; mo = dates.month.values
    for y in np.unique(years):
        s = (years == y) & (mo >= 5) & (mo <= 9)
        ja = (years == y) & ((mo == 7) | (mo == 8))
        T = D['tas'][s] - 273.15; Tx = D['tasmax'][s] - 273.15; Tn = D['tasmin'][s] - 273.15
        P = D['pr'][s] * 86400.0  # mm/day
        tmean = T.mean(0); tmax = Tx.mean(0); tmin = Tn.mean(0); dtr = (Tx - Tn).mean(0)
        gdd = np.clip(T, 10, 30).sum(0) - 10 * T.shape[0]
        edd = np.clip(T - 30, 0, None).sum(0)
        hot = (Tx >= 30).sum(0).astype(float)
        pr_sum = P.sum(0); pr_ja = (D['pr'][ja] * 86400.0).sum(0)
        # longest run of days with < 1 mm
        dry = P < 1.0
        run = np.zeros(dry.shape[1:], dtype=float); best = np.zeros_like(run)
        for d in range(dry.shape[0]):
            run = np.where(dry[d], run + 1, 0); best = np.maximum(best, run)
        rsds = D['rsds'][s].mean(0); q = D['huss'][s]; p = D['ps'][s]
        es = 0.6108 * np.exp(17.27 * T / (T + 237.3))          # kPa
        ea = q * p / (0.622 + 0.378 * q) / 1000.0              # kPa
        vpd = np.clip(es - ea, 0, None).mean(0); huss = q.mean(0)
        LA, LO = np.meshgrid(lat, lon, indexing='ij')
        df = pd.DataFrame({'year': y, 'lat': LA.ravel(), 'lon': LO.ravel(),
            'tmean': tmean.ravel(), 'tmax': tmax.ravel(), 'tmin': tmin.ravel(), 'dtr': dtr.ravel(),
            'gdd': gdd.ravel(), 'edd': edd.ravel(), 'hot_days': hot.ravel(), 'pr_sum': pr_sum.ravel(),
            'pr_julaug': pr_ja.ravel(), 'dry_spell': best.ravel(), 'rsds': rsds.ravel(),
            'huss': huss.ravel(), 'vpd': vpd.ravel()})
        rows.append(df.dropna(subset=['tmean']))
        print(y, len(rows[-1]), flush=True)
    del D
out = pd.concat(rows); out.to_csv(OUT, index=False, float_format='%.4f')
print('wrote', OUT, out.shape)
