"""Extract GDHY rainfed/all maize yields (t/ha) for the CONUS domain used in the paper (east of 104.5 W, 1981-2010)."""
import sys, os, glob, numpy as np, pandas as pd, netCDF4
src = sys.argv[1]; out = sys.argv[2]
rows = []
for f in sorted(glob.glob(os.path.join(src, "yield_*.nc4"))):
    yr = int(os.path.basename(f)[6:10])
    if yr < 1981 or yr > 2010: continue
    d = netCDF4.Dataset(f); v = d.variables["var"][:]; lat = d.variables["lat"][:]; lon = d.variables["lon"][:]
    lon = np.where(lon > 180, lon - 360, lon)
    a = np.ma.filled(v.astype(float), np.nan)
    ii, jj = np.where(np.isfinite(a))
    for i, j in zip(ii, jj):
        la, lo = float(lat[i]), float(lon[j])
        if 24 <= la <= 50 and -104.5 <= lo <= -66: rows.append((yr, la, lo, float(a[i, j])))
    d.close()
df = pd.DataFrame(rows, columns=["year", "lat", "lon", "obs_yield"])
df.to_csv(out, index=False); print(df.year.nunique(), "years;", df.groupby(["lat", "lon"]).ngroups, "cells;", len(df), "rows")
