"""Extract ISIMIP2a rainfed maize yields (t/ha) at the GDHY CONUS cells, 1981-2010, for all downloaded files."""
import sys, os, glob, numpy as np, pandas as pd, netCDF4
src, gdhy, out = sys.argv[1:4]
cells = pd.read_csv(gdhy)[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
frames = []
for f in sorted(glob.glob(os.path.join(src, "*.nc4"))):
    base = os.path.basename(f); model = base.split("_watch")[0]; harm = base.split("noirr-")[1].split("_")[0]
    d = netCDF4.Dataset(f); v = [k for k in d.variables if k not in ("lon", "lat", "time")][0]
    t = d.variables["time"]; u = t.units; y0 = int(u.split("since")[1].strip()[:4]); years = y0 + np.round(np.array(t[:], dtype=float)).astype(int)
    lat = np.array(d.variables["lat"][:]); lon = np.array(d.variables["lon"][:]); lon = np.where(lon > 180, lon - 360, lon)
    ii = np.array([int(np.argmin(np.abs(lat - la))) for la in cells.lat]); jj = np.array([int(np.argmin(np.abs(lon - lo))) for lo in cells.lon])
    arr = d.variables[v]
    for k, yr in enumerate(years):
        if yr < 1981 or yr > 2010: continue
        a = np.ma.filled(arr[k, :, :].astype(float), np.nan)
        vals = a[ii, jj]
        frames.append(pd.DataFrame({"model": model, "harm": harm, "year": yr, "lat": cells.lat, "lon": cells.lon, "sim": vals}))
    d.close(); print(model, harm, years.min(), years.max(), flush=True)
df = pd.concat(frames); df.to_csv(out, index=False); print(len(df), "rows")
