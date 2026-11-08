"""Topsoil (0-30 cm) HWSD v1.2 predictors for CONUS cells from the ISIMIP3 soil input
(hwsd_soil_data_all_land.nc; Volkholz & Mueller 2020). Output: data/soil_conus.csv"""
import os, numpy as np, pandas as pd, netCDF4 as nc
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
d = nc.Dataset(os.path.join(R, 'soil', 'hwsd_soil_data_all_land.nc'))
lat = d.variables['lat'][:]; lon = d.variables['lon'][:]
ii = np.where((lat >= 24) & (lat <= 50))[0]; jj = np.where((lon >= -105) & (lon <= -66))[0]
V = {'sand':'sand','silt':'silt','clay':'clay','oc':'oc','soil_ph':'ph','bulk_density':'bulk_density','awc':'awc','cec_soil':'cec'}
LA, LO = np.meshgrid(lat[ii], lon[jj], indexing='ij')
out = pd.DataFrame({'lat': LA.ravel(), 'lon': LO.ravel()})
for k, n in V.items():
    a = d.variables[k][ii][:, jj]; out[n] = np.ma.filled(a.astype(float), np.nan).ravel()
out = out.dropna(subset=['sand']); out.to_csv(os.path.join(R, 'soil_conus.csv'), index=False, float_format='%.3f')
print(out.shape); print(out.describe().T[['mean','min','max']].round(2))
